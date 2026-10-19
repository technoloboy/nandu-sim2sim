#include "tron2_controllers/WFYGController.h"
#include "tron2_controllers/WFYGMath.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <kdl_parser/kdl_parser.hpp>
#include <pluginlib/class_list_macros.h>
#include <urdf/model.h>

namespace tron2_controller
{

bool WFYGController::init(hardware_interface::RobotHW *robot_hw, ros::NodeHandle &nh)
{
    if (!ControllerBase::init(robot_hw, nh))
    {
        return false;
    }
    if (cfg_.jointNames.size() != 18 || (cfg_.actionsSize != 10 && cfg_.actionsSize != 16))
    {
        ROS_ERROR("WFYG requires 18 hardware joints and 10/16 policy actions");
        return false;
    }
    if (!initializeKinematics() || !loadJointLimits())
    {
        return false;
    }
    filteredPositionTargets_.setZero(cfg_.jointNames.size());
    eeTargetOrientation_.normalize();
    if (cfg_.runtimeProfile == "wholebody_pose")
    {
        poseCommandSub_ = nh.subscribe("/tron2_controller/wholebody_pose_command", 1, &WFYGController::poseCommandCallback, this);
    }
    else if (cfg_.runtimeProfile == "wholebody_motion" || cfg_.runtimeProfile == "height_motion")
    {
        motionCommandSub_ = nh.subscribe("/tron2_controller/ee_motion_command", 1, &WFYGController::motionCommandCallback, this);
    }
    eePosePub_ = nh.advertise<geometry_msgs::PoseStamped>("/tron2_controller/ee_pose", 1);
    ROS_INFO("WFYG controller registered task=%s runtime=%s decoder=%s asset=%s",
             cfg_.taskId.c_str(), cfg_.runtimeProfile.c_str(), cfg_.actionDecoder.c_str(), cfg_.assetProfile.c_str());
    return true;
}

void WFYGController::starting(const ros::Time &time)
{
    ControllerBase::starting(time);
}

void WFYGController::resetRuntimeState()
{
    actionFilterInitialized_ = false;
}

bool WFYGController::initializeKinematics()
{
    KDL::Tree tree;
    if (!kdl_parser::treeFromParam("robot_description", tree) ||
        !tree.getChain("arm_base_Link", "gripper_base_Link", fkChain_))
    {
        ROS_ERROR("Cannot build WFYG FK chain arm_base_Link -> gripper_base_Link");
        return false;
    }
    fkJointHandleIndices_.clear();
    for (unsigned int segmentIndex = 0; segmentIndex < fkChain_.getNrOfSegments(); ++segmentIndex)
    {
        const auto &joint = fkChain_.getSegment(segmentIndex).getJoint();
        if (joint.getType() == KDL::Joint::None)
        {
            continue;
        }
        const auto found = std::find(cfg_.jointNames.begin(), cfg_.jointNames.end(), joint.getName());
        if (found == cfg_.jointNames.end())
        {
            ROS_ERROR("FK joint %s is absent from WFYG hardware order", joint.getName().c_str());
            return false;
        }
        fkJointHandleIndices_.push_back(static_cast<int>(std::distance(cfg_.jointNames.begin(), found)));
    }
    if (fkJointHandleIndices_.size() != fkChain_.getNrOfJoints())
    {
        ROS_ERROR("WFYG FK joint mapping is incomplete");
        return false;
    }
    fkSolver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(fkChain_);
    return true;
}

bool WFYGController::loadJointLimits()
{
    urdf::Model model;
    if (!model.initParam("robot_description"))
    {
        ROS_ERROR("Cannot load robot_description for WFYG joint limits");
        return false;
    }
    softLowerLimits_.resize(cfg_.jointNames.size(), 0.0);
    softUpperLimits_.resize(cfg_.jointNames.size(), 0.0);
    for (size_t index = 0; index < cfg_.jointNames.size(); ++index)
    {
        const auto joint = model.getJoint(cfg_.jointNames[index]);
        if (!joint || !joint->limits)
        {
            if (cfg_.jointNames[index].find("wheel_") != std::string::npos)
            {
                softLowerLimits_[index] = -1.0e9;
                softUpperLimits_[index] = 1.0e9;
                continue;
            }
            ROS_ERROR("Missing URDF limits for %s", cfg_.jointNames[index].c_str());
            return false;
        }
        const auto limits = wfyg_math::softLimits(joint->limits->lower, joint->limits->upper);
        softLowerLimits_[index] = limits.first;
        softUpperLimits_[index] = limits.second;
    }
    return true;
}

bool WFYGController::computeEndEffectorPose(Eigen::Vector3d &position, Eigen::Quaterniond &orientation)
{
    KDL::JntArray joints(fkChain_.getNrOfJoints());
    for (unsigned int index = 0; index < joints.rows(); ++index)
    {
        joints(index) = hybridJointHandles_[fkJointHandleIndices_[index]].getPosition();
    }
    KDL::Frame frame;
    if (fkSolver_->JntToCart(joints, frame) < 0)
    {
        return false;
    }
    position = Eigen::Vector3d(frame.p.x(), frame.p.y(), frame.p.z());
    double x, y, z, w;
    frame.M.GetQuaternion(x, y, z, w);
    orientation = Eigen::Quaterniond(w, x, y, z).normalized();
    return position.allFinite() && orientation.coeffs().allFinite();
}

void WFYGController::writeRotation6D(Eigen::VectorXf &target, int offset, const Eigen::Matrix3d &rotation) const
{
    target.segment<6>(offset) = wfyg_math::rotation6D(rotation);
}

Eigen::Vector3d WFYGController::orientationAxisAngleError(
    const Eigen::Quaterniond &current, const Eigen::Quaterniond &target) const
{
    return wfyg_math::orientationAxisAngleError(current, target);
}

void WFYGController::computeObservation()
{
    const double *quat = imuSensorHandle_.getOrientation();
    const double *gyro = imuSensorHandle_.getAngularVelocity();
    Eigen::Quaterniond baseOrientation(quat[3], quat[0], quat[1], quat[2]);
    if (!baseOrientation.coeffs().allFinite() || baseOrientation.norm() < 1.0e-8)
    {
        throw std::runtime_error("invalid IMU quaternion");
    }
    baseOrientation.normalize();
    const Eigen::Vector3f angularVelocity(
        static_cast<float>(gyro[0] * 0.25),
        static_cast<float>(gyro[1] * 0.25),
        static_cast<float>(gyro[2] * 0.25));
    const Eigen::Vector3f projectedGravity =
        (baseOrientation.toRotationMatrix().transpose() * Eigen::Vector3d(0.0, 0.0, -1.0)).cast<float>();

    static const std::vector<int> kLegPositionIndices = {0, 1, 2, 3, 5, 6, 7, 8};
    static const std::vector<int> kWholePositionIndices = {0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15};
    const auto &positionIndices = cfg_.actionsSize == 10 ? kLegPositionIndices : kWholePositionIndices;
    int offset = 0;
    observations_.segment<3>(offset) = angularVelocity;
    offset += 3;
    observations_.segment<3>(offset) = projectedGravity;
    offset += 3;
    for (int jointIndex : positionIndices)
    {
        observations_(offset++) = static_cast<float>(
            hybridJointHandles_[jointIndex].getPosition() - cfg_.defaultDofPos[jointIndex]);
    }
    for (int jointIndex = 0; jointIndex < cfg_.actionsSize; ++jointIndex)
    {
        observations_(offset++) = static_cast<float>(hybridJointHandles_[jointIndex].getVelocity() * 0.05);
    }

    Eigen::Vector3d eePosition;
    Eigen::Quaterniond eeOrientation;
    if (cfg_.runtimeProfile == "wholebody_pose")
    {
        if (!computeEndEffectorPose(eePosition, eeOrientation))
        {
            throw std::runtime_error("WFYG forward kinematics failed");
        }
        observations_.segment<3>(offset) = eePosition.cast<float>();
        offset += 3;
        writeRotation6D(observations_, offset, eeOrientation.toRotationMatrix());
        offset += 6;
        observations_.segment<3>(offset) = (eeTargetPosition_ - eePosition).cast<float>();
        offset += 3;
        observations_.segment<3>(offset) = orientationAxisAngleError(eeOrientation, eeTargetOrientation_).cast<float>();
        offset += 3;
    }
    observations_.segment(offset, cfg_.actionsSize) = lastActions_;
    offset += cfg_.actionsSize;
    if (offset != cfg_.policyObsSize)
    {
        throw std::runtime_error("WFYG observation layout does not close at the registered dimension");
    }
    for (int index = 0; index < observations_.size(); ++index)
    {
        observations_(index) = std::clamp(
            observations_(index), static_cast<float>(-cfg_.clipObs), static_cast<float>(cfg_.clipObs));
    }

    if (computeEndEffectorPose(eePosition, eeOrientation))
    {
        geometry_msgs::PoseStamped message;
        message.header.stamp = ros::Time::now();
        message.header.frame_id = "arm_base_Link";
        message.pose.position.x = eePosition.x();
        message.pose.position.y = eePosition.y();
        message.pose.position.z = eePosition.z();
        message.pose.orientation.x = eeOrientation.x();
        message.pose.orientation.y = eeOrientation.y();
        message.pose.orientation.z = eeOrientation.z();
        message.pose.orientation.w = eeOrientation.w();
        eePosePub_.publish(message);
    }
}

double WFYGController::torqueSafePosition(size_t jointIndex, double requested) const
{
    const double kp = cfg_.kp[jointIndex];
    if (kp <= 1.0e-8)
    {
        return requested;
    }
    const double q = hybridJointHandles_[jointIndex].getPosition();
    const double dq = hybridJointHandles_[jointIndex].getVelocity();
    const double kd = cfg_.kd[jointIndex];
    const double limit = cfg_.torqueLimits[jointIndex];
    return std::clamp(requested, q + (kd * dq - limit) / kp, q + (kd * dq + limit) / kp);
}

double WFYGController::torqueSafeVelocity(size_t jointIndex, double requested) const
{
    const double kd = cfg_.kd[jointIndex];
    if (kd <= 1.0e-8)
    {
        return requested;
    }
    const double dq = hybridJointHandles_[jointIndex].getVelocity();
    const double limit = cfg_.torqueLimits[jointIndex];
    return std::clamp(requested, dq - limit / kd, dq + limit / kd);
}

void WFYGController::applyActions()
{
    if (!actionFilterInitialized_)
    {
        for (size_t index = 0; index < hybridJointHandles_.size(); ++index)
        {
            filteredPositionTargets_(index) = hybridJointHandles_[index].getPosition();
        }
        actionFilterInitialized_ = true;
    }

    const bool heightDecoder = cfg_.actionDecoder == "height_v1_v2";
    const double heightScales[8] = {0.55, 0.25, 0.25, 1.0, 0.55, 0.25, 0.25, 1.0};
    int legPositionAction = 0;
    for (size_t index = 0; index < 10; ++index)
    {
        if (cfg_.jointNames[index].find("wheel_") != std::string::npos)
        {
            const double velocity = torqueSafeVelocity(index, actions_(index));
            hybridJointHandles_[index].setCommand(0.0, velocity, 0.0, cfg_.kd[index], 0.0, 0);
            continue;
        }
        const double scale = heightDecoder ? heightScales[legPositionAction] : 0.25;
        double target = cfg_.defaultDofPos[index] + scale * static_cast<double>(actions_(index));
        if (heightDecoder)
        {
            const double margin = 0.02 * (softUpperLimits_[index] - softLowerLimits_[index]);
            target = std::clamp(target, softLowerLimits_[index] + margin, softUpperLimits_[index] - margin);
            target = wfyg_math::ema(target, filteredPositionTargets_(index));
            target = std::clamp(target, softLowerLimits_[index] + margin, softUpperLimits_[index] - margin);
            filteredPositionTargets_(index) = target;
        }
        target = torqueSafePosition(index, target);
        hybridJointHandles_[index].setCommand(target, 0.0, cfg_.kp[index], cfg_.kd[index], 0.0, 0);
        ++legPositionAction;
    }

    if (cfg_.actionsSize == 16)
    {
        for (size_t index = 10; index < 16; ++index)
        {
            double target = wfyg_math::unscaleToLimits(
                static_cast<double>(actions_(index)), softLowerLimits_[index], softUpperLimits_[index]);
            target = wfyg_math::ema(target, filteredPositionTargets_(index));
            target = std::clamp(target, softLowerLimits_[index], softUpperLimits_[index]);
            filteredPositionTargets_(index) = target;
            target = torqueSafePosition(index, target);
            hybridJointHandles_[index].setCommand(target, 0.0, cfg_.kp[index], cfg_.kd[index], 0.0, 0);
        }
    }
    else
    {
        for (size_t index = 10; index < 16; ++index)
        {
            hybridJointHandles_[index].setCommand(
                cfg_.defaultDofPos[index], 0.0, 800.0, 40.0, 0.0, 0);
        }
    }
    for (size_t index = 16; index < 18; ++index)
    {
        hybridJointHandles_[index].setCommand(
            cfg_.defaultDofPos[index], 0.0, cfg_.kp[index], cfg_.kd[index], 0.0, 0);
    }
}

void WFYGController::poseCommandCallback(const tron2_controllers::WholeBodyPoseCommand::ConstPtr &message)
{
    if (!message->header.frame_id.empty() && message->header.frame_id != "arm_base_Link")
    {
        enterFault("WholeBodyPoseCommand frame must be arm_base_Link");
        return;
    }
    const auto &p = message->ee_pose.position;
    const auto &q = message->ee_pose.orientation;
    Eigen::Quaterniond orientation(q.w, q.x, q.y, q.z);
    Eigen::Vector3d position(p.x, p.y, p.z);
    if (!position.allFinite() || !orientation.coeffs().allFinite() || orientation.norm() < 1.0e-8)
    {
        enterFault("WholeBodyPoseCommand contains invalid pose values");
        return;
    }
    orientation.normalize();
    position.x() = std::clamp(position.x(), 0.25, 0.58);
    position.y() = std::clamp(position.y(), -0.22, 0.22);
    position.z() = std::clamp(position.z(), 0.18, 0.62);
    eeTargetPosition_ = position;
    eeTargetOrientation_ = orientation;
    policyCommands_(0) = static_cast<float>(std::clamp(message->base_velocity.linear.x, -0.4, 0.6));
    policyCommands_(1) = 0.0f;
    policyCommands_(2) = static_cast<float>(std::clamp(message->base_velocity.angular.z, -1.0, 1.0));
    policyCommands_.segment<3>(3) = position.cast<float>();
    writeRotation6D(policyCommands_, 6, orientation.toRotationMatrix());
    commandReceived_ = true;
    lastCommandTime_ = ros::Time::now();
}

void WFYGController::motionCommandCallback(const tron2_controllers::EEMotionCommand::ConstPtr &message)
{
    const auto &v = message->linear_velocity;
    const auto &q = message->orientation;
    Eigen::Vector3d velocity(v.x, v.y, v.z);
    Eigen::Quaterniond orientation(q.w, q.x, q.y, q.z);
    if (!velocity.allFinite() || !orientation.coeffs().allFinite() || orientation.norm() < 1.0e-8)
    {
        enterFault("EEMotionCommand contains invalid values");
        return;
    }
    orientation.normalize();
    const std::string &frame = message->header.frame_id;
    if (frame == "world" || frame == "odom")
    {
        const double *imu = imuSensorHandle_.getOrientation();
        Eigen::Quaterniond base(imu[3], imu[0], imu[1], imu[2]);
        base.normalize();
        const double yaw = std::atan2(
            2.0 * (base.w() * base.z() + base.x() * base.y()),
            1.0 - 2.0 * (base.y() * base.y() + base.z() * base.z()));
        const Eigen::Quaterniond heading(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
        velocity = heading.conjugate() * velocity;
        orientation = (heading.conjugate() * orientation).normalized();
    }
    else if (!frame.empty() && frame != "base_heading")
    {
        enterFault("EEMotionCommand frame must be base_heading, world, or odom");
        return;
    }
    velocity.x() = std::clamp(velocity.x(), -0.60, 0.60);
    velocity.y() = std::clamp(velocity.y(), -0.25, 0.25);
    velocity.z() = std::clamp(velocity.z(), -0.15, 0.15);
    policyCommands_.head<3>() = velocity.cast<float>();
    writeRotation6D(policyCommands_, 3, orientation.toRotationMatrix());
    commandReceived_ = true;
    lastCommandTime_ = ros::Time::now();
}

} // namespace tron2_controller

PLUGINLIB_EXPORT_CLASS(tron2_controller::WFYGController, controller_interface::ControllerBase)
