#pragma once

#include "tron2_controllers/ControllerBase.h"

#include <memory>
#include <vector>

#include <geometry_msgs/PoseStamped.h>
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/jntarray.hpp>
#include <tron2_controllers/EEMotionCommand.h>
#include <tron2_controllers/WholeBodyPoseCommand.h>

namespace tron2_controller
{

class WFYGController final : public ControllerBase
{
public:
    bool init(hardware_interface::RobotHW *robot_hw, ros::NodeHandle &nh) override;
    void starting(const ros::Time &time) override;

protected:
    void computeObservation() override;
    void applyActions() override;
    void resetRuntimeState() override;

private:
    bool initializeKinematics();
    bool loadJointLimits();
    bool computeEndEffectorPose(Eigen::Vector3d &position, Eigen::Quaterniond &orientation);
    void poseCommandCallback(const tron2_controllers::WholeBodyPoseCommand::ConstPtr &message);
    void motionCommandCallback(const tron2_controllers::EEMotionCommand::ConstPtr &message);
    void writeRotation6D(Eigen::VectorXf &target, int offset, const Eigen::Matrix3d &rotation) const;
    Eigen::Vector3d orientationAxisAngleError(
        const Eigen::Quaterniond &current, const Eigen::Quaterniond &target) const;
    double torqueSafePosition(size_t jointIndex, double requested) const;
    double torqueSafeVelocity(size_t jointIndex, double requested) const;

    KDL::Chain fkChain_;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fkSolver_;
    std::vector<int> fkJointHandleIndices_;
    std::vector<double> softLowerLimits_;
    std::vector<double> softUpperLimits_;
    Eigen::VectorXd filteredPositionTargets_;
    bool actionFilterInitialized_{false};

    Eigen::Vector3d eeTargetPosition_{0.4209992, 0.0, 0.4273805};
    Eigen::Quaterniond eeTargetOrientation_{0.7070451, 0.0, 0.7071684, 0.0};

    ros::Subscriber poseCommandSub_;
    ros::Subscriber motionCommandSub_;
    ros::Publisher eePosePub_;
};

} // namespace tron2_controller
