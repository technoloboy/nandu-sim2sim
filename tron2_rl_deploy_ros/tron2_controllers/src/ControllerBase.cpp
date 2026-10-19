#include "tron2_controllers/ControllerBase.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <diagnostic_msgs/DiagnosticStatus.h>
#include <diagnostic_msgs/KeyValue.h>
#include <openssl/sha.h>

namespace tron2_controller
{

namespace
{
struct TaskSpec
{
    std::string taskId;
    std::string playAlias;
    std::string runtime;
    std::string decoder;
    int observationSize;
    int commandSize;
    int actionSize;
    int encoderInputSize;
    int policyInputSize;
};

const std::vector<TaskSpec> kWFYGTasks = {
    {"Isaac-Limx-WFYG-TRON2A-Blind-Flat-v0", "Isaac-Limx-WFYG-TRON2A-Blind-Flat-Play-v0", "locomotion", "blind_default_offset", 34, 3, 10, 340, 40},
    {"Isaac-Limx-WFYG-TRON2A-WholeBody-Flat-v0", "Isaac-Limx-WFYG-TRON2A-WholeBody-Flat-Play-v0", "wholebody_pose", "wholebody_v0", 67, 12, 16, 670, 82},
    {"Isaac-Limx-WFYG-TRON2A-WholeBody-Flat-v1", "Isaac-Limx-WFYG-TRON2A-WholeBody-Flat-Play-v1", "wholebody_motion", "wholebody_v1", 52, 9, 16, 520, 64},
    {"Isaac-Limx-WFYG-TRON2A-WholeBody-Flat-Heading-v1", "Isaac-Limx-WFYG-TRON2A-WholeBody-Flat-Heading-Play-v1", "wholebody_motion", "wholebody_v1", 52, 9, 16, 520, 64},
    {"Isaac-Limx-WFYG-TRON2A-WholeBody-Height-Subject-Heading-v1", "Isaac-Limx-WFYG-TRON2A-WholeBody-Height-Subject-Heading-Play-v1", "height_motion", "height_v1_v2", 52, 9, 16, 520, 64},
    {"Isaac-Limx-WFYG-TRON2A-WholeBody-Height-Subject-Heading-v2", "Isaac-Limx-WFYG-TRON2A-WholeBody-Height-Subject-Heading-Play-v2", "height_motion", "height_v1_v2", 52, 9, 16, 520, 64},
};

std::string sha256File(const std::string &path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("Cannot open file for SHA-256: " + path);
    }
    SHA256_CTX context;
    SHA256_Init(&context);
    char buffer[1024 * 1024];
    while (stream.good())
    {
        stream.read(buffer, sizeof(buffer));
        if (stream.gcount() > 0)
        {
            SHA256_Update(&context, buffer, static_cast<size_t>(stream.gcount()));
        }
    }
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256_Final(digest, &context);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned char value : digest)
    {
        output << std::setw(2) << static_cast<unsigned int>(value);
    }
    return output.str();
}

bool shapeMatches(const std::vector<int64_t> &shape, int width)
{
    return shape.size() == 2 && (shape[0] == 1 || shape[0] == -1) && shape[1] == width;
}
} // namespace

ControllerBase::~ControllerBase()
{
    delete policySessionPtr_;
    delete encoderSessionPtr_;
    delete onnxEnvPtr_;
}

bool ControllerBase::init(hardware_interface::RobotHW *robot_hw, ros::NodeHandle &nh)
{
    if (!loadRLCfg(nh))
    {
        ROS_ERROR("Failed to load RL config");
        return false;
    }
    if (!loadModel(nh))
    {
        ROS_ERROR("Failed to load ONNX model");
        return false;
    }

    auto *hybridJointInterface = robot_hw->get<robot_common::HybridJointInterface>();
    for (const auto &name : cfg_.jointNames)
    {
        hybridJointHandles_.push_back(hybridJointInterface->getHandle(name));
    }

    auto *imuInterface = robot_hw->get<hardware_interface::ImuSensorInterface>();
    nh.getParam("imu_name", tron2ImuHandleName_);
    imuSensorHandle_ = imuInterface->getHandle(tron2ImuHandleName_);

    cmdVelSub_ = nh.subscribe("/cmd_vel", 1, &ControllerBase::cmdVelCallback, this);
    modeSub_ = nh.subscribe("/tron2_controller/set_mode", 1, &ControllerBase::modeCallback, this);
    diagnosticsPub_ = nh.advertise<diagnostic_msgs::DiagnosticArray>("/tron2_controller/diagnostics", 1);
    tron2Command_.setZero();
    policyCommands_.setZero(cfg_.commandsObsSize);

    observations_.setZero(cfg_.policyObsSize);
    actions_.setZero(cfg_.actionsSize);
    lastActions_.setZero(cfg_.actionsSize);
    encoderOut_.setZero(cfg_.encoderOutputSize);
    proprioHistoryBuffer_.setZero(cfg_.obsHistoryLength * cfg_.policyObsSize);

    nh.param("policy_dt", tron2PolicyDt_, 0.02);
    nh.param("hw_control_dt", tron2HwControlDt_, 0.002);

    nh.getParam("/Tron2Cfg/stand_mode/stand_duration", standDuration_);
    tron2InitJointAngles_.setZero(cfg_.jointNames.size());

    ROS_INFO("Controller initialized: variant=%s imu=%s, policyDt=%.4f, decimation=%d, policyObsSize=%d, actionsSize=%d",
             robotVariant_.c_str(), tron2ImuHandleName_.c_str(), tron2PolicyDt_, cfg_.decimation, cfg_.policyObsSize, cfg_.actionsSize);

    return true;
}

void ControllerBase::starting(const ros::Time &)
{
    loopCount_ = 0;
    isFirstObs_ = true;
    footGaitPhaseIndex_ = 0.0;
    mode_ = STAND;
    standPercent_ = 0.0;
    for (size_t i = 0; i < hybridJointHandles_.size(); i++)
    {
        tron2InitJointAngles_(i) = hybridJointHandles_[i].getPosition();
    }
    lastActions_.setZero();
    actions_.setZero();
    proprioHistoryBuffer_.setZero();
    tron2Command_.setZero();
    policyCommands_.setZero();
    commandReceived_ = false;
    lastCommandTime_ = ros::Time(0);
    faultReason_.clear();
    resetRuntimeState();
}

void ControllerBase::update(const ros::Time &time, const ros::Duration &period)
{
    const double measuredDt = period.toSec() > 0.0 ? period.toSec() : tron2HwControlDt_;
    tron2HwControlDt_ = measuredDt;
    tron2PolicyDt_ = measuredDt * static_cast<double>(std::max(cfg_.decimation, 1));

    if (mode_ == IDLE || mode_ == FAULT)
    {
        for (size_t i = 0; i < hybridJointHandles_.size(); i++)
        {
            double current_pos = hybridJointHandles_[i].getPosition();
            hybridJointHandles_[i].setCommand(current_pos, 0.0, 0.0, cfg_.kd[i], 0.0, 0);
        }
        publishDiagnostics(time);
        return;
    }

    if (mode_ == STAND)
    {
        if (standPercent_ < 1.0)
        {
            for (size_t i = 0; i < hybridJointHandles_.size(); i++)
            {
                double pos_des = tron2InitJointAngles_(i) * (1.0 - standPercent_) + cfg_.defaultDofPos[i] * standPercent_;
                hybridJointHandles_[i].setCommand(pos_des, 0.0, cfg_.kp[i], cfg_.kd[i], 0.0, 0);
            }
            standPercent_ += measuredDt / std::max(standDuration_, measuredDt);
        }
        else
        {
            for (size_t i = 0; i < hybridJointHandles_.size(); i++)
            {
                hybridJointHandles_[i].setCommand(cfg_.defaultDofPos[i], 0.0, cfg_.kp[i], cfg_.kd[i], 0.0, 0);
            }
        }
        publishDiagnostics(time);
        return;
    }

    if (isWFYGVariant_ && cfg_.faultOnCommandTimeout &&
        (!commandReceived_ || (time - lastCommandTime_).toSec() > cfg_.commandTimeout))
    {
        enterFault("policy command timeout");
        publishDiagnostics(time);
        return;
    }

    loopCount_++;
    if (loopCount_ % cfg_.decimation == 0)
    {
        try
        {
            computeObservation();
            if (!valuesFinite(observations_))
            {
                throw std::runtime_error("observation contains NaN or Inf");
            }
            updateObsHistory();
            computeEncoder();
            computeActions();
            if (!valuesFinite(encoderOut_) || !valuesFinite(actions_))
            {
                throw std::runtime_error("model output contains NaN or Inf");
            }
        }
        catch (const std::exception &error)
        {
            enterFault(error.what());
            publishDiagnostics(time);
            return;
        }

        // Limit action range
        float actionMin = -cfg_.clipActions;
        float actionMax = cfg_.clipActions;
        std::transform(actions_.data(), actions_.data() + actions_.size(), actions_.data(),
                       [actionMin, actionMax](float x) { return std::max(actionMin, std::min(actionMax, x)); });

        lastActions_ = actions_;
    }

    applyActions();
    publishDiagnostics(time);
}

bool ControllerBase::loadRLCfg(ros::NodeHandle &nh)
{
    const std::string ns = "/Tron2Cfg";
    nh.getParam(ns + "/robot_variant", robotVariant_);
    isWFYGVariant_ = (robotVariant_ == "WFYG_TRON2A");
    isWheelfootVariant_ = (robotVariant_.find("WF_") != std::string::npos) ||
                          (robotVariant_.find("WHEELFOOT") != std::string::npos) || isWFYGVariant_;
    isSolefootVariant_ = (robotVariant_.find("SF_") != std::string::npos) ||
                         (robotVariant_.find("SOLEFOOT") != std::string::npos);
    nh.getParam(ns + "/init_state/joint_names", cfg_.jointNames);
    if (cfg_.jointNames.empty())
    {
        ROS_ERROR("No joint_names found in %s", (ns + "/init_state/joint_names").c_str());
        return false;
    }

    cfg_.sensorJointNames = cfg_.jointNames;
    nh.getParam(ns + "/init_state/sensor_joint_names", cfg_.sensorJointNames);

    const int numJoints = static_cast<int>(cfg_.jointNames.size());
    cfg_.defaultDofPos.resize(numJoints, 0.0);
    cfg_.defaultDofPosSensor.resize(numJoints, 0.0);

    for (int i = 0; i < numJoints; i++)
    {
        nh.getParam(ns + "/init_state/default_joint_angle/" + cfg_.jointNames[i], cfg_.defaultDofPos[i]);
        nh.getParam(ns + "/init_state/default_joint_angle/" + cfg_.sensorJointNames[i], cfg_.defaultDofPosSensor[i]);
    }

    nh.getParam(ns + "/control/joint_control_modes", cfg_.jointModes);

    double legStiffness = 0.0, legDamping = 0.0, lightStiffness = 0.0, lightDamping = 0.0;
    double legTorqueLimit = 0.0, lightTorqueLimit = 0.0;
    double wheelDamping = 0.0, wheelTorqueLimit = 0.0;
    double armStiffness = 80.0, armDamping = 4.0, armTorqueLimit = 100.0;
    double gripperStiffness = 800.0, gripperDamping = 40.0, gripperTorqueLimit = 20.0;
    nh.param(ns + "/control/leg_joint_stiffness", legStiffness, 159.67);
    nh.param(ns + "/control/leg_joint_damping", legDamping, 10.16);
    nh.param(ns + "/control/light_joint_stiffness", lightStiffness, 53.22);
    nh.param(ns + "/control/light_joint_damping", lightDamping, 3.39);
    nh.param(ns + "/control/leg_joint_torque_limit", legTorqueLimit, 140.0);
    nh.param(ns + "/control/light_joint_torque_limit", lightTorqueLimit, 40.0);
    nh.param(ns + "/control/wheel_joint_damping", wheelDamping, 0.6);
    nh.param(ns + "/control/wheel_joint_torque_limit", wheelTorqueLimit, 20.0);
    nh.param(ns + "/control/arm_joint_stiffness", armStiffness, 80.0);
    nh.param(ns + "/control/arm_joint_damping", armDamping, 4.0);
    nh.param(ns + "/control/arm_joint_torque_limit", armTorqueLimit, 100.0);
    nh.param(ns + "/control/gripper_joint_stiffness", gripperStiffness, 800.0);
    nh.param(ns + "/control/gripper_joint_damping", gripperDamping, 40.0);
    nh.param(ns + "/control/gripper_joint_torque_limit", gripperTorqueLimit, 20.0);

    cfg_.kp.resize(numJoints, 0.0);
    cfg_.kd.resize(numJoints, 0.0);
    cfg_.torqueLimits.resize(numJoints, 0.0);
    for (int i = 0; i < numJoints; i++)
    {
        const std::string &name = cfg_.jointNames[i];
        const bool isWheel = name.find("wheel_") != std::string::npos;
        const bool isYaw = name.find("yaw_") != std::string::npos;
        const bool isAnklePitch = name.find("ankle_pitch_") != std::string::npos;
        const bool isArm = name.find("arm") == 0;
        const bool isGripper = name.find("gripper") == 0;
        if (isWheel)
        {
            cfg_.kp[i] = 0.0;
            cfg_.kd[i] = wheelDamping;
            cfg_.torqueLimits[i] = wheelTorqueLimit;
        }
        else if (isArm)
        {
            cfg_.kp[i] = armStiffness;
            cfg_.kd[i] = armDamping;
            cfg_.torqueLimits[i] = armTorqueLimit;
        }
        else if (isGripper)
        {
            cfg_.kp[i] = gripperStiffness;
            cfg_.kd[i] = gripperDamping;
            cfg_.torqueLimits[i] = gripperTorqueLimit;
        }
        else if (isYaw || isAnklePitch)
        {
            cfg_.kp[i] = lightStiffness;
            cfg_.kd[i] = lightDamping;
            cfg_.torqueLimits[i] = lightTorqueLimit;
        }
        else
        {
            cfg_.kp[i] = legStiffness;
            cfg_.kd[i] = legDamping;
            cfg_.torqueLimits[i] = legTorqueLimit;
        }
    }

    nh.getParam(ns + "/control/action_scale_pos", cfg_.actionScalePos);
    nh.getParam(ns + "/control/action_scale_vel", cfg_.actionScaleVel);
    nh.getParam(ns + "/control/decimation", cfg_.decimation);
    nh.getParam(ns + "/normalization/clip_scales/clip_observations", cfg_.clipObs);
    nh.getParam(ns + "/normalization/clip_scales/clip_actions", cfg_.clipActions);

    nh.getParam(ns + "/size/actions_size", cfg_.actionsSize);
    nh.getParam(ns + "/size/policy_obs_size", cfg_.policyObsSize);
    nh.getParam(ns + "/size/commands_obs_size", cfg_.commandsObsSize);
    nh.getParam(ns + "/size/encoder_input_size", cfg_.encoderInputSize);
    nh.getParam(ns + "/size/encoder_output_size", cfg_.encoderOutputSize);
    nh.getParam(ns + "/size/policy_input_size", cfg_.policyInputSize);
    nh.getParam(ns + "/size/obs_history_length", cfg_.obsHistoryLength);

    nh.param(ns + "/safety/command_timeout", cfg_.commandTimeout, 0.5);
    nh.param(ns + "/safety/fault_on_command_timeout", cfg_.faultOnCommandTimeout, false);

    if (isWFYGVariant_ && !loadBundleContract(nh))
    {
        return false;
    }

    if (cfg_.jointModes.size() != cfg_.jointNames.size())
    {
        ROS_ERROR("joint_control_modes has %zu entries, expected %zu", cfg_.jointModes.size(), cfg_.jointNames.size());
        return false;
    }

    nh.getParam(ns + "/gait/enabled", cfg_.gaitEnabled);
    nh.getParam(ns + "/gait/frequency", cfg_.gaitFrequency);
    nh.getParam(ns + "/gait/offset", cfg_.gaitOffset);
    nh.getParam(ns + "/gait/duration", cfg_.gaitDuration);
    nh.getParam(ns + "/gait/swing_height", cfg_.gaitSwingHeight);

    nh.getParam(ns + "/commands/max/lin_vel_x", cfg_.maxCmdVx);
    nh.getParam(ns + "/commands/max/lin_vel_y", cfg_.maxCmdVy);
    nh.getParam(ns + "/commands/max/ang_vel_yaw", cfg_.maxCmdYaw);

    std::unordered_map<std::string, int> sensorIdxByName;
    for (int i = 0; i < static_cast<int>(cfg_.sensorJointNames.size()); i++)
    {
        sensorIdxByName[cfg_.sensorJointNames[i]] = i;
    }
    cfg_.actionToSensorIdx.resize(numJoints);
    for (int i = 0; i < numJoints; i++)
    {
        const auto sensor = sensorIdxByName.find(cfg_.jointNames[i]);
        if (sensor == sensorIdxByName.end())
        {
            ROS_ERROR("Joint %s has no sensor mapping", cfg_.jointNames[i].c_str());
            return false;
        }
        cfg_.actionToSensorIdx[i] = sensor->second;
    }

    cfg_.wheelfootNonWheelSensorIdx.clear();
    for (int i = 0; i < static_cast<int>(cfg_.sensorJointNames.size()); i++)
    {
        if (cfg_.sensorJointNames[i].find("wheel_") == std::string::npos)
        {
            cfg_.wheelfootNonWheelSensorIdx.push_back(i);
        }
    }

    return true;
}

bool ControllerBase::loadBundleContract(ros::NodeHandle &nh)
{
    std::string requested;
    if (!nh.getParam("task_profile", requested))
    {
        nh.getParam("/taskProfile", requested);
    }
    const TaskSpec *selected = nullptr;
    for (const auto &spec : kWFYGTasks)
    {
        if (requested == spec.taskId || requested == spec.playAlias)
        {
            selected = &spec;
            break;
        }
    }
    if (selected == nullptr)
    {
        ROS_ERROR("Unregistered WFYG task profile: %s", requested.c_str());
        return false;
    }

    std::string schema, manifestTask, runtime, decoder, assetProfile;
    int observationSize = 0, commandSize = 0, actionSize = 0;
    int encoderInputSize = 0, encoderOutputSize = 0, policyInputSize = 0, policyOutputSize = 0;
    if (!nh.getParam("/model_bundle/schema", schema) || schema != "limx.wfyg-deployment-bundle/v1" ||
        !nh.getParam("/model_bundle/task/task_id", manifestTask) ||
        !nh.getParam("/model_bundle/task/runtime_profile", runtime) ||
        !nh.getParam("/model_bundle/task/action_decoder", decoder) ||
        !nh.getParam("/model_bundle/assets/profile", assetProfile) ||
        !nh.getParam("/model_bundle/task/observation_size", observationSize) ||
        !nh.getParam("/model_bundle/task/command_size", commandSize) ||
        !nh.getParam("/model_bundle/task/action_size", actionSize) ||
        !nh.getParam("/model_bundle/task/encoder_input_size", encoderInputSize) ||
        !nh.getParam("/model_bundle/task/encoder_output_size", encoderOutputSize) ||
        !nh.getParam("/model_bundle/task/policy_input_size", policyInputSize) ||
        !nh.getParam("/model_bundle/task/policy_output_size", policyOutputSize))
    {
        ROS_ERROR("WFYG bundle manifest is missing required contract fields");
        return false;
    }
    if (manifestTask != selected->taskId || runtime != selected->runtime || decoder != selected->decoder ||
        observationSize != selected->observationSize || commandSize != selected->commandSize ||
        actionSize != selected->actionSize || encoderInputSize != selected->encoderInputSize ||
        policyInputSize != selected->policyInputSize || encoderOutputSize != 3 || policyOutputSize != actionSize)
    {
        ROS_ERROR("WFYG bundle does not match registered profile %s", selected->taskId.c_str());
        return false;
    }
    if (assetProfile != "legacy-nominal" && assetProfile != "payload-3kg")
    {
        ROS_ERROR("Unsupported WFYG asset profile: %s", assetProfile.c_str());
        return false;
    }

    std::vector<std::string> hardwareJoints;
    std::vector<double> defaultPositions;
    if (!nh.getParam("/model_bundle/task/hardware_joint_names", hardwareJoints) || hardwareJoints != cfg_.jointNames ||
        !nh.getParam("/model_bundle/task/default_joint_positions", defaultPositions) ||
        defaultPositions.size() != cfg_.jointNames.size())
    {
        ROS_ERROR("WFYG bundle hardware order/default pose does not match controller configuration");
        return false;
    }
    cfg_.defaultDofPos = defaultPositions;
    cfg_.defaultDofPosSensor = defaultPositions;
    cfg_.taskId = selected->taskId;
    cfg_.runtimeProfile = runtime;
    cfg_.actionDecoder = decoder;
    cfg_.assetProfile = assetProfile;
    cfg_.actionsSize = actionSize;
    cfg_.policyObsSize = observationSize;
    cfg_.commandsObsSize = commandSize;
    cfg_.encoderInputSize = encoderInputSize;
    cfg_.encoderOutputSize = encoderOutputSize;
    cfg_.policyInputSize = policyInputSize;
    cfg_.obsHistoryLength = 10;
    if (cfg_.runtimeProfile == "locomotion")
    {
        for (size_t index = 10; index < 16 && index < cfg_.kp.size(); ++index)
        {
            cfg_.kp[index] = 800.0;
            cfg_.kd[index] = 40.0;
        }
    }
    if (!nh.getParam("/model_bundle/models/policy/sha256", cfg_.expectedPolicySha256) ||
        !nh.getParam("/model_bundle/models/encoder/sha256", cfg_.expectedEncoderSha256))
    {
        ROS_ERROR("WFYG bundle is missing model SHA-256 values");
        return false;
    }
    return true;
}

bool ControllerBase::loadModel(ros::NodeHandle &nh)
{
    std::string policyFile;
    std::string encoderFile;
    if (!nh.getParam("/policyFile", policyFile) || !nh.getParam("/encoderFile", encoderFile))
    {
        ROS_ERROR("policyFile or encoderFile param not found");
        return false;
    }

    try
    {
        if (isWFYGVariant_)
        {
            if (sha256File(policyFile) != cfg_.expectedPolicySha256 ||
                sha256File(encoderFile) != cfg_.expectedEncoderSha256)
            {
                ROS_ERROR("WFYG ONNX checksum does not match bundle manifest");
                return false;
            }
        }
        onnxEnvPtr_ = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Tron2Controller");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetInterOpNumThreads(1);

        policySessionPtr_ = new Ort::Session(*onnxEnvPtr_, policyFile.c_str(), sessionOptions);
        encoderSessionPtr_ = new Ort::Session(*onnxEnvPtr_, encoderFile.c_str(), sessionOptions);
    }
    catch (const std::exception &error)
    {
        ROS_ERROR("Failed to open ONNX bundle: %s", error.what());
        return false;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < policySessionPtr_->GetInputCount(); i++)
    {
        policyInputNames_.push_back(policySessionPtr_->GetInputName(i, allocator));
        policyInputShapes_.push_back(policySessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    for (size_t i = 0; i < policySessionPtr_->GetOutputCount(); i++)
    {
        policyOutputNames_.push_back(policySessionPtr_->GetOutputName(i, allocator));
        policyOutputShapes_.push_back(policySessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    for (size_t i = 0; i < encoderSessionPtr_->GetInputCount(); i++)
    {
        encoderInputNames_.push_back(encoderSessionPtr_->GetInputName(i, allocator));
        encoderInputShapes_.push_back(encoderSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    for (size_t i = 0; i < encoderSessionPtr_->GetOutputCount(); i++)
    {
        encoderOutputNames_.push_back(encoderSessionPtr_->GetOutputName(i, allocator));
        encoderOutputShapes_.push_back(encoderSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    if (isWFYGVariant_)
    {
        const bool countsValid = policyInputNames_.size() == 1 && policyOutputNames_.size() == 1 &&
                                 encoderInputNames_.size() == 1 && encoderOutputNames_.size() == 1;
        const bool namesValid = countsValid && std::string(policyInputNames_[0]) == "policy_input" &&
                                std::string(policyOutputNames_[0]) == "actions" &&
                                std::string(encoderInputNames_[0]) == "obs_history" &&
                                std::string(encoderOutputNames_[0]) == "latent";
        const bool shapesValid = countsValid && shapeMatches(policyInputShapes_[0], cfg_.policyInputSize) &&
                                 shapeMatches(policyOutputShapes_[0], cfg_.actionsSize) &&
                                 shapeMatches(encoderInputShapes_[0], cfg_.encoderInputSize) &&
                                 shapeMatches(encoderOutputShapes_[0], cfg_.encoderOutputSize);
        if (!namesValid || !shapesValid)
        {
            ROS_ERROR("WFYG ONNX names/shapes do not match the versioned bundle contract");
            return false;
        }
    }

    return true;
}

void ControllerBase::updateObsHistory()
{
    const int obsSize = cfg_.policyObsSize;
    const int histLen = cfg_.obsHistoryLength;
    if (isFirstObs_)
    {
        for (int t = 0; t < histLen; t++)
        {
            proprioHistoryBuffer_.segment(t * obsSize, obsSize) = observations_;
        }
        isFirstObs_ = false;
        return;
    }

    for (int t = 0; t < histLen - 1; t++)
    {
        proprioHistoryBuffer_.segment(t * obsSize, obsSize) =
            proprioHistoryBuffer_.segment((t + 1) * obsSize, obsSize);
    }
    proprioHistoryBuffer_.segment((histLen - 1) * obsSize, obsSize) = observations_;
}

void ControllerBase::computeEncoder()
{
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                 OrtMemType::OrtMemTypeDefault);
    std::vector<int64_t> shape = encoderInputShapes_[0];
    if (shape.size() == 1)
    {
        shape[0] = cfg_.encoderInputSize;
    }
    else if (shape.size() == 2)
    {
        shape[0] = 1;
        shape[1] = cfg_.encoderInputSize;
    }

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, proprioHistoryBuffer_.data(), cfg_.encoderInputSize, shape.data(), shape.size());
    auto output = encoderSessionPtr_->Run(Ort::RunOptions{nullptr},
                                          encoderInputNames_.data(), &inputTensor, 1,
                                          encoderOutputNames_.data(), 1);
    const float *outputData = output[0].GetTensorData<float>();
    for (int i = 0; i < cfg_.encoderOutputSize; i++)
    {
        encoderOut_(i) = outputData[i];
    }
}

void ControllerBase::computeActions()
{
    Eigen::VectorXf policyInput(cfg_.policyInputSize);
    int offset = 0;
    policyInput.segment(offset, cfg_.encoderOutputSize) = encoderOut_;
    offset += cfg_.encoderOutputSize;
    policyInput.segment(offset, cfg_.policyObsSize) = observations_;
    offset += cfg_.policyObsSize;
    policyInput.segment(offset, cfg_.commandsObsSize) = policyCommands_;

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                 OrtMemType::OrtMemTypeDefault);
    std::vector<int64_t> shape = policyInputShapes_[0];
    if (shape.size() == 1)
    {
        shape[0] = cfg_.policyInputSize;
    }
    else if (shape.size() == 2)
    {
        shape[0] = 1;
        shape[1] = cfg_.policyInputSize;
    }

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, policyInput.data(), cfg_.policyInputSize, shape.data(), shape.size());
    auto output = policySessionPtr_->Run(Ort::RunOptions{nullptr},
                                         policyInputNames_.data(), &inputTensor, 1,
                                         policyOutputNames_.data(), 1);
    const float *outputData = output[0].GetTensorData<float>();
    for (int i = 0; i < cfg_.actionsSize; i++)
    {
        actions_(i) = std::clamp(outputData[i],
                                 static_cast<float>(-cfg_.clipActions),
                                 static_cast<float>(cfg_.clipActions));
    }
}

void ControllerBase::clipActionsByTorque()
{
    for (int i = 0; i < cfg_.actionsSize; i++)
    {
        if (cfg_.jointModes[i] != "position" || cfg_.kp[i] <= 1e-6)
        {
            continue;
        }
        const double q = hybridJointHandles_[i].getPosition();
        const double dq = hybridJointHandles_[i].getVelocity();
        const double deltaPos = q - cfg_.defaultDofPos[i];
        const double kp = cfg_.kp[i];
        const double kd = cfg_.kd[i];
        const double tl = cfg_.torqueLimits[i];
        const double actMin = (deltaPos + (kd * dq - tl) / kp) / cfg_.actionScalePos;
        const double actMax = (deltaPos + (kd * dq + tl) / kp) / cfg_.actionScalePos;
        actions_(i) = std::clamp(static_cast<double>(actions_(i)), actMin, actMax);
    }
}

void ControllerBase::cmdVelCallback(const geometry_msgs::Twist::ConstPtr &msg)
{
    tron2Command_(0) = std::clamp(msg->linear.x, -cfg_.maxCmdVx, cfg_.maxCmdVx);
    tron2Command_(1) = std::clamp(msg->linear.y, -cfg_.maxCmdVy, cfg_.maxCmdVy);
    tron2Command_(2) = std::clamp(msg->angular.z, -cfg_.maxCmdYaw, cfg_.maxCmdYaw);
    if (cfg_.runtimeProfile.empty() || cfg_.runtimeProfile == "locomotion")
    {
        if (policyCommands_.size() >= 3)
        {
            policyCommands_.head<3>() = tron2Command_.cast<float>();
        }
        commandReceived_ = true;
        lastCommandTime_ = ros::Time::now();
    }
}

void ControllerBase::modeCallback(const std_msgs::String::ConstPtr &msg)
{
    const bool requestWalk = msg->data == "WALK" && (!isWFYGVariant_ || cfg_.runtimeProfile == "locomotion");
    const bool requestTrack = msg->data == "TRACK" && isWFYGVariant_ && cfg_.runtimeProfile != "locomotion";
    if ((requestWalk || requestTrack) && mode_ != FAULT && mode_ != WALK && mode_ != TRACK)
    {
        mode_ = requestTrack ? TRACK : WALK;
        loopCount_ = 0;
        isFirstObs_ = true;
        lastActions_.setZero();
        proprioHistoryBuffer_.setZero();
        resetRuntimeState();
        ROS_INFO("Switching to %s mode via topic", requestTrack ? "TRACK" : "WALK");
    }
    else if (msg->data == "STAND" && mode_ != FAULT && mode_ != STAND)
    {
        mode_ = STAND;
        standPercent_ = 0.0;
        for (size_t i = 0; i < hybridJointHandles_.size(); ++i)
        {
            tron2InitJointAngles_(i) = hybridJointHandles_[i].getPosition();
        }
        ROS_INFO("Switching to STAND mode via topic");
    }
    else if (msg->data == "IDLE" && mode_ != FAULT && mode_ != IDLE)
    {
        mode_ = IDLE;
        ROS_INFO("Switching to IDLE mode via topic");
    }
    else if (msg->data == "RESET_FAULT" && mode_ == FAULT)
    {
        faultReason_.clear();
        mode_ = STAND;
        standPercent_ = 0.0;
        commandReceived_ = false;
        isFirstObs_ = true;
        actions_.setZero();
        lastActions_.setZero();
        proprioHistoryBuffer_.setZero();
        resetRuntimeState();
        for (size_t i = 0; i < hybridJointHandles_.size(); ++i)
        {
            tron2InitJointAngles_(i) = hybridJointHandles_[i].getPosition();
        }
        ROS_WARN("WFYG fault reset; controller returned to STAND and requires a fresh command");
    }
    else if (mode_ == FAULT)
    {
        ROS_WARN_THROTTLE(1.0, "FAULT is latched; send RESET_FAULT before changing mode");
    }
    else if ((msg->data == "WALK" || msg->data == "TRACK") && isWFYGVariant_)
    {
        ROS_ERROR("Mode %s is invalid for runtime profile %s", msg->data.c_str(), cfg_.runtimeProfile.c_str());
    }
}

void ControllerBase::enterFault(const std::string &reason)
{
    if (mode_ != FAULT || faultReason_ != reason)
    {
        ROS_ERROR("TRON2 controller entered latched FAULT: %s", reason.c_str());
    }
    faultReason_ = reason;
    mode_ = FAULT;
}

bool ControllerBase::valuesFinite(const Eigen::VectorXf &values) const
{
    return values.allFinite();
}

void ControllerBase::publishDiagnostics(const ros::Time &time)
{
    static ros::Time lastPublish;
    if (!lastPublish.isZero() && (time - lastPublish).toSec() < 0.1)
    {
        return;
    }
    lastPublish = time;
    diagnostic_msgs::DiagnosticArray message;
    message.header.stamp = time;
    diagnostic_msgs::DiagnosticStatus status;
    status.name = "tron2_controller/policy";
    status.hardware_id = robotVariant_;
    status.level = mode_ == FAULT ? diagnostic_msgs::DiagnosticStatus::ERROR : diagnostic_msgs::DiagnosticStatus::OK;
    status.message = mode_ == FAULT ? faultReason_ : "controller healthy";
    diagnostic_msgs::KeyValue task;
    task.key = "task_id";
    task.value = cfg_.taskId;
    status.values.push_back(task);
    diagnostic_msgs::KeyValue asset;
    asset.key = "asset_profile";
    asset.value = cfg_.assetProfile;
    status.values.push_back(asset);
    message.status.push_back(status);
    diagnosticsPub_.publish(message);
}

} // namespace tron2_controller
