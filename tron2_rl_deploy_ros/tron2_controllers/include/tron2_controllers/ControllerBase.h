#pragma once

#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <diagnostic_msgs/DiagnosticArray.h>
#include <geometry_msgs/Twist.h>
#include <hardware_interface/imu_sensor_interface.h>
#include <onnxruntime_cxx_api.h>
#include <robot_common/hardware_interface/HybridJointInterface.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

#include <Eigen/Dense>

namespace tron2_controller
{

struct Tron2RLCfg
{
    int actionsSize{};
    int policyObsSize{};
    int commandsObsSize{};
    int encoderInputSize{};
    int encoderOutputSize{};
    int policyInputSize{};
    int obsHistoryLength{};

    double clipObs{};
    double clipActions{};
    double actionScalePos{};
    double actionScaleVel{};
    int decimation{};
    double commandTimeout{0.5};
    bool faultOnCommandTimeout{false};

    bool gaitEnabled{};
    double gaitFrequency{};
    double gaitOffset{};
    double gaitDuration{};
    double gaitSwingHeight{};

    double maxCmdVx{};
    double maxCmdVy{};
    double maxCmdYaw{};

    std::string taskId;
    std::string runtimeProfile;
    std::string actionDecoder;
    std::string assetProfile;
    std::string expectedPolicySha256;
    std::string expectedEncoderSha256;

    std::vector<std::string> jointNames;
    std::vector<std::string> sensorJointNames;
    std::vector<double> defaultDofPos;
    std::vector<double> defaultDofPosSensor;
    std::vector<std::string> jointModes;
    std::vector<double> kp;
    std::vector<double> kd;
    std::vector<double> torqueLimits;

    std::vector<int> actionToSensorIdx;
    std::vector<int> wheelfootNonWheelSensorIdx;
};

class ControllerBase
    : public controller_interface::MultiInterfaceController<robot_common::HybridJointInterface,
                                                             hardware_interface::ImuSensorInterface>
{
public:
    ~ControllerBase() override;
    bool init(hardware_interface::RobotHW *robot_hw, ros::NodeHandle &nh) override;
    void starting(const ros::Time &time) override;
    void update(const ros::Time &time, const ros::Duration &period) override;

protected:
    virtual void computeObservation() = 0;
    virtual void applyActions() = 0;
    virtual void resetRuntimeState() {}

    bool loadRLCfg(ros::NodeHandle &nh);
    bool loadBundleContract(ros::NodeHandle &nh);
    bool loadModel(ros::NodeHandle &nh);

    void computeEncoder();
    void computeActions();
    void updateObsHistory();
    void clipActionsByTorque();

    void cmdVelCallback(const geometry_msgs::Twist::ConstPtr &msg);
    void modeCallback(const std_msgs::String::ConstPtr &msg);
    void enterFault(const std::string &reason);
    void publishDiagnostics(const ros::Time &time);
    bool valuesFinite(const Eigen::VectorXf &values) const;

    Tron2RLCfg cfg_;

    std::vector<robot_common::HybridJointHandle> hybridJointHandles_;
    hardware_interface::ImuSensorHandle imuSensorHandle_;

    Eigen::VectorXf observations_;
    Eigen::VectorXf actions_;
    Eigen::VectorXf lastActions_;
    Eigen::VectorXf encoderOut_;
    Eigen::VectorXf proprioHistoryBuffer_;

    Eigen::Vector3d tron2Command_;
    Eigen::VectorXf policyCommands_;

    bool isFirstObs_{true};
    int loopCount_{0};
    double footGaitPhaseIndex_{0.0};
    double tron2PolicyDt_{0.02};
    double tron2HwControlDt_{0.001};
    std::string tron2ImuHandleName_{"base_imu"};
    std::string robotVariant_;
    bool isSolefootVariant_{false};
    bool isWheelfootVariant_{false};
    bool isWFYGVariant_{false};
    bool commandReceived_{false};
    ros::Time lastCommandTime_;
    std::string faultReason_;

    Ort::Env *onnxEnvPtr_{nullptr};
    Ort::Session *policySessionPtr_{nullptr};
    Ort::Session *encoderSessionPtr_{nullptr};
    std::vector<const char *> policyInputNames_;
    std::vector<const char *> policyOutputNames_;
    std::vector<std::vector<int64_t>> policyInputShapes_;
    std::vector<std::vector<int64_t>> policyOutputShapes_;
    std::vector<const char *> encoderInputNames_;
    std::vector<const char *> encoderOutputNames_;
    std::vector<std::vector<int64_t>> encoderInputShapes_;
    std::vector<std::vector<int64_t>> encoderOutputShapes_;

    ros::Subscriber cmdVelSub_;
    ros::Subscriber modeSub_;
    ros::Publisher diagnosticsPub_;

    enum ControlMode
    {
        IDLE,
        STAND,
        WALK,
        TRACK,
        FAULT
    };
    ControlMode mode_{IDLE};

    double standPercent_{0.0};
    double standDuration_{1.0};
    Eigen::VectorXd tron2InitJointAngles_;
};

} // namespace tron2_controller
