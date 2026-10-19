#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <controller_manager_msgs/ListControllers.h>
#include <controller_manager_msgs/SwitchController.h>
#include <geometry_msgs/Twist.h>
#include <hardware_interface/imu_sensor_interface.h>
#include <hardware_interface/joint_state_interface.h>
#include <hardware_interface/robot_hw.h>
#include <realtime_tools/realtime_buffer.h>
#include <robot_common/hardware_interface/HybridJointInterface.h>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/UInt64.h>
#include <sensor_msgs/Imu.h>
#include <std_srvs/Trigger.h>
#include <urdf/model.h>

#if __has_include(<limxsdk/tron2.h>)
#include <limxsdk/tron2.h>
#elif __has_include(<tron2.h>)
#include <tron2.h>
#else
#error "Cannot find limxsdk tron2 header."
#endif

#include "tron2_hw/RobotData.h"

namespace tron2_hw
{

class Tron2HW : public hardware_interface::RobotHW
{
public:
    Tron2HW() = default;

    bool init(ros::NodeHandle &root_nh, ros::NodeHandle &robot_hw_nh);
    void read(const ros::Time &time, const ros::Duration &period);
    void write(const ros::Time &time, const ros::Duration &period);

    bool startController();
    bool stopController();

    bool firstStateReceived() const { return firstStateReceived_.load(); }
    bool simMode() const { return simMode_; }

private:
    bool loadUrdf(ros::NodeHandle &nh);
    bool loadJointNames(ros::NodeHandle &nh);
    void setupInterfaces();
    void setupSubscriptions(ros::NodeHandle &root_nh, ros::NodeHandle &robot_hw_nh);
    void setupServices(ros::NodeHandle &robot_hw_nh);

    bool onStartController(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);

    bool isControllerRunning();

    hardware_interface::JointStateInterface jointStateInterface_;
    robot_common::HybridJointInterface hybridJointInterface_;
    hardware_interface::ImuSensorInterface imuSensorInterface_;

    std::vector<MotorData> jointData_;
    std::vector<std::string> jointNames_;
    ImuData imuData_{};
    ImuData up3ImuData_{};
    realtime_tools::RealtimeBuffer<sensor_msgs::Imu> up3ImuBuffer_;
    ros::Subscriber up3ImuSub_;
    ros::Subscriber nandu1EpochSub_;
    std::atomic<uint64_t> nandu1Epoch_{0};
    ros::Publisher sensorHealthPub_;
    bool nandu1Mode_{false};
    std::atomic<double> stateReceiveTime_{0.0}, baseReceiveTime_{0.0}, up3ReceiveTime_{0.0};

    std::shared_ptr<urdf::Model> urdfModel_;

    realtime_tools::RealtimeBuffer<limxsdk::RobotState> robotStateBuffer_;
    realtime_tools::RealtimeBuffer<limxsdk::ImuData> imuDataBuffer_;
    limxsdk::RobotCmd robotCmd_;
    limxsdk::Tron2 *robot_{nullptr};
    bool simMode_{false};

    std::string imuName_{"base_imu"};
    std::string imuFrameId_{"base_imu"};
    std::string controllerName_{"tron2_controller"};
    std::string controllerManagerNs_{"/tron2_hw/controller_manager"};

    bool requireCalibration_{true};
    bool hasCalibrationState_{false};
    int calibrationState_{0};
    std::atomic<bool> firstStateReceived_{false};

    double defaultKp_{0.0};
    double defaultKd_{1.0};

    double joystickScaleVx_{0.5};
    double joystickScaleVy_{0.5};
    double joystickScaleYaw_{0.5};
    bool useSdkJoystickInput_{true};
    bool debugJoystickInput_{false};
    std::atomic<bool> sdkJoystickMsgReceived_{false};

    std::map<std::string, int> tron2JoystickButtonMap_;
    std::map<std::string, int> tron2JoystickAxisMap_;
    ros::Publisher tron2CmdVelPub_;
    ros::Publisher tron2ModePub_;
    ros::WallTimer sdkJoystickWatchdogTimer_;

    ros::ServiceClient switchControllersClient_;
    ros::ServiceClient listControllersClient_;
    ros::ServiceServer startControllerSrv_;
};

} // namespace tron2_hw
