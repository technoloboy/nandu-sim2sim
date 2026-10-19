#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include <controller_manager/controller_manager.h>
#include <ros/ros.h>

#include "tron2_hw/Tron2HW.h"

namespace tron2_hw
{

class RobotHWLoop
{
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;

public:
    RobotHWLoop(ros::NodeHandle &nh, ros::NodeHandle &robot_hw_nh, std::shared_ptr<Tron2HW> hardware_interface);
    ~RobotHWLoop();

    void startControlLoop(ros::NodeHandle &nh);
    void update();

private:
    double cycleTimeErrorThreshold_{0.002};
    double loopHz_{500.0};
    std::thread loopThread_;
    std::atomic_bool loopRunning_{false};
    ros::Duration elapsedTime_;
    Clock::time_point lastTime_;

    std::shared_ptr<controller_manager::ControllerManager> controllerManager_;
    std::shared_ptr<Tron2HW> hardwareInterface_;
};

} // namespace tron2_hw
