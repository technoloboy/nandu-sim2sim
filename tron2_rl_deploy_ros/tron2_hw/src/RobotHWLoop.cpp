#include "tron2_hw/RobotHWLoop.h"

namespace tron2_hw
{

RobotHWLoop::RobotHWLoop(ros::NodeHandle &, ros::NodeHandle &robot_hw_nh, std::shared_ptr<Tron2HW> hardware_interface)
    : loopRunning_(false), hardwareInterface_(std::move(hardware_interface))
{
    controllerManager_ = std::make_shared<controller_manager::ControllerManager>(hardwareInterface_.get(), robot_hw_nh);
}

void RobotHWLoop::startControlLoop(ros::NodeHandle &nh)
{
    nh.param("/robot_hw/loop_frequency", loopHz_);
    nh.param("/robot_hw/cycle_time_error_threshold", cycleTimeErrorThreshold_, 0.002);

    ROS_INFO("Start control loop. loopHz=%.3f, cycleTimeErrorThreshold=%.6f",
             loopHz_, cycleTimeErrorThreshold_);

    lastTime_ = Clock::now();
    loopRunning_ = true;
    loopThread_ = std::thread([this]()
                              {
        while (loopRunning_)
        {
            update();
        } });
}

void RobotHWLoop::update()
{
    const auto currentTime = Clock::now();
    const Duration desiredDuration(1.0 / loopHz_);

    Duration time_span = std::chrono::duration_cast<Duration>(currentTime - lastTime_);
    elapsedTime_ = ros::Duration(time_span.count());
    lastTime_ = currentTime;

    const double cycleTimeError = (elapsedTime_ - ros::Duration(desiredDuration.count())).toSec();
    if (cycleTimeError > cycleTimeErrorThreshold_)
    {
        ROS_WARN_THROTTLE(1.0, "Cycle time exceeded threshold: error=%.6f, elapsed=%.6f, threshold=%.6f",
                          cycleTimeError, elapsedTime_.toSec(), cycleTimeErrorThreshold_);
    }

    hardwareInterface_->read(ros::Time::now(), elapsedTime_);
    controllerManager_->update(ros::Time::now(), elapsedTime_);
    hardwareInterface_->write(ros::Time::now(), elapsedTime_);

    const auto sleepTill = currentTime + std::chrono::duration_cast<Clock::duration>(desiredDuration);
    std::this_thread::sleep_until(sleepTill);
}

RobotHWLoop::~RobotHWLoop()
{
    loopRunning_ = false;
    if (loopThread_.joinable())
    {
        loopThread_.join();
    }
}

} // namespace tron2_hw
