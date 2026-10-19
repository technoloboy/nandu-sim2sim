#include "tron2_hw/RobotHWLoop.h"
#include "tron2_hw/Tron2HW.h"

#include <chrono>
#include <thread>

#include <ros/ros.h>
#include <rosgraph_msgs/Clock.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "tron2_hw");

    ros::NodeHandle nh;
    ros::NodeHandle robotHwNh("~");

    if (argc > 1)
    {
        robotHwNh.setParam("robot_ip", std::string(argv[1]));
    }

    std::string robotIp;
    robotHwNh.param<std::string>("robot_ip", robotIp, std::string("10.192.1.2"));
    const bool simMode = (robotIp == "127.0.0.1");
    bool waitForSimClock = true;
    nh.param("/robot_hw/wait_for_sim_clock", waitForSimClock, true);

    if (simMode && waitForSimClock)
    {
        ROS_INFO("Tron2HW: sim mode, waiting for /clock so Gazebo Tron2HWSim is ready...");
        auto clockMsg = ros::topic::waitForMessage<rosgraph_msgs::Clock>("/clock", ros::Duration(30.0));
        if (clockMsg == nullptr)
        {
            ROS_WARN("Tron2HW: /clock not seen within 30s, proceeding anyway.");
        }
    }
    else if (simMode)
    {
        ROS_INFO("Tron2HW: MuJoCo SDK mode, /clock wait disabled.");
    }

    auto hw = std::make_shared<tron2_hw::Tron2HW>();
    if (!hw->init(nh, robotHwNh))
    {
        ROS_ERROR("Failed to initialize Tron2HW.");
        return 1;
    }

    if (simMode)
    {
        ROS_INFO("Tron2HW: waiting for first SDK RobotState (handshake), timeout 15s...");
        ros::Rate rate(50);
        const int maxTicks = 50 * 15;  // 15s timeout
        int waited = 0;
        while (ros::ok() && !hw->firstStateReceived() && waited < maxTicks)
        {
            rate.sleep();
            if (++waited % 100 == 0)
            {
                ROS_WARN("Tron2HW: still waiting for SDK RobotState after %.1fs", waited / 50.0);
            }
        }
        if (hw->firstStateReceived())
        {
            ROS_INFO("Tron2HW: SDK handshake confirmed, bringing up ControllerManager.");
        }
        else
        {
            ROS_ERROR("Tron2HW: SDK handshake TIMEOUT — Gazebo Tron2HWSim plugin likely failed to load. "
                      "Bringing up ControllerManager anyway, but cmd will not reach Gazebo.");
        }
    }

    tron2_hw::RobotHWLoop loop(nh, robotHwNh, hw);
    loop.startControlLoop(nh);

    std::thread startThread([hw]()
                            {
        std::this_thread::sleep_for(std::chrono::milliseconds(1200));
        hw->startController(); });
    startThread.detach();

    ros::spin();
    return 0;
}
