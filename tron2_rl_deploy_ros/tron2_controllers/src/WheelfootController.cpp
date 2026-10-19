#include "tron2_controllers/WheelfootController.h"

#include <cmath>
#include <pluginlib/class_list_macros.h>

namespace tron2_controller
{

void WheelfootController::computeObservation()
{
    const double *quat = imuSensorHandle_.getOrientation();
    const double *gyro = imuSensorHandle_.getAngularVelocity();

    const double qx = quat[0], qy = quat[1], qz = quat[2], qw = quat[3];
    Eigen::Quaterniond qEigen(qw, qx, qy, qz);
    Eigen::Matrix3d invRot = qEigen.toRotationMatrix().transpose();

    constexpr float angVelScale = 0.25f;
    Eigen::Vector3f baseAngVel;
    baseAngVel << static_cast<float>(gyro[0]) * angVelScale,
        static_cast<float>(gyro[1]) * angVelScale,
        static_cast<float>(gyro[2]) * angVelScale;

    Eigen::Vector3d gravityWorld(0.0, 0.0, -1.0);
    Eigen::Vector3f projGravity = (invRot * gravityWorld).cast<float>();

    const int n = cfg_.actionsSize;
    Eigen::VectorXf qSensor(n), dqSensor(n);
    for (int i = 0; i < n; i++)
    {
        const int sIdx = cfg_.actionToSensorIdx[i];
        qSensor(sIdx) = static_cast<float>(hybridJointHandles_[i].getPosition());
        dqSensor(sIdx) = static_cast<float>(hybridJointHandles_[i].getVelocity());
    }

    const int nonWheelCount = static_cast<int>(cfg_.wheelfootNonWheelSensorIdx.size());
    Eigen::VectorXf jointPos(nonWheelCount);
    for (int i = 0; i < nonWheelCount; i++)
    {
        const int idx = cfg_.wheelfootNonWheelSensorIdx[i];
        jointPos(i) = qSensor(idx) - static_cast<float>(cfg_.defaultDofPosSensor[idx]);
    }

    constexpr float velScale = 0.05f;
    Eigen::VectorXf jointVel = dqSensor * velScale;

    int offset = 0;
    observations_.segment(offset, 3) = baseAngVel;
    offset += 3;
    observations_.segment(offset, 3) = projGravity;
    offset += 3;
    observations_.segment(offset, nonWheelCount) = jointPos;
    offset += nonWheelCount;
    observations_.segment(offset, n) = jointVel;
    offset += n;
    observations_.segment(offset, n) = lastActions_;
    offset += n;

    if (cfg_.gaitEnabled)
    {
        if (tron2Command_.norm() < 0.02)
        {
            footGaitPhaseIndex_ = 0.0;
        }
        else
        {
            footGaitPhaseIndex_ = std::fmod(footGaitPhaseIndex_ + tron2PolicyDt_ * cfg_.gaitFrequency, 1.0);
        }

        Eigen::Vector2f gaitPhase;
        gaitPhase << static_cast<float>(std::sin(footGaitPhaseIndex_ * 2.0 * M_PI)),
            static_cast<float>(std::cos(footGaitPhaseIndex_ * 2.0 * M_PI));
        observations_.segment(offset, 2) = gaitPhase;
        offset += 2;

        Eigen::Vector4f gaitCmd;
        gaitCmd << static_cast<float>(cfg_.gaitFrequency),
            static_cast<float>(cfg_.gaitOffset),
            static_cast<float>(cfg_.gaitDuration),
            static_cast<float>(cfg_.gaitSwingHeight);
        observations_.segment(offset, 4) = gaitCmd;
    }

    for (int i = 0; i < cfg_.policyObsSize; i++)
    {
        observations_(i) = std::clamp(observations_(i),
                                      static_cast<float>(-cfg_.clipObs),
                                      static_cast<float>(cfg_.clipObs));
    }
}

void WheelfootController::applyActions()
{
    for (int i = 0; i < cfg_.actionsSize; i++)
    {
        if (cfg_.jointModes[i] == "velocity")
        {
            const double targetVel = actions_(i) * cfg_.actionScaleVel;
            hybridJointHandles_[i].setCommand(0.0, targetVel, 0.0, cfg_.kd[i], 0.0, 0);
        }
        else
        {
            const double targetQ = actions_(i) * cfg_.actionScalePos + cfg_.defaultDofPos[i];
            hybridJointHandles_[i].setCommand(targetQ, 0.0, cfg_.kp[i], cfg_.kd[i], 0.0, 0);
        }
    }
}

} // namespace tron2_controller

PLUGINLIB_EXPORT_CLASS(tron2_controller::WheelfootController, controller_interface::ControllerBase)
