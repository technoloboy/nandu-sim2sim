#pragma once

#include <cstddef>
#include <cstdint>

namespace tron2_hw
{

struct MotorData
{
    double pos_{0.0};
    double vel_{0.0};
    double tau_{0.0};

    double posDes_{0.0};
    double velDes_{0.0};
    double kp_{0.0};
    double kd_{0.0};
    double tauFf_{0.0};
    uint8_t mode_{0};
};

struct ImuData
{
    double ori_[4]{0.0, 0.0, 0.0, 1.0};
    double oriCov_[9]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double angularVel_[3]{0.0, 0.0, 0.0};
    double angularVelCov_[9]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double linearAcc_[3]{0.0, 0.0, 0.0};
    double linearAccCov_[9]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

} // namespace tron2_hw
