#pragma once

#include <algorithm>
#include <cmath>
#include <utility>

#include <Eigen/Dense>

namespace tron2_controller::wfyg_math
{

inline Eigen::Matrix<float, 6, 1> rotation6D(const Eigen::Matrix3d &rotation)
{
    Eigen::Matrix<float, 6, 1> result;
    result.segment<3>(0) = rotation.col(0).cast<float>();
    result.segment<3>(3) = rotation.col(1).cast<float>();
    return result;
}

inline Eigen::Vector3d orientationAxisAngleError(
    const Eigen::Quaterniond &current, const Eigen::Quaterniond &target)
{
    Eigen::Quaterniond error = (target * current.conjugate()).normalized();
    if (error.w() < 0.0)
    {
        error.coeffs() *= -1.0;
    }
    const Eigen::AngleAxisd angleAxis(error);
    if (std::abs(angleAxis.angle()) < 1.0e-8)
    {
        return Eigen::Vector3d::Zero();
    }
    return angleAxis.axis() * angleAxis.angle();
}

inline std::pair<double, double> softLimits(double lower, double upper, double factor = 0.9)
{
    const double middle = 0.5 * (lower + upper);
    const double halfRange = 0.5 * (upper - lower) * factor;
    return {middle - halfRange, middle + halfRange};
}

inline double unscaleToLimits(double rawAction, double lower, double upper)
{
    const double scaled = std::clamp(0.9 * rawAction, -1.0, 1.0);
    return lower + 0.5 * (scaled + 1.0) * (upper - lower);
}

inline double ema(double target, double previous, double alpha = 0.2)
{
    return alpha * target + (1.0 - alpha) * previous;
}

} // namespace tron2_controller::wfyg_math
