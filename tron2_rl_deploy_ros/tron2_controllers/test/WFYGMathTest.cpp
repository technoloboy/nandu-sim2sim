#include <cmath>

#include <gtest/gtest.h>

#include "tron2_controllers/WFYGMath.h"

namespace math = tron2_controller::wfyg_math;

TEST(WFYGMath, RotationSixDUsesColumns)
{
    constexpr double kHalfPi = 1.5707963267948966;
    EXPECT_TRUE(math::rotation6D(Eigen::Matrix3d::Identity()).isApprox(
        (Eigen::Matrix<float, 6, 1>() << 1, 0, 0, 0, 1, 0).finished()));
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(kHalfPi, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    EXPECT_TRUE(math::rotation6D(rotation).isApprox(
        (Eigen::Matrix<float, 6, 1>() << 0, 1, 0, -1, 0, 0).finished(), 1.0e-6f));
}

TEST(WFYGMath, PoseErrorAndActionDecoders)
{
    constexpr double kHalfPi = 1.5707963267948966;
    const Eigen::Quaterniond target(Eigen::AngleAxisd(kHalfPi, Eigen::Vector3d::UnitZ()));
    EXPECT_TRUE(math::orientationAxisAngleError(Eigen::Quaterniond::Identity(), target).isApprox(
        Eigen::Vector3d(0, 0, kHalfPi), 1.0e-9));
    const auto limits = math::softLimits(-3.14159, 2.60054);
    EXPECT_NEAR(limits.first, -2.8544835, 2.0e-6);
    EXPECT_NEAR(limits.second, 2.3134335, 2.0e-6);
    EXPECT_NEAR(math::unscaleToLimits(0.0, limits.first, limits.second), -0.270525, 2.0e-6);
    EXPECT_NEAR(math::ema(1.0, 0.0), 0.2, 1.0e-12);
}
