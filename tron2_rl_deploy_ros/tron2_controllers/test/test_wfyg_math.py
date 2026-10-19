#!/usr/bin/env python3
import math
import unittest


class WFYGMathGoldenTest(unittest.TestCase):
    def test_rotation_6d_columns(self):
        identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        self.assertEqual(tuple(identity[row][0] for row in range(3)) + tuple(identity[row][1] for row in range(3)), (1, 0, 0, 0, 1, 0))
        z90 = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
        self.assertEqual(tuple(z90[row][0] for row in range(3)) + tuple(z90[row][1] for row in range(3)), (0, 1, 0, -1, 0, 0))

    def test_soft_limit_arm_mapping_and_ema(self):
        lower, upper = -3.14159, 2.60054
        middle = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower) * 0.9
        soft_lower, soft_upper = middle - half_range, middle + half_range
        self.assertAlmostEqual(soft_lower, -2.8544835)
        self.assertAlmostEqual(soft_upper, 2.3134335)
        self.assertAlmostEqual(soft_lower + 0.5 * (soft_upper - soft_lower), -0.270525)
        self.assertAlmostEqual(0.2 * 1.0 + 0.8 * 0.0, 0.2)
        self.assertAlmostEqual(math.pi / 2.0, 1.5707963267948966)


if __name__ == "__main__":
    unittest.main()
