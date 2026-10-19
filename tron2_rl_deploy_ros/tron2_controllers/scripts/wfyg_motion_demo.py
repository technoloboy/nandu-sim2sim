#!/usr/bin/env python3
"""Safe, bounded EE-motion demonstrations for WFYG v1/Heading/Height policies."""

from __future__ import annotations

import math
import threading

import rospy
from geometry_msgs.msg import PoseStamped
from tron2_controllers.msg import EEMotionCommand


class MotionDemo:
    def __init__(self) -> None:
        self.mode = rospy.get_param("~mode", "home")
        if self.mode not in {"home", "line", "orbit", "zigzag", "vertical"}:
            raise ValueError(f"Unsupported demo mode: {self.mode}")
        self.rate_hz = float(rospy.get_param("~rate", 50.0))
        self.speed = max(0.01, min(float(rospy.get_param("~speed", 0.10)), 0.20))
        self.amplitude = max(0.01, min(float(rospy.get_param("~amplitude", 0.08)), 0.15))
        self.feedback_gain = max(0.0, min(float(rospy.get_param("~feedback_gain", 1.5)), 3.0))
        self.position = None
        self.lock = threading.Lock()
        self.publisher = rospy.Publisher("/tron2_controller/ee_motion_command", EEMotionCommand, queue_size=1)
        self.subscriber = rospy.Subscriber("/tron2_controller/ee_pose", PoseStamped, self.pose_callback, queue_size=1)

    def pose_callback(self, message: PoseStamped) -> None:
        with self.lock:
            self.position = (
                message.pose.position.x,
                message.pose.position.y,
                message.pose.position.z,
            )

    @staticmethod
    def triangle(phase: float) -> float:
        return 2.0 / math.pi * math.asin(math.sin(phase))

    def desired(self, elapsed: float):
        home = (0.4209992, 0.0, 0.4273805)
        omega = self.speed / self.amplitude
        phase = omega * elapsed
        if self.mode == "home":
            return home, (0.0, 0.0, 0.0)
        if self.mode == "line":
            return (
                (home[0] + self.amplitude * math.sin(phase), home[1], home[2]),
                (self.speed * math.cos(phase), 0.0, 0.0),
            )
        if self.mode == "orbit":
            return (
                (home[0] + self.amplitude * math.cos(phase), home[1] + self.amplitude * math.sin(phase), home[2]),
                (-self.speed * math.sin(phase), self.speed * math.cos(phase), 0.0),
            )
        if self.mode == "vertical":
            return (
                (home[0], home[1], home[2] + self.amplitude * math.sin(phase)),
                (0.0, 0.0, self.speed * math.cos(phase)),
            )
        # Bounded triangular x/y references; feedback closes the corners without
        # placing trajectory state in the actor input.
        return (
            (
                home[0] + self.amplitude * self.triangle(phase),
                home[1] + 0.5 * self.amplitude * self.triangle(2.0 * phase + math.pi / 2.0),
                home[2],
            ),
            (0.0, 0.0, 0.0),
        )

    def run(self) -> None:
        rate = rospy.Rate(self.rate_hz)
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            with self.lock:
                position = self.position
            if position is None:
                rospy.logwarn_throttle(2.0, "Waiting for /tron2_controller/ee_pose before publishing demo commands")
                rate.sleep()
                continue
            elapsed = (rospy.Time.now() - start).to_sec()
            target, feedforward = self.desired(elapsed)
            velocity = [
                feedforward[index] + self.feedback_gain * (target[index] - position[index])
                for index in range(3)
            ]
            limits = (0.60, 0.25, 0.15)
            velocity = [max(-limit, min(limit, value)) for value, limit in zip(velocity, limits)]
            command = EEMotionCommand()
            command.header.stamp = rospy.Time.now()
            command.header.frame_id = "base_heading"
            command.linear_velocity.x, command.linear_velocity.y, command.linear_velocity.z = velocity
            # Training home orientation is wxyz [0.7070451, 0, 0.7071684, 0].
            command.orientation.x = 0.0
            command.orientation.y = 0.7071684
            command.orientation.z = 0.0
            command.orientation.w = 0.7070451
            self.publisher.publish(command)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("wfyg_motion_demo")
    MotionDemo().run()
