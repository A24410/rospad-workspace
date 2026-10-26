"""
joint_sine_demo.py — animates the UR5 arm with a smooth sine wave

Run this in the ROSpad terminal to see the arm move:
    python src/ur5_description/scripts/joint_sine_demo.py
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# Amplitude and frequency for each joint
AMPLITUDES  = [0.8,  0.4,   0.6,  0.4,   0.5,  0.3]
FREQUENCIES = [0.3,  0.5,   0.4,  0.7,   0.6,  0.9]
OFFSETS     = [0.0, -math.pi/2, 0.0, -math.pi/2, 0.0, 0.0]


class JointSineDemo(Node):
    def __init__(self):
        super().__init__('joint_sine_demo')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.033, self.publish_state)  # ~30 Hz
        self.t = 0.0
        self.get_logger().info('Sine demo started — watch the arm move!')

    def publish_state(self):
        self.t += 0.033
        positions = [
            OFFSETS[i] + AMPLITUDES[i] * math.sin(2 * math.pi * FREQUENCIES[i] * self.t)
            for i in range(6)
        ]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = positions
        msg.velocity = [0.0] * 6
        msg.effort = [0.0] * 6
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointSineDemo()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
