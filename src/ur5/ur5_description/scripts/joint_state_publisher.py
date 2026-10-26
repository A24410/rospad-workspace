"""
joint_state_publisher.py — browser-compatible UR5 joint state publisher

Publishes /joint_states at a fixed rate so the 3D sim follows the joint
positions you set. Edit JOINT_POSITIONS below and run this node in the
ROSpad terminal to drive the arm.

Usage (ROSpad terminal):
    python src/ur5_description/scripts/joint_state_publisher.py

Joint order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
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

# Edit these to move the arm (radians)
JOINT_POSITIONS = [0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0]


class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.05, self.publish_state)  # 20 Hz
        self.get_logger().info('Joint state publisher started — publishing at 20 Hz')

    def publish_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = list(JOINT_POSITIONS)
        msg.velocity = [0.0] * 6
        msg.effort = [0.0] * 6
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
