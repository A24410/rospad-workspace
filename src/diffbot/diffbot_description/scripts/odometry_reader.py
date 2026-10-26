"""
odometry_reader.py — subscribe to /odom and print robot position

Run this while the DiffBot is moving to see its position update in real-time.

Usage: open this file → click ▶ Run
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class OdomReader(Node):
    def __init__(self):
        super().__init__('odom_reader')
        self.sub = self.create_subscription(Odometry, '/odom', self.cb, 10)
        self.get_logger().info('Listening on /odom...')

    def cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.get_logger().info(f'Position → x={x:.3f}  y={y:.3f}')


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(OdomReader())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
