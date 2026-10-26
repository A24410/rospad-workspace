"""
teleop_keyboard.py — keyboard-style teleop for DiffBot in ROSpad

Publishes /cmd_vel at a fixed rate with the velocities set below.
Edit LINEAR and ANGULAR, then click Run to drive the robot.

The 3D sim also responds to WASD keys directly (click the sim viewport first).

Usage: open this file → click ▶ Run
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ── Edit these to drive the robot ────────────────────────────────────────────
LINEAR  =  0.4   # m/s   (positive = forward, negative = reverse)
ANGULAR =  0.0   # rad/s (positive = left, negative = right)
# ─────────────────────────────────────────────────────────────────────────────


class TeleopPublisher(Node):
    def __init__(self):
        super().__init__('teleop_publisher')
        self.pub   = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_cmd)
        self.get_logger().info(
            f'Teleop running — linear={LINEAR} m/s, angular={ANGULAR} rad/s'
        )

    def publish_cmd(self):
        msg = Twist()
        msg.linear.x  = float(LINEAR)
        msg.angular.z = float(ANGULAR)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(TeleopPublisher())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
