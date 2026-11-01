# """
# color_tracker.py — Vision-based DiffBot control using /camera/image_raw

# Subscribes to the camera topic published when diffbot_camera.urdf is loaded,
# detects the red target ball in the scene with numpy, and publishes /cmd_vel
# to steer the DiffBot toward it.

# Prerequisites:
#   1. Launch diffbot_camera_description:  ros2 launch diffbot_camera_description diffbot_camera.launch.py
#   2. Add a red obstacle in the sim (Obstacles panel → Add Box, set color #ff2020)
#   3. Run this node:  ros2 run diffbot_vision color_tracker
#   4. Enable /camera/image_raw in the Viz panel to see the camera feed live

# Image format: sensor_msgs/Image, encoding=rgb8, 320×240
# The raw pixel array arrives as a JsProxy (Uint8Array) — use numpy to process it.
# """

# import asyncio
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist


# class ColorTracker(Node):
#     def __init__(self):
#         super().__init__('color_tracker')
#         self._sub = self.create_subscription(
#             Image, '/camera/image_raw', self._on_image, 10
#         )
#         self._pub = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.get_logger().info('ColorTracker started — waiting for /camera/image_raw')
#         self._frame_count = 0

#     def _on_image(self, msg):
#         """Process each camera frame and publish a steering command."""
#         w, h = msg.width, msg.height
#         if w == 0 or h == 0:
#             return

#         # Convert JsProxy Uint8Array to numpy RGB array
#         # msg.data is a JsProxy wrapping the raw pixel bytes from sim.js
#         arr = np.asarray(list(msg.data), dtype=np.uint8).reshape(h, w, 3)

#         # Detect red pixels — thresholds account for Lambert shading (ambient 0.4 →
#         # shadow at 40% brightness) and the sim palette red 0xf85149 (G=81 in full light).
#         # Matches wrist_detector.py 'red' bounds: R ∈ [70,255], G ∈ [0,95], B ∈ [0,95].
#         red_mask = (arr[:, :, 0] > 70) & (arr[:, :, 1] < 95) & (arr[:, :, 2] < 95)
#         red_count = int(np.sum(red_mask))

#         twist = Twist()

#         if red_count < 50:
#             # No red visible — spin slowly to search
#             twist.angular.z = 0.4
#             if self._frame_count % 20 == 0:
#                 self.get_logger().info(f'Searching... (red pixels: {red_count})')
#         else:
#             # Find centroid of red region
#             ys, xs = np.where(red_mask)
#             cx = float(np.mean(xs))
#             cy = float(np.mean(ys))

#             # Normalise centroid: -1 (left) to +1 (right)
#             offset = (cx - w / 2.0) / (w / 2.0)

#             # Target area (fraction of image covered by red)
#             area_frac = red_count / (w * h)

#             # Stop when the target occupies a large part of the image
#             if area_frac >= 0.12:
#                 twist.linear.x = 0.0
#                 twist.angular.z = 0.0

#                 if self._frame_count % 10 == 0:
#                     self.get_logger().info(
#                         f'Target close — stopping (area={area_frac*100:.1f}%)'
#                     )
#             else:
#                 # Approach target and steer toward it
#                 twist.linear.x = max(0.05, 0.3 * (1.0 - area_frac * 5))
#                 twist.angular.z = -0.8 * offset

#                 if self._frame_count % 10 == 0:
#                     self.get_logger().info(
#                         f'Target at cx={cx:.0f}px  offset={offset:+.2f}  '
#                         f'area={area_frac*100:.1f}%  '
#                         f'vx={twist.linear.x:.2f}  wz={twist.angular.z:.2f}'
#                     )

#         self._pub.publish(twist)
#         self._frame_count += 1


# def main():
#     rclpy.init()
#     node = ColorTracker()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()


import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from diffbot_vision.red_pixel_model import predict_red_mask

class ColorTracker(Node):
    def __init__(self):
        super().__init__('color_tracker')

        self._sub = self.create_subscription(
            Image, '/camera/image_raw', self._on_image, 10
        )
        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.state = 'SEARCH'
        self.state_count = 0
        self._frame_count = 0

        self.get_logger().info('ColorTracker started')

    def _on_image(self, msg):
        w, h = msg.width, msg.height
        if w == 0 or h == 0:
            return

        arr = np.asarray(
            list(msg.data), dtype=np.uint8
        ).reshape(h, w, 3)

        # 赤色領域を検出
        # red_mask = (
        #     (arr[:, :, 0] > 70) &
        #     (arr[:, :, 1] < 95) &
        #     (arr[:, :, 2] < 95)
        # )
        red_mask = predict_red_mask(arr)
        red_count = int(np.sum(red_mask))
        area_frac = red_count / (w * h)

        twist = Twist()

        # SEARCH: 赤い物体を探す
        if self.state == 'SEARCH':
            twist.angular.z = 0.4

            if red_count >= 50:
                self.state = 'APPROACH'
                self.state_count = 0

        # APPROACH: 赤い物体に接近する
        elif self.state == 'APPROACH':
            if red_count < 50:
                self.state = 'SEARCH'
                twist.angular.z = 0.4
            else:
                ys, xs = np.where(red_mask)
                cx = float(np.mean(xs))
                offset = (cx - w / 2.0) / (w / 2.0)

                if area_frac >= 0.12:
                    self.state = 'RETREAT'
                    self.state_count = 0
                    twist.linear.x = -0.12
                else:
                    twist.linear.x = max(
                        0.05, 0.3 * (1.0 - area_frac * 5)
                    )
                    twist.angular.z = -0.8 * offset

        # RETREAT: 一定時間後退する
        elif self.state == 'RETREAT':
            twist.linear.x = -0.12
            self.state_count += 1

            if self.state_count >= 15:
                self.state = 'TURN'
                self.state_count = 0

        # TURN: 向きを変えて再探索する
        elif self.state == 'TURN':
            twist.angular.z = -0.4
            self.state_count += 1

            if self.state_count >= 15:
                self.state = 'SEARCH'
                self.state_count = 0

        self._pub.publish(twist)

        if self._frame_count % 10 == 0:
            self.get_logger().info(
                f'State={self.state} '
                f'Red={red_count} '
                f'Area={area_frac * 100:.1f}%'
            )

        self._frame_count += 1


def main():
    rclpy.init()
    node = ColorTracker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 停止指令を送ってから終了
        node._pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

