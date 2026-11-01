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

#**********************************************************
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist

# # --- AI model ---
# def sigmoid(x):
#     x = np.clip(x, -30, 30)
#     return 1.0 / (1.0 + np.exp(-x))


# def make_ai_model():
#     rng = np.random.default_rng(42)
#     n = 4000

#     r = rng.uniform(0.65, 1.0, n)
#     g = rng.uniform(0.05, 0.40, n)
#     b = rng.uniform(0.04, 0.38, n)
#     brightness = rng.uniform(0.35, 1.0, n)

#     red = np.column_stack([r, g, b]) * brightness[:, None]
#     red += rng.normal(0, 0.025, red.shape)
#     red = np.clip(red, 0, 1)

#     non_red = []
#     while sum(len(v) for v in non_red) < n:
#         samples = rng.uniform(0, 1, (n, 3))
#         is_red = (
#             (samples[:, 0] > samples[:, 1] * 1.2) &
#             (samples[:, 0] > samples[:, 2] * 1.2) &
#             (samples[:, 1] < 0.5) &
#             (samples[:, 2] < 0.5)
#         )
#         non_red.append(samples[~is_red])

#     non_red = np.concatenate(non_red)[:n]

#     x = np.vstack([red, non_red]).astype(np.float32)
#     y = np.concatenate([
#         np.ones(n), np.zeros(n)
#     ]).reshape(-1, 1).astype(np.float32)

#     order = rng.permutation(len(x))
#     x, y = x[order], y[order]

#     w1 = rng.normal(0, 0.5, (3, 8)).astype(np.float32)
#     b1 = np.zeros((1, 8), dtype=np.float32)
#     w2 = rng.normal(0, 0.5, (8, 1)).astype(np.float32)
#     b2 = np.zeros((1, 1), dtype=np.float32)

#     for _ in range(100):
#         z1 = x @ w1 + b1
#         h1 = np.maximum(0, z1)
#         output = sigmoid(h1 @ w2 + b2)

#         dz2 = output - y
#         dw2 = h1.T @ dz2 / len(x)
#         db2 = np.mean(dz2, axis=0, keepdims=True)

#         dz1 = (dz2 @ w2.T) * (z1 > 0)
#         dw1 = x.T @ dz1 / len(x)
#         db1 = np.mean(dz1, axis=0, keepdims=True)

#         w1 -= 0.3 * dw1
#         b1 -= 0.3 * db1
#         w2 -= 0.3 * dw2
#         b2 -= 0.3 * db2

#     accuracy = np.mean((output >= 0.5) == (y >= 0.5))
#     print(f"[AI] Training accuracy: {accuracy * 100:.1f}%")

#     return w1, b1, w2, b2


# AI_W1, AI_B1, AI_W2, AI_B2 = make_ai_model()


# def predict_red_mask(image):
#     h, w = image.shape[:2]
#     pixels = image.reshape(-1, 3).astype(np.float32) / 255.0

#     hidden = np.maximum(0, pixels @ AI_W1 + AI_B1)
#     probability = sigmoid(hidden @ AI_W2 + AI_B2)

#     return (probability[:, 0] > 0.75).reshape(h, w)

# class ColorTracker(Node):
#     def __init__(self):
#         super().__init__('color_tracker')

#         self._sub = self.create_subscription(
#             Image, '/camera/image_raw', self._on_image, 10
#         )
#         self._pub = self.create_publisher(Twist, '/cmd_vel', 10)

#         self.state = 'SEARCH'
#         self.state_count = 0
#         self._frame_count = 0

#         self.get_logger().info('ColorTracker started')

#     def _on_image(self, msg):
#         w, h = msg.width, msg.height
#         if w == 0 or h == 0:
#             return

#         arr = np.asarray(
#             list(msg.data), dtype=np.uint8
#         ).reshape(h, w, 3)

#         # 赤色領域を検出
#         # red_mask = (
#         #     (arr[:, :, 0] > 70) &
#         #     (arr[:, :, 1] < 95) &
#         #     (arr[:, :, 2] < 95)
#         # )
#         red_mask = predict_red_mask(arr)

#         # AI判定の診断：画像の代表画素を確認
#         if self._frame_count % 30 == 0:
#             points = [
#                 (h // 2, w // 2),
#                 (h // 4, w // 4),
#                 (h // 4, 3 * w // 4),
#                 (3 * h // 4, w // 4),
#                 (3 * h // 4, 3 * w // 4),
#             ]

#             print("[AI DEBUG] RGB / Red prediction")
#             for py, px in points:
#                 rgb = arr[py, px].tolist()
#                 result = int(red_mask[py, px])
#                 print(f"  ({px},{py}) RGB={rgb} Red={result}")

#         red_count = int(np.sum(red_mask))
#         area_frac = red_count / (w * h)

#         twist = Twist()

#         # SEARCH: 赤い物体を探す
#         if self.state == 'SEARCH':
#             twist.angular.z = 0.4

#             if red_count >= 50:
#                 self.state = 'APPROACH'
#                 self.state_count = 0

#         # APPROACH: 赤い物体に接近する
#         elif self.state == 'APPROACH':
#             if red_count < 50:
#                 self.state = 'SEARCH'
#                 twist.angular.z = 0.4
#             else:
#                 ys, xs = np.where(red_mask)
#                 cx = float(np.mean(xs))
#                 offset = (cx - w / 2.0) / (w / 2.0)

#                 if area_frac >= 0.12:
#                     self.state = 'RETREAT'
#                     self.state_count = 0
#                     twist.linear.x = -0.12
#                 else:
#                     twist.linear.x = max(
#                         0.05, 0.3 * (1.0 - area_frac * 5)
#                     )
#                     twist.angular.z = -0.8 * offset

#         # RETREAT: 一定時間後退する
#         elif self.state == 'RETREAT':
#             twist.linear.x = -0.12
#             self.state_count += 1

#             if self.state_count >= 15:
#                 self.state = 'TURN'
#                 self.state_count = 0

#         # TURN: 向きを変えて再探索する
#         elif self.state == 'TURN':
#             twist.angular.z = -0.4
#             self.state_count += 1

#             if self.state_count >= 15:
#                 self.state = 'SEARCH'
#                 self.state_count = 0

#         self._pub.publish(twist)

#         if self._frame_count % 10 == 0:
#             self.get_logger().info(
#                 f'State={self.state} '
#                 f'Red={red_count} '
#                 f'Area={area_frac * 100:.1f}%'
#             )

#         self._frame_count += 1


# def main():
#     rclpy.init()
#     node = ColorTracker()

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         # 停止指令を送ってから終了
#         node._pub.publish(Twist())
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()
#**********************************************************
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def is_red_rule(rgb):
    """教師ラベル用: 赤の定義 (RGB順, 0-1)"""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return (r > 0.3) & (g < 0.4 * r) & (b < 0.4 * r)


def make_ai_model():
    rng = np.random.default_rng(42)
    n = 6000

    # --- 赤(正例): 明るさ違いも含める ---
    r = rng.uniform(0.3, 1.0, n)
    g = r * rng.uniform(0.0, 0.35, n)
    b = r * rng.uniform(0.0, 0.35, n)
    red = np.column_stack([r, g, b])

    # --- 非赤(負例): 一様乱数から赤を除外 + 青/緑/白/黒を明示的に追加 ---
    uni = rng.uniform(0.0, 1.0, (n * 3, 3))
    uni = uni[~is_red_rule(uni)][: n // 2]

    blue = np.column_stack([
        rng.uniform(0.0, 0.3, n // 4),
        rng.uniform(0.0, 0.4, n // 4),
        rng.uniform(0.3, 1.0, n // 4),
    ])
    green = np.column_stack([
        rng.uniform(0.0, 0.3, n // 8),
        rng.uniform(0.3, 1.0, n // 8),
        rng.uniform(0.0, 0.4, n // 8),
    ])
    gray = np.repeat(rng.uniform(0.0, 1.0, (n // 8, 1)), 3, axis=1)

    non_red = np.vstack([uni, blue, green, gray])
    non_red = non_red[~is_red_rule(non_red)]

    x = np.vstack([red, non_red]).astype(np.float32)
    y = np.concatenate([np.ones(len(red)), np.zeros(len(non_red))])
    y = y.reshape(-1, 1).astype(np.float32)

    order = rng.permutation(len(x))
    x, y = x[order], y[order]

    # --- 2層NN (3 -> 16 -> 1) ---
    w1 = rng.normal(0, 0.5, (3, 16)).astype(np.float32)
    b1 = np.zeros((1, 16), dtype=np.float32)
    w2 = rng.normal(0, 0.5, (16, 1)).astype(np.float32)
    b2 = np.zeros((1, 1), dtype=np.float32)

    lr = 0.5
    for _ in range(3000):
        z1 = x @ w1 + b1
        h1 = np.maximum(0, z1)
        out = sigmoid(h1 @ w2 + b2)

        dz2 = (out - y) / len(x)
        dw2 = h1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        dz1 = (dz2 @ w2.T) * (z1 > 0)
        dw1 = x.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2

    acc = ((sigmoid(np.maximum(0, x @ w1 + b1) @ w2 + b2) > 0.5) == y).mean()
    print(f'[AI] training accuracy: {acc * 100:.1f}%')
    return w1, b1, w2, b2


AI_W1, AI_B1, AI_W2, AI_B2 = make_ai_model()


def predict_red_mask(rgb_image):
    """rgb_image は必ず RGB 順の (h, w, 3) uint8"""
    h, w = rgb_image.shape[:2]
    pixels = rgb_image.reshape(-1, 3).astype(np.float32) / 255.0
    hidden = np.maximum(0, pixels @ AI_W1 + AI_B1)
    prob = sigmoid(hidden @ AI_W2 + AI_B2)
    return (prob[:, 0] > 0.6).reshape(h, w)


def image_msg_to_rgb(msg):
    """Imageメッセージ -> RGB配列。BGR/RGB, step のパディングに対応"""
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ('rgb8', 'bgr8'):
        arr = data.reshape(h, msg.step)[:, : w * 3].reshape(h, w, 3)
        if enc == 'bgr8':
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
    if enc in ('rgba8', 'bgra8'):
        arr = data.reshape(h, msg.step)[:, : w * 4].reshape(h, w, 4)[:, :, :3]
        if enc == 'bgra8':
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
    return None


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
        self._logged_encoding = False

        self.get_logger().info('ColorTracker started')

    def _on_image(self, msg):
        w, h = msg.width, msg.height
        if w == 0 or h == 0:
            return

        if not self._logged_encoding:
            self.get_logger().info(f'image encoding = {msg.encoding}')
            self._logged_encoding = True

        arr = image_msg_to_rgb(msg)
        if arr is None:
            self.get_logger().warn(f'unsupported encoding: {msg.encoding}')
            return

        red_mask = predict_red_mask(arr)
        red_count = int(np.sum(red_mask))
        area_frac = red_count / (w * h)

        twist = Twist()

        if self.state == 'SEARCH':
            twist.angular.z = 0.6
            if red_count >= 15:
                self.state = 'APPROACH'
                self.state_count = 0

        elif self.state == 'APPROACH':
            if red_count < 8:
                self.state = 'SEARCH'
                twist.angular.z = 0.6
            else:
                ys, xs = np.where(red_mask)
                cx = float(np.mean(xs))
                offset = (cx - w / 2.0) / (w / 2.0)

                if area_frac >= 0.18:
                    self.state = 'RETREAT'
                    self.state_count = 0
                    twist.linear.x = -0.2
                else:
                    twist.linear.x = max(0.08, 0.4 * (1.0 - area_frac * 4))
                    twist.angular.z = -0.9 * offset

        elif self.state == 'RETREAT':
            twist.linear.x = -0.2
            self.state_count += 1
            if self.state_count >= 15:
                self.state = 'TURN'
                self.state_count = 0

        elif self.state == 'TURN':
            twist.angular.z = -0.6
            self.state_count += 1
            if self.state_count >= 15:
                self.state = 'SEARCH'
                self.state_count = 0

        self._pub.publish(twist)

        if self._frame_count % 10 == 0:
            self.get_logger().info(
                f'State={self.state} Red={red_count} Area={area_frac*100:.1f}%'
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
        node._pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()