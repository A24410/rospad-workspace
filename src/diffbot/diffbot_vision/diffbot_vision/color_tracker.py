import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


# ==================================================
# NumPyだけで動く小型ニューラルネットワーク
# 入力：縦横比、充填率、円形度
# 出力：球体らしさ（0～1）
# ==================================================

class SphereAI:

    def __init__(self):
        rng = np.random.RandomState(42)

        # 合成した形状特徴データで学習
        n = 600

        # 球体の特徴データ
        positive = np.column_stack([
            rng.uniform(0.82, 1.22, n),
            rng.uniform(0.55, 0.95, n),
            rng.uniform(0.55, 0.95, n)
        ])

        # 球体以外の特徴データ
        negative = np.column_stack([
            rng.uniform(0.30, 2.50, n),
            rng.uniform(0.10, 0.99, n),
            rng.uniform(0.05, 0.95, n)
        ])

        x = np.vstack([positive, negative])
        y = np.vstack([
            np.ones((n, 1)),
            np.zeros((n, 1))
        ])

        # 学習データをシャッフル
        order = rng.permutation(len(x))
        x = x[order]
        y = y[order]

        # 特徴量を正規化
        x[:, 0] = (x[:, 0] - 1.0) / 1.5

        # 3入力 → 8中間層 → 1出力
        self.w1 = rng.normal(0, 0.3, (3, 8))
        self.b1 = np.zeros((1, 8))
        self.w2 = rng.normal(0, 0.3, (8, 1))
        self.b2 = np.zeros((1, 1))

        # 学習
        lr = 0.15

        for _ in range(300):
            z1 = np.maximum(0, x @ self.w1 + self.b1)
            z2 = np.clip(z1 @ self.w2 + self.b2, -30, 30)
            pred = 1.0 / (1.0 + np.exp(-z2))

            dz2 = (pred - y) / len(x)
            dw2 = z1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            dz1 = (dz2 @ self.w2.T) * (z1 > 0)
            dw1 = x.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            self.w2 -= lr * dw2
            self.b2 -= lr * db2
            self.w1 -= lr * dw1
            self.b1 -= lr * db1

    def predict(self, aspect, fill, circularity):

        x = np.array([[
            (aspect - 1.0) / 1.5,
            fill,
            circularity
        ]])

        z1 = np.maximum(0, x @ self.w1 + self.b1)
        z2 = np.clip(z1 @ self.w2 + self.b2, -30, 30)

        return float(1.0 / (1.0 + np.exp(-z2))[0, 0])


# ==================================================
# ロボット
# ==================================================

class SphereTracker(Node):

    def __init__(self):
        super().__init__('color_tracker')

        self._sub = self.create_subscription(
            Image, '/camera/image_raw', self._on_image, 10
        )

        self._pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        self.ai = SphereAI()

        self.state = 'SEARCH'
        self.state_count = 0
        self._frame_count = 0

        self.hit_count = 0
        self.miss_count = 0

        self.get_logger().info(
            'SphereTracker started - Neural Network ready'
        )

    # ----------------------------------------------
    # 球体候補抽出 ＋ AI推論
    # ----------------------------------------------

    def detect_sphere(self, image):

        h, w = image.shape[:2]

        gray = (
            0.299 * image[:, :, 0] +
            0.587 * image[:, :, 1] +
            0.114 * image[:, :, 2]
        )

        small = gray[::3, ::3]
        sh, sw = small.shape

        border = np.concatenate([
            small[0, :],
            small[-1, :],
            small[:, 0],
            small[:, -1]
        ])

        background = np.median(border)

        mask = np.abs(small - background) > 22

        visited = np.zeros(mask.shape, dtype=bool)

        best_component = None
        best_area = 0
        best_perimeter = 0

        for y in range(sh):
            for x in range(sw):

                if not mask[y, x] or visited[y, x]:
                    continue

                stack = [(y, x)]
                visited[y, x] = True

                points = []
                perimeter = 0

                while stack:
                    cy, cx = stack.pop()
                    points.append((cy, cx))

                    for dy, dx in (
                        (-1, 0), (1, 0),
                        (0, -1), (0, 1)
                    ):
                        ny = cy + dy
                        nx = cx + dx

                        if not (0 <= ny < sh and 0 <= nx < sw):
                            perimeter += 1
                            continue

                        if not mask[ny, nx]:
                            perimeter += 1
                            continue

                        if not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                area = len(points)

                if area > best_area:
                    best_area = area
                    best_component = points
                    best_perimeter = perimeter

        if best_component is None or best_area < 12:
            return None

        points = np.array(best_component)

        ys = points[:, 0]
        xs = points[:, 1]

        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)

        box_w = max_x - min_x + 1
        box_h = max_y - min_y + 1

        aspect = box_w / max(box_h, 1)
        fill = best_area / (box_w * box_h)

        circularity = (
            4.0 * np.pi * best_area /
            max(best_perimeter ** 2, 1)
        )

        # ニューラルネットワークによる推論
        score = self.ai.predict(
            aspect, fill, circularity
        )

        if score < 0.55:
            return None

        cx = float(np.mean(xs) * 3)
        cy = float(np.mean(ys) * 3)

        area_fraction = best_area * 9 / (h * w)

        return cx, cy, area_fraction, score

    # ----------------------------------------------
    # 移動制御
    # ----------------------------------------------

    def _on_image(self, msg):

        w = msg.width
        h = msg.height

        if w == 0 or h == 0:
            return

        arr = np.asarray(
            list(msg.data),
            dtype=np.uint8
        ).reshape(h, w, 3)

        result = self.detect_sphere(arr)

        twist = Twist()

        if result is not None:
            self.hit_count += 1
            self.miss_count = 0
        else:
            self.miss_count += 1
            self.hit_count = 0

        # SEARCH：球体を探す
        if self.state == 'SEARCH':

            if self.hit_count >= 2:
                self.state = 'APPROACH'
                self.state_count = 0
            else:
                twist.angular.z = 0.4

        # APPROACH：球体に近づく
        elif self.state == 'APPROACH':

            if self.miss_count >= 4:
                self.state = 'SEARCH'
                twist.angular.z = 0.4

            elif result is not None:

                cx, cy, area, score = result

                offset = (cx - w / 2.0) / (w / 2.0)

                # 接近しすぎたら後退
                if area >= 0.09:
                    self.state = 'RETREAT'
                    self.state_count = 0
                    twist.linear.x = -0.12

                else:
                    # 接近速度を距離に応じて調整
                    twist.linear.x = max(
                        0.05,
                        min(0.20, 0.25 * (1.0 - area * 4))
                    )

                    twist.angular.z = -0.8 * offset

        # RETREAT：一定時間後退
        elif self.state == 'RETREAT':

            twist.linear.x = -0.12
            self.state_count += 1

            if self.state_count >= 15:
                self.state = 'TURN'
                self.state_count = 0

        # TURN：旋回して再探索
        elif self.state == 'TURN':

            twist.angular.z = -0.4
            self.state_count += 1

            if self.state_count >= 15:
                self.state = 'SEARCH'
                self.state_count = 0
                self.hit_count = 0
                self.miss_count = 0

        self._pub.publish(twist)

        self._frame_count += 1

        if self._frame_count % 10 == 0:

            if result is None:
                self.get_logger().info(
                    f'State={self.state} Sphere=NOT_FOUND'
                )
            else:
                cx, cy, area, score = result

                self.get_logger().info(
                    f'State={self.state} '
                    f'Sphere=FOUND '
                    f'AI={score:.2f} '
                    f'Center=({cx:.0f},{cy:.0f}) '
                    f'Area={area*100:.1f}%'
                )


def main():

    rclpy.init()
    node = SphereTracker()

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