import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class ColorTracker(Node):

    def __init__(self):
        super().__init__('color_tracker')

        self._sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self._on_image,
            10
        )

        self._pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.state = 'SEARCH'
        self.state_count = 0
        self._frame_count = 0

        self.get_logger().info(
            'SphereTracker started'
        )

    # ----------------------------------
    # 球体候補の検出
    # ----------------------------------

    def detect_sphere(self, image):

        h, w = image.shape[:2]

        # グレースケール化
        gray = (
            0.299 * image[:, :, 0] +
            0.587 * image[:, :, 1] +
            0.114 * image[:, :, 2]
        )

        # 処理負荷を下げるため縮小
        small = gray[::3, ::3]

        sh, sw = small.shape

        # 画像の外周部分から背景の明るさを推定
        border = np.concatenate([
            small[0, :],
            small[-1, :],
            small[:, 0],
            small[:, -1]
        ])

        background = np.median(border)

        # 背景との明るさの差で物体候補を抽出
        mask = np.abs(small - background) > 22

        # 連結成分を探索
        visited = np.zeros(mask.shape, dtype=bool)

        best_component = None
        best_area = 0

        for y in range(sh):
            for x in range(sw):

                if not mask[y, x] or visited[y, x]:
                    continue

                stack = [(y, x)]
                visited[y, x] = True

                points = []

                while stack:
                    cy, cx = stack.pop()
                    points.append((cy, cx))

                    for dy, dx in (
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1)
                    ):
                        ny = cy + dy
                        nx = cx + dx

                        if (
                            0 <= ny < sh and
                            0 <= nx < sw and
                            mask[ny, nx] and
                            not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                area = len(points)

                if area > best_area:
                    best_area = area
                    best_component = points

        if best_component is None or best_area < 12:
            return None

        points = np.array(best_component)

        ys = points[:, 0]
        xs = points[:, 1]

        min_x = np.min(xs)
        max_x = np.max(xs)
        min_y = np.min(ys)
        max_y = np.max(ys)

        box_w = max_x - min_x + 1
        box_h = max_y - min_y + 1

        # 縦横比
        aspect = box_w / max(box_h, 1)

        # バウンディングボックス内の充填率
        fill = best_area / (box_w * box_h)

        # 球体は正面画像で概ね円形になる
        if not (0.65 <= aspect <= 1.35):
            return None

        if fill < 0.45:
            return None

        # 重心を元画像の座標に戻す
        cx = float(np.mean(xs) * 3)
        cy = float(np.mean(ys) * 3)

        # 元画像に対する占有面積
        area_fraction = (
            best_area * 9 / (h * w)
        )

        return cx, cy, area_fraction

    # ----------------------------------
    # ロボット制御
    # ----------------------------------

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

        if self.state == 'SEARCH':

            if result is None:
                twist.angular.z = 0.4

            else:
                self.state = 'APPROACH'
                self.state_count = 0

        elif self.state == 'APPROACH':

            if result is None:
                self.state = 'SEARCH'
                twist.angular.z = 0.4

            else:
                cx, cy, area = result

                offset = (
                    cx - w / 2.0
                ) / (w / 2.0)

                # 球体が十分大きく映ったら後退
                if area >= 0.12:
                    self.state = 'RETREAT'
                    self.state_count = 0
                    twist.linear.x = -0.12

                else:
                    twist.linear.x = max(
                        0.05,
                        0.3 * (1.0 - area * 5)
                    )

                    twist.angular.z = -0.8 * offset

        elif self.state == 'RETREAT':

            twist.linear.x = -0.12
            self.state_count += 1

            if self.state_count >= 15:
                self.state = 'TURN'
                self.state_count = 0

        elif self.state == 'TURN':

            twist.angular.z = -0.4
            self.state_count += 1

            if self.state_count >= 15:
                self.state = 'SEARCH'
                self.state_count = 0

        self._pub.publish(twist)

        self._frame_count += 1

        if self._frame_count % 10 == 0:

            if result is None:
                self.get_logger().info(
                    f'State={self.state} Sphere=NOT_FOUND'
                )

            else:
                cx, cy, area = result

                self.get_logger().info(
                    f'State={self.state} '
                    f'Sphere=FOUND '
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
        stop = Twist()
        node._pub.publish(stop)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()