
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class SphereTracker(Node):

    def __init__(self):
        super().__init__('color_tracker')

        self._sub = self.create_subscription(
            Image, '/camera/image_raw',
            self._on_image, 10
        )
        self._pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        self.state = 'SEARCH'
        self.state_count = 0
        self.frame_count = 0

        self.detect_hits = 0
        self.detect_misses = 0

        self.last_cx = None
        self.last_area = 0.0

        # 画像処理設定
        self.scale = 3
        self.threshold = 22
        self.min_component_area = 12

        # 制御設定
        self.orbit_speed = 0.12
        self.search_speed = 0.35
        self.turn_speed = 0.35
        self.retreat_speed = -0.10

        self.get_logger().info(
            'Object巡回ロボット started'
        )

    # -----------------------------------------
    # 複数物体の抽出
    # -----------------------------------------

    def detect_objects(self, image):

        h, w = image.shape[:2]

        gray = (
            0.299 * image[:, :, 0] +
            0.587 * image[:, :, 1] +
            0.114 * image[:, :, 2]
        )

        small = gray[::self.scale, ::self.scale]
        sh, sw = small.shape

        border = np.concatenate([
            small[0, :],
            small[-1, :],
            small[:, 0],
            small[:, -1]
        ])

        background = np.median(border)

        mask = (
            np.abs(small - background)
            > self.threshold
        )

        visited = np.zeros(mask.shape, dtype=bool)

        objects = []

        for y in range(sh):
            for x in range(sw):

                if not mask[y, x] or visited[y, x]:
                    continue

                stack = [(y, x)]
                visited[y, x] = True

                xs = []
                ys = []
                perimeter = 0

                while stack:

                    cy, cx = stack.pop()

                    xs.append(cx)
                    ys.append(cy)

                    for dy, dx in (
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1)
                    ):

                        ny = cy + dy
                        nx = cx + dx

                        if not (
                            0 <= ny < sh and
                            0 <= nx < sw
                        ):
                            perimeter += 1
                            continue

                        if not mask[ny, nx]:
                            perimeter += 1
                            continue

                        if not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                area = len(xs)

                if area < self.min_component_area:
                    continue

                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)

                box_w = max_x - min_x + 1
                box_h = max_y - min_y + 1

                aspect = box_w / max(box_h, 1)
                fill = area / (box_w * box_h)

                # 極端に細長い領域や疎な領域を除外
                if aspect < 0.20 or aspect > 4.0:
                    continue

                if fill < 0.20:
                    continue

                cx = float(np.mean(xs) * self.scale)
                cy = float(np.mean(ys) * self.scale)

                area_fraction = (
                    area * self.scale * self.scale
                    / (h * w)
                )

                objects.append({
                    'cx': cx,
                    'cy': cy,
                    'area': area_fraction,
                    'aspect': aspect,
                    'fill': fill
                })

        return objects

    # -----------------------------------------
    # 物体群の中心位置
    # -----------------------------------------

    def get_scene_center(self, objects):

        if not objects:
            return None

        # 大きな物体ほど重みを大きくする
        weights = np.array([
            max(obj['area'], 0.001)
            for obj in objects
        ])

        xs = np.array([
            obj['cx'] for obj in objects
        ])

        ys = np.array([
            obj['cy'] for obj in objects
        ])

        cx = float(np.average(xs, weights=weights))
        cy = float(np.average(ys, weights=weights))

        total_area = float(np.sum(weights))

        return cx, cy, total_area

    # -----------------------------------------
    # メイン制御
    # -----------------------------------------

    def _on_image(self, msg):

        w = msg.width
        h = msg.height

        if w == 0 or h == 0:
            return

        arr = np.asarray(
            list(msg.data),
            dtype=np.uint8
        ).reshape(h, w, 3)

        objects = self.detect_objects(arr)
        center = self.get_scene_center(objects)

        twist = Twist()

        count = len(objects)

        if count > 0:
            self.detect_hits += 1
            self.detect_misses = 0
        else:
            self.detect_misses += 1
            self.detect_hits = 0

        # -------------------------------------
        # SEARCH：物体を探す
        # -------------------------------------

        if self.state == 'SEARCH':

            if self.detect_hits >= 2:
                self.state = 'ORBIT'
                self.state_count = 0

            else:
                twist.angular.z = self.search_speed

        # -------------------------------------
        # ORBIT：物体の周囲を巡回
        # -------------------------------------

        elif self.state == 'ORBIT':

            if self.detect_misses >= 4:

                self.state = 'SEARCH'
                self.state_count = 0

                twist.angular.z = self.search_speed

            elif center is not None:

                cx, cy, total_area = center

                offset = (
                    cx - w / 2.0
                ) / (w / 2.0)

                # 物体群の中心が画面中央に来るよう調整
                twist.angular.z = -0.7 * offset

                # 近づきすぎたら後退
                if total_area >= 0.16:

                    self.state = 'RETREAT'
                    self.state_count = 0

                    twist.linear.x = self.retreat_speed

                else:

                    # 巡回のため前進しながら旋回
                    twist.linear.x = self.orbit_speed

                    # 物体が画面端にある場合は旋回を強める
                    if abs(offset) > 0.65:
                        twist.angular.z = -0.9 * offset

        # -------------------------------------
        # RETREAT：後退
        # -------------------------------------

        elif self.state == 'RETREAT':

            twist.linear.x = self.retreat_speed
            self.state_count += 1

            if self.state_count >= 12:

                self.state = 'TURN'
                self.state_count = 0

        # -------------------------------------
        # TURN：向きを変える
        # -------------------------------------

        elif self.state == 'TURN':

            twist.angular.z = -self.turn_speed
            self.state_count += 1

            if self.state_count >= 15:

                self.state = 'SEARCH'
                self.state_count = 0

                self.detect_hits = 0
                self.detect_misses = 0

        self._pub.publish(twist)

        self.frame_count += 1

        if self.frame_count % 10 == 0:

            if center is None:
                self.get_logger().info(
                    f'State={self.state} Objects=0'
                )
            else:
                cx, cy, total_area = center

                self.get_logger().info(
                    f'State={self.state} '
                    f'Objects={count} '
                    f'Center=({cx:.0f},{cy:.0f}) '
                    f'Area={total_area*100:.1f}%'
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
