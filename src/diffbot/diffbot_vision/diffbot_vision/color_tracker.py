import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# ============================================================
# 設定
# ============================================================
SCALE = 3                 # 画像の間引き (大きいほど速いが粗い)
THRESHOLD = 22            # 背景との明るさの差
MIN_AREA = 12             # 小さすぎる塊を無視 (間引き後の画素数)
SPHERE_PROB = 0.6         # この確率以上なら球体と判定
RETREAT_AREA = 0.10       # 球体の面積がこれ以上なら後退


# ============================================================
# 小さなNN: 形の特徴 (fill, corner_fill, aspect) -> 球体らしさ
# ============================================================
def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def make_ai_model():
    rng = np.random.default_rng(0)
    n = 3000

    # 球体: 充填率 約0.78, 四隅が空く, ほぼ正方形
    sphere = np.column_stack([
        rng.normal(0.78, 0.06, n),
        rng.normal(0.25, 0.12, n),
        rng.normal(1.0, 0.12, n),
    ])

    # 箱: 充填率が高く, 四隅も埋まる
    box = np.column_stack([
        rng.normal(0.95, 0.05, n),
        rng.normal(0.90, 0.08, n),
        rng.uniform(0.5, 2.0, n),
    ])

    # 欠けた塊 / ノイズ: 充填率が低い
    junk = np.column_stack([
        rng.uniform(0.2, 0.65, n),
        rng.uniform(0.0, 1.0, n),
        rng.uniform(0.3, 3.0, n),
    ])

    x = np.vstack([sphere, box, junk]).astype(np.float32)
    y = np.concatenate([
        np.ones(n), np.zeros(n), np.zeros(n)
    ]).reshape(-1, 1).astype(np.float32)

    order = rng.permutation(len(x))
    x, y = x[order], y[order]

    w1 = rng.normal(0, 0.5, (3, 12)).astype(np.float32)
    b1 = np.zeros((1, 12), dtype=np.float32)
    w2 = rng.normal(0, 0.5, (12, 1)).astype(np.float32)
    b2 = np.zeros((1, 1), dtype=np.float32)

    lr = 0.5
    for _ in range(2500):
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

    pred = sigmoid(np.maximum(0, x @ w1 + b1) @ w2 + b2) > 0.5
    print(f'[AI] training accuracy: {(pred == y).mean() * 100:.1f}%')
    return w1, b1, w2, b2


AI_W1, AI_B1, AI_W2, AI_B2 = make_ai_model()


def sphere_probability(fill, corner_fill, aspect):
    f = np.array([[fill, corner_fill, aspect]], dtype=np.float32)
    h = np.maximum(0, f @ AI_W1 + AI_B1)
    return float(sigmoid(h @ AI_W2 + AI_B2)[0, 0])


# ============================================================
# 画像変換
# ============================================================
def image_msg_to_rgb(msg):
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


# ============================================================
# 物体抽出 + 形の特徴
# ============================================================
def corner_fill_ratio(crop):
    """外接矩形の四隅(各辺の1/4)の塗りつぶし率の平均"""
    bh, bw = crop.shape
    ch = max(1, bh // 4)
    cw = max(1, bw // 4)
    corners = [
        crop[:ch, :cw], crop[:ch, -cw:],
        crop[-ch:, :cw], crop[-ch:, -cw:],
    ]
    return float(np.mean([c.mean() for c in corners]))


def detect_blobs(rgb):
    h, w = rgb.shape[:2]
    gray = (
        0.299 * rgb[:, :, 0] +
        0.587 * rgb[:, :, 1] +
        0.114 * rgb[:, :, 2]
    )
    small = gray[::SCALE, ::SCALE]
    sh, sw = small.shape

    # 行ごとの中央値を背景とする (地平線や床の明暗に強い)
    background = np.median(small, axis=1, keepdims=True)
    mask = np.abs(small - background) > THRESHOLD

    visited = np.zeros(mask.shape, dtype=bool)
    blobs = []

    for y in range(sh):
        for x in range(sw):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            xs, ys = [], []

            while stack:
                cy, cx = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < sh and 0 <= nx < sw
                            and mask[ny, nx] and not visited[ny, nx]):
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            area = len(xs)
            if area < MIN_AREA:
                continue

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bw_ = max_x - min_x + 1
            bh_ = max_y - min_y + 1

            crop = np.zeros((bh_, bw_), dtype=bool)
            crop[np.array(ys) - min_y, np.array(xs) - min_x] = True

            fill = area / (bw_ * bh_)
            corner = corner_fill_ratio(crop)
            aspect = bw_ / bh_
            prob = sphere_probability(fill, corner, aspect)

            blobs.append({
                'cx': float(np.mean(xs) * SCALE),
                'area': area * SCALE * SCALE / (h * w),
                'fill': fill,
                'corner': corner,
                'aspect': aspect,
                'prob': prob,
            })

    return blobs


# ============================================================
# ノード
# ============================================================
class SphereTracker(Node):
    def __init__(self):
        super().__init__('sphere_tracker')
        self._sub = self.create_subscription(
            Image, '/camera/image_raw', self._on_image, 10
        )
        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.state = 'SEARCH'
        self.state_count = 0
        self.frame_count = 0
        self.hits = 0
        self.misses = 0

        self.get_logger().info('SphereTracker started')

    def _on_image(self, msg):
        w, h = msg.width, msg.height
        if w == 0 or h == 0:
            return

        rgb = image_msg_to_rgb(msg)
        if rgb is None:
            self.get_logger().warn(f'unsupported encoding: {msg.encoding}')
            return

        blobs = detect_blobs(rgb)
        spheres = [b for b in blobs if b['prob'] >= SPHERE_PROB]
        target = max(spheres, key=lambda b: b['area']) if spheres else None

        if target is not None:
            self.hits += 1
            self.misses = 0
        else:
            self.misses += 1
            self.hits = 0

        twist = Twist()

        if self.state == 'SEARCH':
            twist.angular.z = 0.4
            if self.hits >= 2:
                self.state = 'APPROACH'
                self.state_count = 0

        elif self.state == 'APPROACH':
            if self.misses >= 4 or target is None:
                if self.misses >= 4:
                    self.state = 'SEARCH'
                    twist.angular.z = 0.4
            else:
                offset = (target['cx'] - w / 2.0) / (w / 2.0)
                if target['area'] >= RETREAT_AREA:
                    self.state = 'RETREAT'
                    self.state_count = 0
                    twist.linear.x = -0.15
                else:
                    twist.linear.x = max(0.06, 0.3 * (1.0 - target['area'] * 4))
                    twist.angular.z = -0.9 * offset

        elif self.state == 'RETREAT':
            twist.linear.x = -0.15
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
                self.hits = 0
                self.misses = 0

        self._pub.publish(twist)

        # 10フレームごとに、検出した塊ごとの特徴をログ出力
        if self.frame_count % 10 == 0:
            self.get_logger().info(
                f'State={self.state} Blobs={len(blobs)} Spheres={len(spheres)}'
            )
            for b in blobs[:5]:
                self.get_logger().info(
                    f'  blob fill={b["fill"]:.2f} corner={b["corner"]:.2f} '
                    f'aspect={b["aspect"]:.2f} area={b["area"]*100:.1f}% '
                    f'P(sphere)={b["prob"]:.2f}'
                )
        self.frame_count += 1


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