import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# ============================================================
# 設定
# ============================================================
SCALE = 2                 # 間引きを減らして、小さい物体を細かく見る (3 -> 2)
THRESHOLD = 18            # 背景との差を少し緩める (22 -> 18)
MIN_AREA = 20             # SCALEを小さくした分、最小の塊も増やす (12 -> 20)
SPHERE_PROB = 0.45        # 見つけるときのしきい値を緩める (0.6 -> 0.45)
KEEP_PROB = 0.3           # 追跡中のしきい値
RETREAT_AREA = 0.09       # これ以上の面積で「到着」(大きいほど近くで止まる)

# 後退の量 (短く・ゆっくりにすると、離れすぎない)
RETREAT_FRAMES = 0        # 後退するフレーム数
RETREAT_SPEED = -0.10     # 後退の速さ

NEAR_AREA = 0.04          # 見失う直前にこれ以上なら「到着」とみなす
LOST_LIMIT = 12           # 見失っても追跡を続けるフレーム数

# 旋回の向き。球体が画面の端へ逃げていくなら -1.0 にする
STEER_SIGN = 1.0
STEER_GAIN = 0.9

# 「ここにあるよ〜」のプルプル動作
WIGGLE_FRAMES = 30        # プルプルする長さ (フレーム数)
WIGGLE_PERIOD = 3         # 何フレームごとに向きを反転するか (小さいほど細かく震える)
WIGGLE_SPEED = 0.9        # 震える強さ (角速度)


# ============================================================
# 小さなNN: 形の特徴 (fill, corner_fill, aspect) -> 球体らしさ
# ============================================================
def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def make_ai_model():
    rng = np.random.default_rng(0)
    n = 3000

    # 球体: 実測ログに合わせ、影で横に伸びた形も含める
    sphere = np.column_stack([
        rng.normal(0.74, 0.08, n),
        rng.normal(0.30, 0.15, n),
        rng.normal(1.15, 0.30, n),
    ])
    # 箱: 充填率が高く、四隅も埋まる
    box = np.column_stack([
        rng.normal(0.95, 0.05, n),
        rng.normal(0.90, 0.08, n),
        rng.uniform(0.5, 2.0, n),
    ])
    # 欠けた塊 / ノイズ: 充填率が低い
    junk = np.column_stack([
        rng.uniform(0.2, 0.62, n),
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
        self.lost = 0
        self.last_cx = None
        self.last_offset = 0.0
        self.last_area = 0.0

        self.get_logger().info('SphereTracker started')

    def set_state(self, new_state, reason):
        self.get_logger().info(
            f'STATE {self.state} -> {new_state} ({reason})'
        )
        self.state = new_state
        self.state_count = 0
        self.hits = 0
        self.lost = 0

    def steer(self, offset):
        """球体を画面中央に寄せる旋回指令"""
        return STEER_SIGN * STEER_GAIN * offset

    def _on_image(self, msg):
        w, h = msg.width, msg.height
        if w == 0 or h == 0:
            return

        rgb = image_msg_to_rgb(msg)
        if rgb is None:
            self.get_logger().warn(f'unsupported encoding: {msg.encoding}')
            return

        if self.last_cx is None:
            self.last_cx = w / 2.0

        blobs = detect_blobs(rgb)

        # 追跡中はしきい値を下げ、前回の位置に近い球体を選ぶ
        tracking = (self.state == 'APPROACH')
        min_prob = KEEP_PROB if tracking else SPHERE_PROB
        cands = [b for b in blobs if b['prob'] >= min_prob]

        target = None
        if cands:
            if tracking:
                target = min(cands, key=lambda b: abs(b['cx'] - self.last_cx))
            else:
                target = max(cands, key=lambda b: b['area'])

        twist = Twist()

        if self.state == 'SEARCH':
            twist.angular.z = 0.35
            if target is not None:
                self.hits += 1
            else:
                self.hits = 0
            if self.hits >= 2:
                self.set_state('APPROACH', 'sphere found')

        elif self.state == 'APPROACH':
            if target is not None:
                self.lost = 0
                offset = (target['cx'] - w / 2.0) / (w / 2.0)
                self.last_offset = offset
                self.last_cx = target['cx']
                self.last_area = target['area']

                if target['area'] >= RETREAT_AREA:
                    self.set_state(
                        'WIGGLE',
                        f'arrived area={target["area"]*100:.1f}%'
                    )
                else:
                    twist.linear.x = max(0.06, 0.3 * (1.0 - target['area'] * 4))
                    twist.angular.z = self.steer(offset)
            else:
                self.lost += 1
                if self.lost >= LOST_LIMIT:
                    if self.last_area >= NEAR_AREA:
                        self.set_state(
                            'WIGGLE',
                            f'lost while near area={self.last_area*100:.1f}%'
                        )
                    else:
                        self.set_state('SEARCH', 'lost while far')
                else:
                    # 見失った直後は、最後に見えた方向へゆっくり進む
                    twist.linear.x = 0.05
                    twist.angular.z = self.steer(self.last_offset)

        elif self.state == 'WIGGLE':
            # その場で左右に小刻みに首を振る (「ここにあるよ〜」)
            direction = 1.0 if (self.state_count // WIGGLE_PERIOD) % 2 == 0 else -1.0
            twist.angular.z = direction * WIGGLE_SPEED
            self.state_count += 1
            if self.state_count >= WIGGLE_FRAMES:
                self.set_state('RETREAT', 'wiggle done')

        elif self.state == 'RETREAT':
            twist.linear.x = RETREAT_SPEED
            self.state_count += 1
            if self.state_count >= RETREAT_FRAMES:
                self.set_state('TURN', 'retreat done')

        elif self.state == 'TURN':
            twist.angular.z = -0.4
            self.state_count += 1
            if self.state_count >= 15:
                self.set_state('SEARCH', 'turn done')

        self._pub.publish(twist)

        if self.frame_count % 10 == 0:
            self.get_logger().info(
                f'State={self.state} Blobs={len(blobs)} '
                f'Target={"yes" if target else "no"} lost={self.lost} '
                f'cmd(x={twist.linear.x:.2f}, z={twist.angular.z:.2f})'
            )
            for b in blobs[:5]:
                off = (b['cx'] - w / 2.0) / (w / 2.0)
                self.get_logger().info(
                    f'  blob fill={b["fill"]:.2f} corner={b["corner"]:.2f} '
                    f'aspect={b["aspect"]:.2f} area={b["area"]*100:.1f}% '
                    f'offset={off:+.2f} P(sphere)={b["prob"]:.2f}'
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