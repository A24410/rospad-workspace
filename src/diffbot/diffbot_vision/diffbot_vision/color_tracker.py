import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# ============================================================
# 設定
# ============================================================
# True: ロボットは動かず、検出した塊の特徴値だけを出力する (データ収集用)
# False: 通常の動作
COLLECT_MODE = False

SCALE = 3                 # 画像の間引き (大きいほど速いが粗い)
THRESHOLD = 22            # 背景との明るさの差
MIN_AREA = 12             # 小さすぎる塊を無視 (間引き後の画素数)
SPHERE_PROB = 0.7         # 見つけるときのしきい値
KEEP_PROB = 0.4           # 追跡中のしきい値
RETREAT_AREA = 0.09       # これ以上の面積で「到着」(大きいほど近くで止まる)
NEAR_AREA = 0.04          # 見失う直前にこれ以上なら「到着」とみなす
LOST_LIMIT = 12           # 見失っても追跡を続けるフレーム数

# 箱らしい塊を除外する足切り (球体は充填率が低く、四隅が空く)
VETO_FILL = 0.88          # 充填率がこれを超えたら箱とみなす
VETO_CORNER = 0.70        # 四隅の充填率がこれを超えたら箱とみなす

# 旋回の向き。球体が画面の端へ逃げていくなら -1.0 にする
STEER_SIGN = 1.0
STEER_GAIN = 0.9

# 後退の量
RETREAT_FRAMES = 5        # 後退するフレーム数 (0にすると後退なし)
RETREAT_SPEED = -0.10     # 後退の速さ

# 「ここにあるよ〜」のプルプル動作
WIGGLE_FRAMES = 30        # プルプルする長さ (フレーム数)
WIGGLE_PERIOD = 3         # 何フレームごとに向きを反転するか
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