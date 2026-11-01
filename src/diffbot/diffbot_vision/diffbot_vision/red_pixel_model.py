import numpy as np

# ==========================================
# NumPy only: 小型ニューラルネットワーク
# RGB画素が赤色かどうかを分類する
# ==========================================

rng = np.random.default_rng(42)


def make_training_data(n=4000):
    """人工的な赤色・非赤色サンプルを生成する。"""

    # 赤色サンプル：明るさや色のばらつきを持たせる
    r = rng.uniform(0.65, 1.0, n)
    g = rng.uniform(0.05, 0.40, n)
    b = rng.uniform(0.04, 0.38, n)
    brightness = rng.uniform(0.35, 1.0, n)

    red = np.column_stack([r, g, b]) * brightness[:, None]
    red += rng.normal(0, 0.025, red.shape)
    red = np.clip(red, 0, 1)

    # 非赤色サンプル：赤色に該当しない色を生成
    non_red = []
    while sum(len(x) for x in non_red) < n:
        samples = rng.uniform(0, 1, (n, 3))
        is_red_like = (
            (samples[:, 0] > samples[:, 1] * 1.2) &
            (samples[:, 0] > samples[:, 2] * 1.2) &
            (samples[:, 1] < 0.5) &
            (samples[:, 2] < 0.5)
        )
        non_red.append(samples[~is_red_like])

    non_red = np.concatenate(non_red, axis=0)[:n]

    x = np.vstack([red, non_red]).astype(np.float32)
    y = np.concatenate([
        np.ones(n),
        np.zeros(n)
    ]).reshape(-1, 1).astype(np.float32)

    order = rng.permutation(len(x))
    return x[order], y[order]


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def train_model():
    """3入力・8隠れユニット・1出力のニューラルネットワークを学習。"""

    x, y = make_training_data()

    w1 = rng.normal(0, 0.5, (3, 8)).astype(np.float32)
    b1 = np.zeros((1, 8), dtype=np.float32)
    w2 = rng.normal(0, 0.5, (8, 1)).astype(np.float32)
    b2 = np.zeros((1, 1), dtype=np.float32)

    learning_rate = 0.3

    for epoch in range(100):
        # 順方向の計算
        z1 = x @ w1 + b1
        h1 = np.maximum(0, z1)
        output = sigmoid(h1 @ w2 + b2)

        # 誤差逆伝播
        dz2 = output - y
        dw2 = h1.T @ dz2 / len(x)
        db2 = np.mean(dz2, axis=0, keepdims=True)

        dh1 = dz2 @ w2.T
        dz1 = dh1 * (z1 > 0)

        dw1 = x.T @ dz1 / len(x)
        db1 = np.mean(dz1, axis=0, keepdims=True)

        # 重み更新
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2

    accuracy = np.mean((output >= 0.5) == (y >= 0.5))

    return w1, b1, w2, b2, accuracy


# 起動時にモデルを学習
W1, B1, W2, B2, TRAIN_ACCURACY = train_model()

print("[AI] Neural network training completed")
print(f"[AI] Training accuracy: {TRAIN_ACCURACY * 100:.1f}%")


def predict_red_mask(image):
    """
    カメラ画像の各画素をニューラルネットワークで推論する。
    image: H x W x 3 のRGB画像
    return: 赤色と判定した画素のマスク
    """

    h, w = image.shape[:2]

    pixels = image.reshape(-1, 3).astype(np.float32) / 255.0

    # ニューラルネットワークによる推論
    hidden = np.maximum(0, pixels @ W1 + B1)
    probabilities = sigmoid(hidden @ W2 + B2)

    mask = (probabilities[:, 0] > 0.75).reshape(h, w)

    return mask


if __name__ == "__main__":
    # テスト用の色を推論
    test_colors = np.array([
        [248, 81, 73],   # 明るい赤
        [100, 30, 25],   # 暗い赤
        [30, 180, 50],   # 緑
        [30, 50, 200],   # 青
        [200, 200, 200]  # 灰色
    ], dtype=np.float32) / 255.0

    hidden = np.maximum(0, test_colors @ W1 + B1)
    results = sigmoid(hidden @ W2 + B2)

    print("=== 推論テスト ===")
    for color, probability in zip(test_colors, results):
        print(
            f"RGB={np.round(color * 255).astype(int)} "
            f"Red probability={probability[0]:.3f}"
        )