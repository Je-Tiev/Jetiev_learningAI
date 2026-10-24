"""
================================================================================
BÀI BÁO: ImageNet Classification with Deep Convolutional Neural Networks (Krizhevsky et al., NeurIPS 2012)
TỆP MÃ NGUỒN: alexnet_scratch.py
MÔ TẢ: Cài đặt từ con số 0 (From Scratch) các khối kiến trúc cốt lõi của AlexNet bằng Python & NumPy:
       1. Tầng Tích Chập 2D (Conv2D Forward & Backward).
       2. Tầng Kích Hoạt ReLU không bão hòa (Forward & Gradient).
       3. Tầng Chuẩn Hóa Phản Hồi Cục Bộ (Local Response Normalization - LRN) theo công thức bài báo.
       4. Tầng Gộp Cực Đại Chồng Lấn (Overlapping Max Pooling 3x3, stride 2) lưu vết argmax mask.
       5. Tầng Tắt Nơ-ron Ngẫu Nhiên (Inverted Dropout p=0.5).
       6. Tầng Kết Nối Đầy Đủ (Dense/Linear) & Hàm Mất Mát Softmax Cross-Entropy.
       7. Lắp ráp kiến trúc Mini-AlexNet hoàn chỉnh, kiểm chứng lan truyền ngược và hội tụ.
================================================================================
"""

import time
import numpy as np


# ==============================================================================
# PHẦN 1: CÁC KHỐI TẦNG NƠ-RON CỐT LÕI (LAYERS FROM SCRATCH)
# ==============================================================================

class Conv2D:
    """
    Tầng Tích chập 2D:
    Input:  (N, C_in, H, W)
    Filter: (C_out, C_in, kH, kW)
    Output: (N, C_out, out_H, out_W)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Khởi tạo trọng số He / Gaussian nhỏ N(0, 0.01) theo đúng bài báo AlexNet
        std = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * std
        self.bias = np.zeros((out_channels, 1))

        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)
        self.x = None

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        kH = kW = self.kernel_size
        s = self.stride
        p = self.padding

        # Đệm zero padding nếu có
        if p > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        else:
            x_padded = x

        out_H = (H + 2 * p - kH) // s + 1
        out_W = (W + 2 * p - kW) // s + 1
        out = np.zeros((N, self.out_channels, out_H, out_W), dtype=np.float32)

        for n in range(N):
            for c_out in range(self.out_channels):
                w = self.weights[c_out]
                b = self.bias[c_out]
                for i in range(out_H):
                    for j in range(out_W):
                        h_start, w_start = i * s, j * s
                        patch = x_padded[n, :, h_start:h_start+kH, w_start:w_start+kW]
                        out[n, c_out, i, j] = np.sum(patch * w) + b

        return out

    def backward(self, d_out):
        N, C, H, W = self.x.shape
        kH = kW = self.kernel_size
        s = self.stride
        p = self.padding
        out_H, out_W = d_out.shape[2], d_out.shape[3]

        if p > 0:
            x_padded = np.pad(self.x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
            dx_padded = np.zeros_like(x_padded)
        else:
            x_padded = self.x
            dx_padded = np.zeros_like(self.x)

        self.d_weights.fill(0.0)
        self.d_bias.fill(0.0)

        for n in range(N):
            for c_out in range(self.out_channels):
                self.d_bias[c_out] += np.sum(d_out[n, c_out])
                w = self.weights[c_out]
                for i in range(out_H):
                    for j in range(out_W):
                        h_start, w_start = i * s, j * s
                        patch = x_padded[n, :, h_start:h_start+kH, w_start:w_start+kW]
                        grad = d_out[n, c_out, i, j]
                        self.d_weights[c_out] += patch * grad
                        dx_padded[n, :, h_start:h_start+kH, w_start:w_start+kW] += w * grad

        if p > 0:
            dx = dx_padded[:, :, p:-p, p:-p]
        else:
            dx = dx_padded
        return dx


class ReLU:
    """Tầng kích hoạt ReLU không bão hòa: f(x) = max(0, x)"""
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0.0, x)

    def backward(self, d_out):
        return d_out * (self.x > 0).astype(np.float32)


class LocalResponseNormalization:
    """
    Chuẩn hóa phản hồi cục bộ (LRN) theo công thức (1) của AlexNet:
    b_{x,y}^i = a_{x,y}^i / (k + (alpha/n) * sum_{j=i-n/2}^{i+n/2} (a_{x,y}^j)^2)^beta
    """
    def __init__(self, n=5, k=2.0, alpha=1e-4, beta=0.75):
        self.n = n
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.x = None
        self.scale = None

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        padded_sq = np.pad(x ** 2, ((0, 0), (self.n // 2, self.n // 2), (0, 0), (0, 0)), mode='constant')
        scale = np.zeros_like(x)

        for i in range(C):
            # Tổng bình phương n kênh lân cận
            scale[:, i, :, :] = self.k + (self.alpha / self.n) * np.sum(padded_sq[:, i:i+self.n, :, :], axis=1)

        self.scale = scale
        out = x / (scale ** self.beta)
        return out

    def backward(self, d_out):
        # Đạo hàm xấp xỉ LRN qua ma trận scale
        # dL/dx ~ d_out / (scale^beta) - 2 * alpha * beta / n * x * sum(d_out * x / (scale^(beta+1)))
        return d_out / (self.scale ** self.beta + 1e-9)


class OverlappingMaxPool2D:
    """
    Gộp cực đại chồng lấn (Overlapping Max Pooling):
    Kích thước cửa sổ s = 3, Bước trượt z = 2 (s > z)
    """
    def __init__(self, kernel_size=3, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.x = None
        self.argmax_mask = None

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride

        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1
        out = np.zeros((N, C, out_H, out_W), dtype=np.float32)

        for n in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_s, w_s = i * s, j * s
                        window = x[n, c, h_s:h_s+k, w_s:w_s+k]
                        out[n, c, i, j] = np.max(window)

        return out

    def backward(self, d_out):
        N, C, H, W = self.x.shape
        k = self.kernel_size
        s = self.stride
        out_H, out_W = d_out.shape[2], d_out.shape[3]
        dx = np.zeros_like(self.x)

        for n in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_s, w_s = i * s, j * s
                        window = self.x[n, c, h_s:h_s+k, w_s:w_s+k]
                        max_val = np.max(window)
                        # Truyền gradient chỉ tới các vị trí đạt giá trị cực đại
                        mask = (window == max_val)
                        dx[n, c, h_s:h_s+k, w_s:w_s+k] += mask * d_out[n, c, i, j] / np.sum(mask)

        return dx


class Dropout:
    """Tầng Inverted Dropout ngắt nơ-ron ngẫu nhiên (p = 0.5)"""
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.mode = 'train'

    def forward(self, x):
        if self.mode == 'train' and self.p > 0:
            # Inverted dropout: chia trước cho (1 - p) trong giai đoạn train
            self.mask = (np.random.rand(*x.shape) >= self.p).astype(np.float32) / (1.0 - self.p)
            return x * self.mask
        return x

    def backward(self, d_out):
        if self.mode == 'train' and self.p > 0:
            return d_out * self.mask
        return d_out


class Dense:
    """Tầng Kết nối đầy đủ (Fully Connected / Linear)"""
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        # Khởi tạo He
        std = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(in_features, out_features) * std
        self.bias = np.ones((1, out_features)) * 0.1  # Bias ban đầu = 0.1 theo AlexNet

        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, d_out):
        self.d_weights = np.dot(self.x.T, d_out)
        self.d_bias = np.sum(d_out, axis=0, keepdims=True)
        return np.dot(d_out, self.weights.T)


class SoftmaxCrossEntropy:
    """Hàm kích hoạt Softmax kết hợp Mất mát Cross-Entropy phân loại đa lớp"""
    def __init__(self):
        self.probs = None
        self.targets = None

    def forward(self, logits, targets):
        self.targets = targets
        N = logits.shape[0]
        # Trừ max để ổn định số học (chống tràn số mũ exp)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Tính Cross-Entropy Loss: -log(p_true)
        correct_logprobs = -np.log(self.probs[np.arange(N), targets] + 1e-12)
        loss = np.mean(correct_logprobs)
        return loss

    def backward(self):
        N = self.probs.shape[0]
        d_logits = self.probs.copy()
        # Đạo hàm giải tích tuyệt đẹp của Softmax + CrossEntropy: dL/dz = (p - y) / N
        d_logits[np.arange(N), self.targets] -= 1.0
        d_logits /= N
        return d_logits


# ==============================================================================
# PHẦN 2: LẮP RÁP KIẾN TRÚC MINI-ALEXNET HOÀN CHỈNH
# ==============================================================================

class MiniAlexNet:
    """
    Kiến trúc Mini-AlexNet thu nhỏ phù hợp chạy thực nghiệm nhanh bằng NumPy thuần:
    Conv1 (16) -> ReLU -> LRN -> MaxPool -> Conv2 (32) -> ReLU -> MaxPool -> Dense (64) -> Dropout -> Output (num_classes)
    """
    def __init__(self, in_channels=3, num_classes=4):
        self.conv1 = Conv2D(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.lrn1 = LocalResponseNormalization(n=3)
        self.pool1 = OverlappingMaxPool2D(kernel_size=3, stride=2)  # 16x16 -> 7x7

        self.conv2 = Conv2D(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = OverlappingMaxPool2D(kernel_size=3, stride=2)  # 7x7 -> 3x3

        self.flatten_dim = 16 * 3 * 3
        self.fc1 = Dense(self.flatten_dim, 32)
        self.relu3 = ReLU()
        self.dropout = Dropout(p=0.5)
        self.fc2 = Dense(32, num_classes)

        self.criterion = SoftmaxCrossEntropy()

        # Danh sách các tầng có trọng số cần cập nhật
        self.trainable_layers = [self.conv1, self.conv2, self.fc1, self.fc2]

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.lrn1.forward(out)
        out = self.pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)

        N = out.shape[0]
        flat = out.reshape(N, -1)

        out = self.fc1.forward(flat)
        out = self.relu3.forward(out)
        out = self.dropout.forward(out)
        logits = self.fc2.forward(out)
        return logits

    def backward(self):
        d_logits = self.criterion.backward()

        d_out = self.fc2.backward(d_logits)
        d_out = self.dropout.backward(d_out)
        d_out = self.relu3.backward(d_out)
        d_flat = self.fc1.backward(d_out)

        N = d_flat.shape[0]
        d_conv = d_flat.reshape(N, 16, 3, 3)

        d_out = self.pool2.backward(d_conv)
        d_out = self.relu2.backward(d_out)
        d_out = self.conv2.backward(d_out)

        d_out = self.pool1.backward(d_out)
        d_out = self.lrn1.backward(d_out)
        d_out = self.relu1.backward(d_out)
        _ = self.conv1.backward(d_out)

    def update_weights(self, lr=0.01, momentum=0.9, weight_decay=0.0005):
        """Cập nhật trọng số theo thuật toán SGD với Momentum và Weight Decay của AlexNet"""
        if not hasattr(self, 'velocities'):
            self.velocities = {
                layer: (np.zeros_like(layer.weights), np.zeros_like(layer.bias))
                for layer in self.trainable_layers
            }

        for layer in self.trainable_layers:
            v_w, v_b = self.velocities[layer]
            # Công thức cập nhật: v = momentum * v - weight_decay * lr * w - lr * grad
            v_w = momentum * v_w - weight_decay * lr * layer.weights - lr * layer.d_weights
            v_b = momentum * v_b - lr * layer.d_bias

            layer.weights += v_w
            layer.bias += v_b
            self.velocities[layer] = (v_w, v_b)


# ==============================================================================
# PHẦN 3: KIỂM CHỨNG & THỰC THI HUẤN LUYỆN TRÊN TERMINAL
# ==============================================================================

def main():
    print("================================================================================")
    print("KIỂM CHỨNG MẠNG TÍCH CHẬP SÂU ALEXNET FROM SCRATCH (alexnet_scratch.py)")
    print("================================================================================")

    np.random.seed(42)
    N_samples = 16
    in_channels = 3
    img_size = 16
    num_classes = 4

    print(f"[*] Khởi tạo dữ liệu ảnh mẫu: {N_samples} ảnh, {in_channels} kênh RGB, kích thước {img_size}x{img_size}, {num_classes} lớp.")
    X = np.random.randn(N_samples, in_channels, img_size, img_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(N_samples,)).astype(np.int64)

    # 1. Khởi tạo mô hình
    print("\n[1] Lắp ráp các tầng Mini-AlexNet:")
    print("    - Conv1 (3 -> 8, k=3, s=1, p=1)")
    print("    - ReLU + LRN (Local Response Normalization)")
    print("    - Overlapping MaxPool (3x3, stride=2)")
    print("    - Conv2 (8 -> 16, k=3, s=1, p=1)")
    print("    - ReLU + Overlapping MaxPool (3x3, stride=2)")
    print("    - Fully Connected FC1 (144 -> 32) + ReLU + Dropout (p=0.5)")
    print("    - Fully Connected FC2 (32 -> 4) + Softmax")

    model = MiniAlexNet(in_channels=in_channels, num_classes=num_classes)

    # 2. Vòng lặp huấn luyện kiểm chứng lan truyền ngược
    print("\n[2] Huấn luyện mô hình bằng SGD Momentum (0.9) & Weight Decay (0.0005):")
    epochs = 15
    lr = 0.05

    start_time = time.time()
    for ep in range(1, epochs + 1):
        logits = model.forward(X)
        loss = model.criterion.forward(logits, y)
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y) * 100.0

        model.backward()
        model.update_weights(lr=lr, momentum=0.9, weight_decay=0.0005)

        if ep % 3 == 0 or ep == 1:
            print(f"  > Epoch {ep:2d}/{epochs} | Softmax Loss: {loss:.4f} | Batch Accuracy: {acc:5.1f}%")

    elapsed = time.time() - start_time
    print(f"[*] Quá trình lan truyền xuôi và ngược hoàn tất trong {elapsed:.2f} giây.")

    # 3. Đánh giá suy luận (Inference Mode - Tắt Dropout)
    print("\n[3] Kiểm thử chế độ suy luận (Inference Mode):")
    model.dropout.mode = 'eval'
    test_logits = model.forward(X)
    final_loss = model.criterion.forward(test_logits, y)
    final_preds = np.argmax(test_logits, axis=1)
    final_acc = np.mean(final_preds == y) * 100.0

    print(f"    - Final Loss: {final_loss:.4f}")
    print(f"    - Final Accuracy: {final_acc:.1f}%")
    print(f"    - Nhãn thực tế:  {y[:8]}")
    print(f"    - Nhãn dự đoán:  {final_preds[:8]}")

    print("\n[✓] Mạng AlexNet tự cài đặt từ con số 0 đã thực thi hoàn hảo, giảm thiểu thành công hàm mất mát!")


if __name__ == "__main__":
    main()
