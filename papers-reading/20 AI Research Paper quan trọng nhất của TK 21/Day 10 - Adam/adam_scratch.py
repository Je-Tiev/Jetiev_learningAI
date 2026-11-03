"""
================================================================================
BÀI BÁO: Adam: A Method for Stochastic Optimization (Kingma & Ba, ICLR 2015)
TỆP MÃ NGUỒN: adam_scratch.py
MÔ TẢ DÀNH CHO NGƯỜI MỚI BẮT ĐẦU:
    Tệp mã nguồn này tự xây dựng từ con số 0 (From Scratch) bằng thuần Python và NumPy
    toàn bộ các thuật toán tối ưu hóa kinh điển trong lịch sử Học máy và Học sâu:

    1. Lớp Bộ Tối Ưu Hóa Cơ Bản (Base Optimizer & Classical Baselines):
       - SGD (Stochastic Gradient Descent)
       - SGD + Momentum (Quán tính Polyak)
       - AdaGrad (Duchi et al., 2011 - Tốc độ học thích ứng qua tổng dồn bậc 2)
       - RMSprop (Hinton, 2012 - Trung bình động hàm mũ bậc 2)

    2. Thuật Toán Đột Phá Adam (Kingma & Ba, 2014):
       - Ước lượng mô-men bậc 1: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
       - Ước lượng mô-men bậc 2: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
       - Hiệu chỉnh độ chệch khởi tạo (Bias Correction): m_hat, v_hat
       - Cập nhật tham số theo tỷ số Tín hiệu trên Nhiễu (SNR)

    3. Bộ Tối Ưu Hiện Đại Kế Thừa: AdamW (Loshchilov & Hutter, 2017):
       - Tách biệt hoàn toàn việc phân rã trọng số (Decoupled Weight Decay) khỏi gradient.

    4. Khối Thực Nghiệm Đối Chiếu Giá Trị Học Thuật (Academic Value Benchmark):
       - Thử thách trên địa hình hẻm núi dốc lệch (Ill-conditioned Valley / Ravine).
       - Thử nghiệm trên bài toán huấn luyện Mạng Nơ-ron Đa Tầng (MLP).
================================================================================
"""

import time
import numpy as np


# ==============================================================================
# PHẦN 1: CÀI ĐẶT CÁC BỘ TỐI ƯU HÓA TỪ ĐẦU (OPTIMIZERS FROM SCRATCH)
# ==============================================================================

class BaseOptimizer:
    """Lớp cơ sở trừu tượng cho tất cả các bộ tối ưu hóa."""
    def __init__(self, lr=0.01):
        self.lr = lr
        self.t = 0 # Đếm bước lặp thời gian

    def step(self, params, grads):
        raise NotImplementedError


class SGDOptimizer(BaseOptimizer):
    """
    1. Vanilla SGD (Robbins-Monro 1951) & SGD có Quán tính (Momentum - Polyak 1964):
    - Không quán tính: theta_t = theta_{t-1} - lr * g_t
    - Có quán tính:     v_t = momentum * v_{t-1} + lr * g_t
                        theta_t = theta_{t-1} - v_t
    """
    def __init__(self, lr=0.01, momentum=0.0):
        super().__init__(lr=lr)
        self.momentum = momentum
        self.v = {}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            if k not in self.v:
                self.v[k] = np.zeros_like(params[k])
            
            if self.momentum > 0.0:
                self.v[k] = self.momentum * self.v[k] + self.lr * grads[k]
                params[k] -= self.v[k]
            else:
                params[k] -= self.lr * grads[k]


class AdaGradOptimizer(BaseOptimizer):
    """
    2. AdaGrad (Duchi et al., 2011):
    - Tích lũy tổng bình phương gradient: G_t = G_{t-1} + g_t^2
    - Cập nhật tham số: theta_t = theta_{t-1} - (lr / (sqrt(G_t) + eps)) * g_t
    - Điểm yếu: G_t tăng đơn điệu không giới hạn, làm tốc độ học hiệu dụng triệt tiêu về 0.
    """
    def __init__(self, lr=0.1, eps=1e-8):
        super().__init__(lr=lr)
        self.eps = eps
        self.G = {}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            if k not in self.G:
                self.G[k] = np.zeros_like(params[k])
            
            self.G[k] += grads[k] ** 2
            step_size = self.lr / (np.sqrt(self.G[k]) + self.eps)
            params[k] -= step_size * grads[k]


class RMSpropOptimizer(BaseOptimizer):
    """
    3. RMSprop (Tieleman & Hinton, Coursera 2012):
    - Thay tổng dồn bằng Trung bình động hàm mũ (EMA):
      v_t = beta * v_{t-1} + (1 - beta) * g_t^2
    - Cập nhật: theta_t = theta_{t-1} - (lr / (sqrt(v_t) + eps)) * g_t
    - Điểm yếu: Thiếu mô-men bậc 1 (quán tính) và thiếu hiệu chỉnh độ chệch ban đầu.
    """
    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        super().__init__(lr=lr)
        self.beta = beta
        self.eps = eps
        self.v = {}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            if k not in self.v:
                self.v[k] = np.zeros_like(params[k])
            
            self.v[k] = self.beta * self.v[k] + (1.0 - self.beta) * (grads[k] ** 2)
            step_size = self.lr / (np.sqrt(self.v[k]) + self.eps)
            params[k] -= step_size * grads[k]


class AdamOptimizer(BaseOptimizer):
    """
    4. Adam (Kingma & Ba, ICLR 2015):
    - Mô-men bậc 1: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    - Mô-men bậc 2: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    - Hiệu chỉnh độ chệch (Bias Correction):
      m_hat_t = m_t / (1 - beta1^t)
      v_hat_t = v_t / (1 - beta2^t)
    - Cập nhật: theta_t = theta_{t-1} - (lr / (sqrt(v_hat_t) + eps)) * m_hat_t
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True):
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.bias_correction = bias_correction
        self.m = {}
        self.v = {}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])

            # Cập nhật mô-men bậc 1 (quán tính hướng đi)
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * grads[k]
            # Cập nhật mô-men bậc 2 (thống kê phương sai năng lượng)
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (grads[k] ** 2)

            if self.bias_correction:
                # Hiệu chỉnh độ chệch khởi tạo giải tích
                m_hat = self.m[k] / (1.0 - (self.beta1 ** self.t))
                v_hat = self.v[k] / (1.0 - (self.beta2 ** self.t))
            else:
                m_hat = self.m[k]
                v_hat = self.v[k]

            # Bước cập nhật tham số
            step_size = self.lr / (np.sqrt(v_hat) + self.eps)
            params[k] -= step_size * m_hat


class AdamWOptimizer(BaseOptimizer):
    """
    5. AdamW (Loshchilov & Hutter, ICLR 2019) - Kế thừa hiện đại:
    - Khắc phục lỗi của Adam khi tích hợp L2 Weight Decay.
    - Phân rã trọng số trực tiếp (Decoupled): theta_t = (1 - lr * weight_decay) * theta_{t-1} - Adam_Step
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])

            # 1. Phân rã trọng số trực tiếp độc lập với gradient
            if self.weight_decay > 0.0:
                params[k] -= self.lr * self.weight_decay * params[k]

            # 2. Tính toán Adam thông thường
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (grads[k] ** 2)

            m_hat = self.m[k] / (1.0 - (self.beta1 ** self.t))
            v_hat = self.v[k] / (1.0 - (self.beta2 ** self.t))

            params[k] -= (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat


# ==============================================================================
# PHẦN 2: THỰC NGHIỆM 1 - ĐỊA HÌNH HẺM NÚI DỐC LỆCH (ILL-CONDITIONED RAVINE)
# ==============================================================================

def ravine_loss_and_grad(x, y, curvature_ratio=50.0):
    """
    Hàm mất mát mô phỏng hẻm núi dốc lệch (Ill-conditioned quadratic ravine):
    f(x, y) = 0.5 * x^2 + 0.5 * curvature_ratio * y^2
    - Chiều x có độ dốc thoai thoải (đáy thung lũng, curvature = 1.0)
    - Chiều y có độ dốc cực kỳ dốc đứng (vách núi, curvature = 50.0)
    - Cực tiểu toàn cục tại (0, 0) với loss = 0.
    """
    loss = 0.5 * (x ** 2) + 0.5 * curvature_ratio * (y ** 2)
    grad_x = x
    grad_y = curvature_ratio * y
    return loss, {"x": grad_x, "y": grad_y}


def run_ravine_experiment():
    """
    Khảo sát quỹ đạo tối ưu của 5 thuật toán trong hẻm núi dốc lệch:
    - Điểm xuất phát: (x0, y0) = (-4.0, 1.0)
    - Số bước tối đa: 200 bước
    """
    print("\n" + "=" * 80)
    print("THỰC NGHIỆM 1: ĐỐI CHUẨN TỐI ƯU HÓA TRÊN ĐỊA HÌNH HẺM NÚI DỐC LỆCH (RAVINE)")
    print("=" * 80)
    print("Địa hình hàm mất mát: f(x, y) = 0.5 * x^2 + 25.0 * y^2 (Tỷ số độ dốc 50:1)")
    print("Xuất phát tại (-4.0, 1.0) -> Đích đến tối ưu: (0.0, 0.0)")

    optimizers = {
        "1. Vanilla SGD": SGDOptimizer(lr=0.035, momentum=0.0),
        "2. SGD + Momentum": SGDOptimizer(lr=0.02, momentum=0.85),
        "3. AdaGrad": AdaGradOptimizer(lr=1.2),
        "4. RMSprop": RMSpropOptimizer(lr=0.08, beta=0.9),
        "5. Adam (2014)": AdamOptimizer(lr=0.15, beta1=0.9, beta2=0.999),
    }

    results = {}
    max_steps = 200

    for name, opt in optimizers.items():
        # Khởi tạo tham số
        params = {"x": np.array(-4.0), "y": np.array(1.0)}
        path = [(params["x"].copy(), params["y"].copy())]
        losses = []

        total_distance = 0.0
        steps_to_converge = None

        for step in range(max_steps):
            loss, grads = ravine_loss_and_grad(params["x"], params["y"], curvature_ratio=50.0)
            losses.append(loss)

            if loss < 1e-3 and steps_to_converge is None:
                steps_to_converge = step

            prev_pt = np.array([params["x"], params["y"]])
            opt.step(params, grads)
            curr_pt = np.array([params["x"], params["y"]])
            total_distance += np.linalg.norm(curr_pt - prev_pt)
            path.append((params["x"].copy(), params["y"].copy()))

        final_loss = losses[-1]
        results[name] = {
            "final_loss": final_loss,
            "final_pos": (float(params["x"]), float(params["y"])),
            "steps_to_converge": steps_to_converge if steps_to_converge is not None else ">200",
            "total_path_length": total_distance,
        }

    print(f"\n{'Thuật Toán Tối Ưu':<22} | {'Loss Cuối Cùng':<16} | {'Vị Trí (x, y)':<22} | {'Số Bước Đến Đích':<18} | {'Chiều Dài Quỹ Đạo'}")
    print("-" * 95)
    for name, res in results.items():
        pos_str = f"({res['final_pos'][0]:5.2f}, {res['final_pos'][1]:5.2f})"
        loss_str = f"{res['final_loss']:.2e}"
        steps_str = str(res['steps_to_converge'])
        path_str = f"{res['total_path_length']:6.2f}"
        print(f"{name:<22} | {loss_str:<16} | {pos_str:<22} | {steps_str:<18} | {path_str}")

    print("\n[!] Phân tích chuyên sâu cho người mới bắt đầu:")
    print("    - Vanilla SGD: Dao động zíc-zắc mạnh theo vách dốc y, tiến rất chậm theo đáy x.")
    print("    - SGD + Momentum: Tăng tốc được theo trục x nhưng bị văng quán tính rộng.")
    print("    - AdaGrad: Mẫu số dồn tích tăng nhanh, bị tắt dần tốc độ học khi chưa tới tâm.")
    print("    - RMSprop: Cân bằng được tốc độ học của 2 trục nhưng dao động nhẹ quanh đáy.")
    print("    - Adam: Kết hợp cả thích ứng tọa độ lẫn đà quán tính -> Quỹ đạo ngắn nhất, hội tụ mượt nhất!")


# ==============================================================================
# PHẦN 3: THỰC NGHIỆM 2 - HUẤN LUYỆN MẠNG NƠ-RON TRÊN DỮ LIỆU CÓ ĐỘ THƯA
# ==============================================================================

class SimpleMLP:
    """Mô hình Mạng Nơ-ron 2 Tầng ẩn đơn giản dùng để đối chuẩn optimizer."""
    def __init__(self, in_features=20, hidden_dim=32, out_features=2, seed=42):
        np.random.seed(seed)
        self.params = {
            "W1": np.random.randn(in_features, hidden_dim) * np.sqrt(2.0 / in_features),
            "b1": np.zeros((1, hidden_dim)),
            "W2": np.random.randn(hidden_dim, out_features) * np.sqrt(2.0 / hidden_dim),
            "b2": np.zeros((1, out_features)),
        }
        self.cache = {}

    def forward(self, X):
        self.cache["X"] = X
        z1 = np.dot(X, self.params["W1"]) + self.params["b1"]
        a1 = np.maximum(0.0, z1) # ReLU
        self.cache["z1"] = z1
        self.cache["a1"] = a1

        logits = np.dot(a1, self.params["W2"]) + self.params["b2"]
        exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-12)
        self.cache["probs"] = probs
        return probs

    def compute_loss_and_grads(self, y_true):
        probs = self.cache["probs"]
        X = self.cache["X"]
        a1 = self.cache["a1"]
        z1 = self.cache["z1"]
        N = X.shape[0]

        # Cross-Entropy Loss
        correct_logprobs = -np.log(probs[np.arange(N), y_true] + 1e-12)
        loss = np.mean(correct_logprobs)

        # Gradients
        dlogits = probs.copy()
        dlogits[np.arange(N), y_true] -= 1.0
        dlogits /= N

        dW2 = np.dot(a1.T, dlogits)
        db2 = np.sum(dlogits, axis=0, keepdims=True)

        da1 = np.dot(dlogits, self.params["W2"].T)
        dz1 = da1 * (z1 > 0).astype(np.float32)

        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return loss, grads


def run_mlp_training_benchmark():
    """
    Huấn luyện Mạng Nơ-ron trên tập dữ liệu nhị phân với đặc trưng thưa (Sparse features):
    So sánh tốc độ suy giảm hàm mất mát qua 100 epochs giữa 5 thuật toán.
    """
    print("\n" + "=" * 80)
    print("THỰC NGHIỆM 2: ĐỐI CHUẨN TỐI ƯU HÓA KHI HUẤN LUYỆN MẠNG NƠ-RON (MLP)")
    print("=" * 80)

    np.random.seed(1337)
    N, D = 300, 20
    X = np.random.randn(N, D)
    # Giả lập đặc trưng thưa (Sparse features - 70% giá trị bằng 0)
    X[np.abs(X) < 1.0] = 0.0
    y = ((X[:, 0] * 2.0 - X[:, 1] * 1.5 + X[:, 2] * 0.5) > 0).astype(np.int64)

    epochs = 100
    learning_rate = 0.02

    optimizers = {
        "Vanilla SGD": SGDOptimizer(lr=learning_rate),
        "SGD + Momentum": SGDOptimizer(lr=learning_rate, momentum=0.9),
        "AdaGrad": AdaGradOptimizer(lr=0.08),
        "RMSprop": RMSpropOptimizer(lr=learning_rate, beta=0.9),
        "Adam (2014)": AdamOptimizer(lr=learning_rate, beta1=0.9, beta2=0.999),
        "AdamW (2017)": AdamWOptimizer(lr=learning_rate, weight_decay=0.01),
    }

    results = {}

    for name, opt in optimizers.items():
        mlp = SimpleMLP(in_features=D, hidden_dim=32, out_features=2, seed=42)
        loss_history = []

        start_time = time.time()
        for ep in range(epochs):
            probs = mlp.forward(X)
            loss, grads = mlp.compute_loss_and_grads(y)
            opt.step(mlp.params, grads)
            loss_history.append(loss)

        elapsed = time.time() - start_time
        final_preds = np.argmax(mlp.forward(X), axis=1)
        final_acc = np.mean(final_preds == y) * 100.0

        results[name] = {
            "initial_loss": loss_history[0],
            "loss_ep10": loss_history[9],
            "loss_ep50": loss_history[49],
            "final_loss": loss_history[-1],
            "final_acc": final_acc,
            "elapsed_ms": elapsed * 1000.0,
        }

    print(f"\n{'Thuật Toán Tối Ưu':<20} | {'Loss Ep 0':<10} | {'Loss Ep 10':<12} | {'Loss Ep 50':<12} | {'Loss Cuối':<10} | {'Train Acc':<10} | {'Thời Gian'}")
    print("-" * 95)
    for name, res in results.items():
        print(f"{name:<20} | {res['initial_loss']:8.4f}  | {res['loss_ep10']:8.4f}    | {res['loss_ep50']:8.4f}    | {res['final_loss']:8.4f}  | {res['final_acc']:6.1f}%    | {res['elapsed_ms']:6.1f} ms")

    print("\n[+] BẢNG SO SÁNH GIÁ TRỊ HỌC THUẬT & BƯỚC CHUYỂN DỊCH HỆ TƯ TƯỞNG:")
    print("┌──────────────────────────┬─────────────────────────────────────┬─────────────────────────────────────┐")
    print("│ Tiêu Chí So Sánh         │ Kỷ Nguyên Tiền-Adam (SGD, RMSprop)  │ Đột Phá Adam (Kingma & Ba, 2014)    │")
    print("├──────────────────────────┼─────────────────────────────────────┼─────────────────────────────────────┤")
    print("│ Ước lượng quán tính      │ Riêng lẻ: Hoặc chỉ có m_t, hoặc v_t │ Kết hợp cả Mô-men bậc 1 & bậc 2     │")
    print("│ Hiệu chỉnh độ chệch      │ Không có (Bị ảnh hưởng bởi zero-init)│ Phép chia giải tích 1 - beta^t      │")
    print("│ Độ nhạy siêu tham số     │ Cực kỳ nhạy với Learning Rate cơ sở │ Rất bền bỉ (Robust), lr=1e-3 chuẩn  │")
    print("│ Địa hình dốc lệch        │ Dễ dao động, trượt khỏi đáy thung lũng│ Tự động cân bằng bước nhảy từng trục│")
    print("│ Bước nhảy hiệu dụng      │ Phụ thuộc độ lớn thô của gradient   │ Bị chặn xấp xỉ bởi alpha * SNR      │")
    print("│ Di sản công nghiệp       │ Đòi hỏi tinh chỉnh lịch trình giảm lr│ Bộ tối ưu hóa mặc định của Deep AI  │")
    print("└──────────────────────────┴─────────────────────────────────────┴─────────────────────────────────────┘")


# ==============================================================================
# HÀM CHÍNH (MAIN ENTRYPOINT)
# ==============================================================================

def main():
    print("================================================================================")
    print("CHƯƠNG TRÌNH MÔ PHỎNG CỐT LÕI BÀI BÁO ADAM (ICLR 2015) - FROM SCRATCH")
    print("================================================================================")

    # 1. Chạy khảo sát hẻm núi dốc lệch
    run_ravine_experiment()

    # 2. Chạy khảo sát huấn luyện Mạng nơ-ron
    run_mlp_training_benchmark()

    print("\n[✓] Hoàn thành trọn vẹn mô phỏng thuật toán và đối chuẩn học thuật của bài báo Adam!")


if __name__ == "__main__":
    main()
