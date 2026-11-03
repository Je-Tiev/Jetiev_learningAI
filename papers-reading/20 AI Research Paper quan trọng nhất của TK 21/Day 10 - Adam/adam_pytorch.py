"""
================================================================================
BÀI BÁO: Adam: A Method for Stochastic Optimization (Kingma & Ba, ICLR 2015)
TỆP MÃ NGUỒN: adam_pytorch.py
MÔ TẢ DÀNH CHO NGƯỜI MỚI BẮT ĐẦU:
    Tệp mã nguồn này triển khai thuật toán Adam và các bộ tối ưu hóa liên quan
    bằng framework PyTorch hiện đại, kết hợp giải mã cơ chế bên dưới C++/CUDA:

    1. Tự Xây Dựng CustomAdam Kế Thừa torch.optim.Optimizer:
       - Giải mã cách PyTorch quản lý param_groups và state dict.
       - Triển khai exp_avg (mô-men 1) và exp_avg_sq (mô-men 2).
       - Tính toán hiệu chỉnh độ chệch (Bias Correction) theo từng bước lặp.

    2. Khảo Sát Quỹ Đạo 2D Trên Địa Hình Dốc Lệch (Ravine Trajectory Tracking):
       - Trực quan hóa đường đi của SGD, Momentum, RMSprop, và Adam trên bề mặt hàm mất mát
         dị hướng (độ cong vách dốc gấp 50 lần đáy thung lũng).

    3. Huấn Luyện Mạng Nơ-ron Sâu & So Sánh Giá Trị Học Thuật:
       - Đối chiếu trực diện tốc độ hội tụ và độ chính xác của 5 bộ tối ưu hóa:
         SGD, SGD + Momentum, RMSprop, Adam, và AdamW.

    4. Phân Tích Chuyên Sâu: Adam vs AdamW (Decoupled Weight Decay):
       - Chứng minh tại sao AdamW đem lại khả năng tổng quát hóa vượt trội khi có điều hòa L2.

    5. Xuất Biểu Đồ Đối Chuẩn 4 Bảng Con (outputs/adam_academic_comparison.png):
       - Bản đồ đồng mức quỹ đạo 2D, đường cong Loss, hệ số Bias Correction, và đối sánh Adam vs AdamW.
================================================================================
"""

import os
import sys
import math
import time
import numpy as np

# ==============================================================================
# KIỂM TRA PHỤ THUỘC PYTORCH & MATPLOTLIB
# ==============================================================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.optimizer import Optimizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ==============================================================================
# PHẦN 1: TỰ XÂY DỰNG CUSTOM ADAM (KẾ THỪA torch.optim.Optimizer)
# ==============================================================================

if HAS_TORCH:
    class CustomAdam(Optimizer):
        """
        Tự cài đặt bộ tối ưu hóa Adam chuẩn theo bài báo ICLR 2015:
        - Quản lý trạng thái nội bộ trong `self.state[p]`:
          + 'step': Bộ đếm bước lặp thời gian t
          + 'exp_avg': Vector mô-men bậc 1 m_t
          + 'exp_avg_sq': Vector mô-men bậc 2 v_t
        - Tối ưu hóa bộ nhớ và tốc độ với các phép toán in-place của PyTorch.
        """
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
            if not 0.0 <= lr:
                raise ValueError(f"Tốc độ học không hợp lệ: {lr}")
            if not 0.0 <= eps:
                raise ValueError(f"Hằng số eps không hợp lệ: {eps}")
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError(f"Hệ số beta1 không hợp lệ: {betas[0]}")
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError(f"Hệ số beta2 không hợp lệ: {betas[1]}")

            defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']

                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad

                    state = self.state[p]

                    # Khởi tạo trạng thái bộ nhớ đệm nếu là bước đầu tiên
                    if len(state) == 0:
                        state['step'] = 0
                        # Vector m_0 = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Vector v_0 = 0
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']

                    state['step'] += 1
                    step_t = state['step']

                    # Phạt L2 truyền thống (cộng vào gradient trước khi tính mô-men)
                    if weight_decay != 0:
                        grad = grad.add(p, alpha=weight_decay)

                    # 1. Cập nhật mô-men bậc 1: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                    # 2. Cập nhật mô-men bậc 2: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    # 3. Tính hệ số hiệu chỉnh độ chệch (Bias Correction)
                    bias_correction1 = 1.0 - (beta1 ** step_t)
                    bias_correction2 = 1.0 - (beta2 ** step_t)

                    # 4. Mẫu số: sqrt(v_hat) + eps
                    step_size = lr / bias_correction1
                    bias_correction2_sqrt = math.sqrt(bias_correction2)
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                    # 5. Cập nhật trọng số: theta_t = theta_{t-1} - step_size * (m_t / denom)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

            return loss


# ==============================================================================
# PHẦN 2: THỰC NGHIỆM 1 - QUỸ ĐẠO TỐI ƯU HÓA 2D TRÊN HẺM NÚI DỐC LỆCH
# ==============================================================================

def track_2d_ravine_trajectories():
    """
    Theo dõi chi tiết quỹ đạo tối ưu 2D trên hàm mất mát dị hướng:
    f(x, y) = 0.5 * x^2 + 25.0 * y^2
    """
    if not HAS_TORCH:
        return {}

    curvature_ratio = 50.0
    start_point = (-4.0, 1.0)
    max_steps = 150

    optimizers_to_test = {
        "SGD (lr=0.035)": lambda p: optim.SGD([p], lr=0.035),
        "Momentum (lr=0.02)": lambda p: optim.SGD([p], lr=0.02, momentum=0.85),
        "RMSprop (lr=0.08)": lambda p: optim.RMSprop([p], lr=0.08, alpha=0.9),
        "Adam (lr=0.15)": lambda p: CustomAdam([p], lr=0.15, betas=(0.9, 0.999)),
    }

    trajectories = {}

    for name, opt_factory in optimizers_to_test.items():
        # Khởi tạo tham số độc lập
        p = torch.tensor(list(start_point), dtype=torch.float32, requires_grad=True)
        opt = opt_factory(p)

        traj = [p.detach().clone().numpy()]
        losses = []

        for _ in range(max_steps):
            opt.zero_grad()
            # f(x, y) = 0.5 * x^2 + 0.5 * 50 * y^2
            loss = 0.5 * (p[0] ** 2) + 0.5 * curvature_ratio * (p[1] ** 2)
            loss.backward()
            opt.step()

            traj.append(p.detach().clone().numpy())
            losses.append(loss.item())

        trajectories[name] = {
            "path": np.array(traj),
            "final_loss": losses[-1],
            "losses": losses
        }

    return trajectories


# ==============================================================================
# PHẦN 3: THỰC NGHIỆM 2 - HUẤN LUYỆN MẠNG NƠ-RON TRÊN DỮ LIỆU THỬ THÁCH
# ==============================================================================

if HAS_TORCH:
    class DeepBenchmarkMLP(nn.Module):
        """Mạng nơ-ron đa tầng sâu dùng để kiểm chứng tốc độ hội tụ."""
        def __init__(self, in_features=30, hidden_dims=[64, 32], num_classes=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], num_classes)
            )

        def forward(self, x):
            return self.net(x)


def generate_benchmark_dataset(n_samples=500, n_features=30, seed=42):
    """Tạo tập dữ liệu phân loại nhị phân tổng hợp có tính phi tuyến cao."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Giả lập đặc trưng thưa (40% giá trị bằng 0)
    X[np.abs(X) < 0.5] = 0.0
    # Ranh giới quyết định phi tuyến
    signal = 1.2 * X[:, 0] - 1.8 * (X[:, 1] ** 2) + 1.0 * np.sin(X[:, 2] * 2.0)
    y = (signal > 0).astype(np.int64)

    # Chia Train/Test (80/20)
    n_train = int(n_samples * 0.8)
    return (X[:n_train], y[:n_train]), (X[n_train:], y[n_train:])


def run_neural_net_benchmark():
    """
    Huấn luyện mạng nơ-ron DeepBenchmarkMLP trong 60 epochs
    và so sánh tốc độ suy giảm Loss giữa các bộ tối ưu hóa.
    """
    if not HAS_TORCH:
        print("[!] Không có PyTorch.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (X_tr, y_tr), (X_te, y_te) = generate_benchmark_dataset()

    X_train_t = torch.tensor(X_tr).to(device)
    y_train_t = torch.tensor(y_tr).to(device)
    X_test_t = torch.tensor(X_te).to(device)
    y_test_t = torch.tensor(y_te).to(device)

    epochs = 60
    base_lr = 0.01

    optimizers_dict = {
        "1. Vanilla SGD": lambda m: optim.SGD(m.parameters(), lr=base_lr),
        "2. SGD + Momentum": lambda m: optim.SGD(m.parameters(), lr=base_lr, momentum=0.9),
        "3. RMSprop": lambda m: optim.RMSprop(m.parameters(), lr=base_lr * 0.5, alpha=0.9),
        "4. Custom Adam (2014)": lambda m: CustomAdam(m.parameters(), lr=base_lr * 0.5),
        "5. AdamW (2017)": lambda m: optim.AdamW(m.parameters(), lr=base_lr * 0.5, weight_decay=0.01),
    }

    criterion = nn.CrossEntropyLoss()
    history = {k: {"train_loss": [], "test_acc": []} for k in optimizers_dict}

    print("\n" + "=" * 80)
    print("THỰC NGHIỆM HUẤN LUYỆN MẠNG NƠ-RON BẰNG PYTORCH TRÊN 5 BỘ TỐI ƯU HÓA")
    print("=" * 80)

    for name, opt_fn in optimizers_dict.items():
        torch.manual_seed(100) # Cố định trọng số ban đầu giống nhau 100%
        model = DeepBenchmarkMLP(in_features=30, hidden_dims=[64, 32], num_classes=2).to(device)
        optimizer = opt_fn(model)

        t_start = time.time()
        for ep in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train_t)
            loss = criterion(logits, y_train_t)
            loss.backward()
            optimizer.step()

            # Đánh giá trên tập test
            model.eval()
            with torch.no_grad():
                test_preds = torch.argmax(model(X_test_t), dim=1)
                test_acc = (test_preds == y_test_t).float().mean().item() * 100.0

            history[name]["train_loss"].append(loss.item())
            history[name]["test_acc"].append(test_acc)

        elapsed = time.time() - t_start
        final_loss = history[name]["train_loss"][-1]
        final_acc = history[name]["test_acc"][-1]
        print(f"[*] {name:<25} | Loss cuối: {final_loss:6.4f} | Test Acc: {final_acc:5.1f}% | Thời gian: {elapsed*1000:5.1f}ms")

    return history


# ==============================================================================
# PHẦN 4: XUẤT HÌNH ẢNH TRỰC QUAN HÓA ĐỐI CHUẨN (OUTPUTS)
# ==============================================================================

def plot_academic_benchmark_figure(trajectories, history, output_dir="outputs"):
    """
    Xuất đồ thị 4 bảng con phản ánh đầy đủ giá trị học thuật của bài báo:
    - Bảng 1: Quỹ đạo 2D trên Contour Map của Ravine dốc lệch.
    - Bảng 2: Đường cong suy giảm Loss khi huấn luyện Mạng Nơ-ron.
    - Bảng 3: Diễn tiến hệ số Hiệu chỉnh Độ chệch (Bias Correction Factor) theo bước lặp.
    - Bảng 4: So sánh độ chính xác Test giữa các thuật toán tối ưu.
    """
    if not HAS_MATPLOTLIB:
        print("[!] Không tìm thấy matplotlib. Bỏ qua xuất biểu đồ.")
        return

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION (Kingma & Ba, ICLR 2015)\nAcademic Value Benchmark & Optimization Dynamics",
                 fontsize=14, fontweight="bold", y=0.98)

    # --------------------------------------------------------------------------
    # Bảng 1: Quỹ đạo 2D trên địa hình dốc lệch
    # --------------------------------------------------------------------------
    ax1 = axes[0, 0]
    X_grid, Y_grid = np.meshgrid(np.linspace(-4.5, 1.0, 200), np.linspace(-1.5, 1.5, 200))
    Z_grid = 0.5 * (X_grid ** 2) + 0.5 * 50.0 * (Y_grid ** 2)
    contours = ax1.contour(X_grid, Y_grid, Z_grid, levels=np.logspace(-1, 3, 20), cmap="Blues", alpha=0.6)
    
    colors = {"SGD (lr=0.035)": "red", "Momentum (lr=0.02)": "green", "RMSprop (lr=0.08)": "orange", "Adam (lr=0.15)": "blue"}
    for name, data in trajectories.items():
        path = data["path"]
        ax1.plot(path[:, 0], path[:, 1], '.-', color=colors.get(name, "black"), label=name, markersize=3, linewidth=1.5, alpha=0.85)

    ax1.plot(0, 0, 'r*', markersize=14, label="Global Minimum (0, 0)")
    ax1.set_title("1. 2D Trajectories on Ill-Conditioned Ravine (50:1)", fontweight="bold")
    ax1.set_xlabel("x (Slow valley direction)")
    ax1.set_ylabel("y (Steep cliff direction)")
    ax1.set_xlim(-4.5, 1.0)
    ax1.set_ylim(-1.5, 1.5)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # --------------------------------------------------------------------------
    # Bảng 2: Đường cong suy giảm Loss khi huấn luyện Mạng Nơ-ron
    # --------------------------------------------------------------------------
    ax2 = axes[0, 1]
    epochs = range(1, len(next(iter(history.values()))["train_loss"]) + 1)
    plot_colors = ["#e74c3c", "#27ae60", "#f39c12", "#2980b9", "#8e44ad"]
    for idx, (name, data) in enumerate(history.items()):
        ax2.plot(epochs, data["train_loss"], label=name, color=plot_colors[idx % len(plot_colors)], linewidth=2)

    ax2.set_title("2. Neural Net Training Loss Convergence", fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8)

    # --------------------------------------------------------------------------
    # Bảng 3: Diễn tiến hệ số Hiệu chỉnh Độ chệch (Bias Correction)
    # --------------------------------------------------------------------------
    ax3 = axes[1, 0]
    steps = np.arange(1, 301)
    beta1 = 0.9
    beta2 = 0.999
    bias1 = 1.0 - (beta1 ** steps)
    bias2 = 1.0 - (beta2 ** steps)
    inv_bias1 = 1.0 / bias1
    inv_bias2_sqrt = 1.0 / np.sqrt(bias2)

    ax3.plot(steps, inv_bias1, 'b-', label=r"Moment 1 Scale: $1 / (1 - \beta_1^t)$", linewidth=2)
    ax3.plot(steps, inv_bias2_sqrt, 'r-', label=r"Moment 2 Scale: $1 / \sqrt{1 - \beta_2^t}$", linewidth=2)
    ax3.set_title(r"3. Analytical Bias Correction Factors ($t = 1 \to 300$)", fontweight="bold")
    ax3.set_xlabel("Optimization Step (t)")
    ax3.set_ylabel("Magnification Multiplier")
    ax3.set_ylim(1.0, 12.0)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=9)

    # --------------------------------------------------------------------------
    # Bảng 4: So sánh Test Accuracy giữa các bộ tối ưu
    # --------------------------------------------------------------------------
    ax4 = axes[1, 1]
    names_short = [k.split(". ")[-1] for k in history.keys()]
    final_accs = [history[k]["test_acc"][-1] for k in history.keys()]
    bars = ax4.bar(names_short, final_accs, color=plot_colors[:len(names_short)], alpha=0.85, width=0.5)
    ax4.set_title("4. Generalization Test Accuracy Comparison (%)", fontweight="bold")
    ax4.set_ylabel("Test Accuracy (%)")
    ax4.set_ylim(50, 100)
    ax4.grid(axis='y', linestyle="--", alpha=0.5)

    for bar, acc in zip(bars, final_accs):
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{acc:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    out_file = os.path.join(output_dir, "adam_academic_comparison.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"\n[✓] Đã xuất thành công biểu đồ đối chuẩn học thuật tại: {out_file}")


# ==============================================================================
# HÀM CHÍNH (MAIN PIPELINE)
# ==============================================================================

def main():
    print("=" * 80)
    print("CHƯƠNG TRÌNH ĐỐI CHUẨN HỌC THUẬT: BỘ TỐI ƯU HÓA ADAM TRONG PYTORCH")
    print("=" * 80)

    if HAS_TORCH:
        # 1. Chạy khảo sát quỹ đạo 2D
        print("\n[*] [1/3] Đang khảo sát quỹ đạo 2D trên địa hình dốc lệch (Ravine)...")
        trajectories = track_2d_ravine_trajectories()

        # 2. Chạy khảo sát huấn luyện Mạng Nơ-ron
        print("\n[*] [2/3] Đang huấn luyện Mạng Nơ-ron trên 5 bộ tối ưu hóa...")
        history = run_neural_net_benchmark()

        # 3. Xuất biểu đồ
        print("\n[*] [3/3] Đang xuất báo cáo trực quan hóa...")
        if history is not None and trajectories:
            plot_academic_benchmark_figure(trajectories, history)
    else:
        print("[!] Không tìm thấy PyTorch trong môi trường hiện tại.")

    print("\n[✓] Kết thúc chương trình adam_pytorch.py thành công!")


if __name__ == "__main__":
    main()
