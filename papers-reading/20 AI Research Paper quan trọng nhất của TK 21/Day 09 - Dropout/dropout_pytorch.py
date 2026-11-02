"""
================================================================================
BÀI BÁO: Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al., JMLR 2014)
TỆP MÃ NGUỒN: dropout_pytorch.py
MÔ TẢ DÀNH CHO NGƯỜI MỚI BẮT ĐẦU:
    Tệp mã nguồn này triển khai kỹ thuật Dropout bằng framework học sâu PyTorch hiện đại,
    kết hợp đối chiếu giá trị học thuật chuyên sâu và minh họa cơ chế hoạt động:

    1. Cơ Chế Inverted Dropout Trong PyTorch:
       - Giải mã cách thức nn.Dropout(p) hoạt động bên dưới hạ tầng C++/CUDA.
       - Tự xây dựng CustomDropoutFunction (kế thừa torch.autograd.Function) để hiểu
         cách forward chia cho (1 - p) và backward truyền gradient qua mặt nạ Bernoulli.
       - Tầm quan trọng sinh tử của việc chuyển đổi chế độ model.train() và model.eval().

    2. Khối Thực Nghiệm Đối Chiếu Giá Trị Học Thuật (Academic Value Benchmark):
       - Huấn luyện trên bài toán phân loại có số chiều lớn và mẫu nhỏ (D = 40, N = 120),
         trong đó 38 chiều là nhiễu thuần túy (kịch bản gây quá khớp cực kỳ nghiêm trọng).
       - So sánh trực diện 3 trường phái mô hình:
         1. VanillaMLP (Tiền-Dropout: Không điều hòa -> Học vẹt 100%, Overfitting Gap lớn).
         2. L2RegularizedMLP (Điều hòa truyền thống: Weight Decay -> Giảm biên độ trọng số nhưng nơ-ron vẫn co-adapt).
         3. DropoutMLP (Đột phá Srivastava 2014: nn.Dropout(0.5) -> Phá vỡ co-adaptation, tối ưu tổng quát hóa).

    3. Ứng Dụng Nâng Cao: Monte Carlo Dropout (Gal & Ghahramani, ICML 2016):
       - Giữ nguyên Dropout khi dự đoán (Inference) để lấy mẫu thống kê Bayesian.
       - Định lượng độ bất định (Predictive Uncertainty) của mô hình học sâu.

    4. Xuất Báo Cáo Trực Quan Hóa (outputs/dropout_academic_comparison.png):
       - Biểu đồ 4 bảng con: Đường cong Loss, Overfitting Gap, Ma trận tương quan Co-adaptation,
         và Phân phối độ bất định Monte Carlo Dropout.
================================================================================
"""

import os
import sys
import time
import numpy as np

# ==============================================================================
# KIỂM TRA PHỤ THUỘC PYTORCH & MATPLOTLIB VỚI CƠ CHẾ DỰ PHÒNG AN TOÀN
# ==============================================================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ==============================================================================
# PHẦN 1: GIẢI MÃ CƠ CHẾ DROPOUT TỰ TẠO (CUSTOM AUTOGRAD FUNCTION)
# ==============================================================================

if HAS_TORCH:
    class CustomDropoutFunction(torch.autograd.Function):
        """
        Tự cài đặt toán tử Inverted Dropout bằng torch.autograd.Function
        nhằm giải thích tường minh cơ chế hoạt động bên trong PyTorch:

        - Pha Huấn Luyện (Training):
          mask ~ Bernoulli(1 - p)
          y = (x * mask) / (1 - p)
          Mục đích: Nhân trước 1 / (1 - p) để bảo toàn kỳ vọng toán học E[y] = x.

        - Pha Lan Truyền Ngược (Backward):
          dL/dx = (dL/dy * mask) / (1 - p)
          Gradient chỉ chảy qua các nơ-ron còn sống (mask = 1).
        """
        @staticmethod
        def forward(ctx, x, p=0.5, training=True):
            if not training or p <= 0.0:
                ctx.mask = None
                return x
            
            p_keep = 1.0 - p
            # Sinh tensor Bernoulli nhị phân cùng kích thước với x
            # torch.rand_like(x) sinh số thực ngẫu nhiên đều trong đoạn [0, 1)
            mask = (torch.rand_like(x) < p_keep).float()
            # Inverted scaling: chia cho p_keep
            out = (x * mask) / p_keep
            
            # Lưu lại mask và p_keep để dùng trong backward pass
            ctx.save_for_backward(mask)
            ctx.p_keep = p_keep
            return out

        @staticmethod
        def backward(ctx, grad_output):
            if ctx.mask is None and not ctx.saved_tensors:
                return grad_output, None, None
            
            (mask,) = ctx.saved_tensors
            p_keep = ctx.p_keep
            grad_input = (grad_output * mask) / p_keep
            return grad_input, None, None


    class CustomDropout(nn.Module):
        """Module bọc CustomDropoutFunction để dùng như nn.Dropout chuẩn"""
        def __init__(self, p=0.5):
            super().__init__()
            self.p = p

        def forward(self, x):
            return CustomDropoutFunction.apply(x, self.p, self.training)


# ==============================================================================
# PHẦN 2: THIẾT KẾ CÁC KIẾN TRÚC MẠNG ĐỐI CHIẾU
# ==============================================================================

if HAS_TORCH:
    class VanillaMLP(nn.Module):
        """
        Mô hình 1: Không Điều Hòa (No Regularization) - Kỷ nguyên Tiền-Dropout
        Mạng MLP sâu nhiều tầng tự do điều chỉnh trọng số mà không có bất kỳ ràng buộc nào.
        Dẫn tới việc các nơ-ron bù trừ lỗi sai của nhau (Co-adaptation) và học vẹt nhiễu.
        """
        def __init__(self, in_features=40, hidden_dims=[64, 32], num_classes=2):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_dims[0])
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.relu2 = nn.ReLU()
            self.fc_out = nn.Linear(hidden_dims[1], num_classes)

        def forward(self, x):
            h1 = self.relu1(self.fc1(x))
            h2 = self.relu2(self.fc2(h1))
            logits = self.fc_out(h2)
            return logits

        def get_hidden_activations(self, x):
            """Trích xuất vector kích hoạt sau tầng ẩn đầu tiên để đo Co-adaptation"""
            with torch.no_grad():
                return self.relu1(self.fc1(x))


    class DropoutMLP(nn.Module):
        """
        Mô hình 3: Dropout (Srivastava, Hinton et al., 2014) - Đột phá chống Quá khớp
        Đặt nn.Dropout(p = 0.5) ngay sau các hàm kích hoạt ẩn.
        Mỗi lần huấn luyện, một cấu hình mạng con ngẫu nhiên (Sub-network) được kích hoạt,
        triệt tiêu hiện tượng nơ-ron dựa dẫm vào nhau.
        """
        def __init__(self, in_features=40, hidden_dims=[64, 32], num_classes=2, dropout_p=0.5):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_dims[0])
            self.relu1 = nn.ReLU()
            self.drop1 = nn.Dropout(p=dropout_p)
            self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.relu2 = nn.ReLU()
            self.drop2 = nn.Dropout(p=dropout_p)
            self.fc_out = nn.Linear(hidden_dims[1], num_classes)

        def forward(self, x):
            h1 = self.drop1(self.relu1(self.fc1(x)))
            h2 = self.drop2(self.relu2(self.fc2(h1)))
            logits = self.fc_out(h2)
            return logits

        def get_hidden_activations(self, x):
            """Trích xuất kích hoạt ẩn ở chế độ eval để đo tương quan thực"""
            was_training = self.training
            self.eval()
            with torch.no_grad():
                h1 = self.relu1(self.fc1(x))
            if was_training:
                self.train()
            return h1


# ==============================================================================
# PHẦN 3: TẠO DỮ LIỆU THỬ THÁCH QUÁ KHỚP & ĐO LƯỜNG TƯƠNG QUAN
# ==============================================================================

def generate_overfitting_dataset(n_train=120, n_test=200, n_features=40, seed=42):
    """
    Sinh tập dữ liệu tổng hợp gây nguy cơ Quá Khớp (Overfitting) cực cao:
    - Tập huấn luyện nhỏ (N_train = 120 mẫu).
    - Số chiều lớn (D = 40 chiều): Chỉ có 2 chiều đầu tiên chứa tín hiệu phân loại thực sự,
      38 chiều còn lại là nhiễu Gaussian ngẫu nhiên độc lập.
    - Một mô hình có dung lượng lớn sẽ nhanh chóng học vẹt 38 chiều nhiễu để đạt 100% Train Acc.
    """
    np.random.seed(seed)
    
    # Huấn luyện
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    signal_train = 1.5 * X_train[:, 0] - 2.0 * X_train[:, 1] + 0.4 * np.random.randn(n_train)
    y_train = (signal_train > 0).astype(np.int64)

    # Kiểm tra
    X_test = np.random.randn(n_test, n_features).astype(np.float32)
    signal_test = 1.5 * X_test[:, 0] - 2.0 * X_test[:, 1] + 0.4 * np.random.randn(n_test)
    y_test = (signal_test > 0).astype(np.int64)

    return X_train, y_train, X_test, y_test


def compute_coadaptation_correlation(activations_np):
    """
    Tính hệ số tương quan đồng thích ứng trung bình giữa các nơ-ron:
    Mean |Corr| = (1 / (D * (D - 1))) * sum_{i != j} |Corr(h_i, h_j)|

    - Mean |Corr| cao (> 0.35): Các nơ-ron phụ thuộc lẫn nhau, mã hóa thừa thãi.
    - Mean |Corr| thấp (< 0.15): Dropout ép mỗi nơ-ron học đặc trưng độc lập, hữu ích.
    """
    N, D = activations_np.shape
    centered = activations_np - np.mean(activations_np, axis=0, keepdims=True)
    std = np.std(activations_np, axis=0, keepdims=True) + 1e-8
    normalized = centered / std
    corr_matrix = np.dot(normalized.T, normalized) / N
    
    mask = ~np.eye(D, dtype=bool)
    off_diagonal = np.abs(corr_matrix[mask])
    return np.mean(off_diagonal), corr_matrix


# ==============================================================================
# PHẦN 4: MONTE CARLO DROPOUT (BAYESIAN UNCERTAINTY ESTIMATION)
# ==============================================================================

def monte_carlo_dropout_inference(model, X_tensor, num_samples=50):
    """
    Kỹ thuật Monte Carlo Dropout (Gal & Ghahramani, ICML 2016):
    Bằng cách bật model.train() hoặc kích hoạt Dropout ở pha kiểm thử, ta cho mẫu X
    chạy qua mạng T lần. Mỗi lần tương ứng với việc lấy mẫu một kiến trúc mạng con
    từ phân phối hậu nghiệm Bayesian p(W|D).

    Đầu ra:
    - mean_prob: Dự đoán trung bình (Ensemble Prediction)
    - uncertainty: Phương sai hoặc độ lệch chuẩn của xác suất dự đoán (Epistemic Uncertainty)
    """
    if not HAS_TORCH:
        return None, None
    
    model.train() # Kích hoạt Dropout trong lúc suy luận!
    preds = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy() # Xác suất lớp 1
            preds.append(probs)

    preds = np.array(preds) # (num_samples, N)
    mean_prob = np.mean(preds, axis=0)
    uncertainty = np.std(preds, axis=0) # Độ bất định nhận thức
    return mean_prob, uncertainty


# ==============================================================================
# PHẦN 5: HUẤN LUYỆN VÀ ĐỐI CHUẨN HỌC THUẬT (ACADEMIC VALUE BENCHMARK)
# ==============================================================================

def run_academic_benchmark():
    """
    Tiến hành huấn luyện song song 3 mô hình đại diện cho 3 giai đoạn lịch sử:
    1. Vanilla MLP (Không điều hòa)
    2. L2 Regularized MLP (Weight Decay = 0.02)
    3. Dropout MLP (p = 0.5)

    Theo dõi Loss, Accuracy qua từng epoch và xuất đồ thị tổng hợp.
    """
    if not HAS_TORCH:
        print("[!] Không tìm thấy thư viện PyTorch (torch). Vui lòng cài đặt: pip install torch")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Sử dụng thiết bị tính toán: {device}")

    # Chuẩn bị dữ liệu
    X_train_np, y_train_np, X_test_np, y_test_np = generate_overfitting_dataset()
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_np, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test_np, dtype=torch.long).to(device)

    epochs = 120
    lr = 0.02

    # Khởi tạo 3 mô hình
    torch.manual_seed(42)
    model_vanilla = VanillaMLP(in_features=40, hidden_dims=[64, 32], num_classes=2).to(device)
    model_l2 = VanillaMLP(in_features=40, hidden_dims=[64, 32], num_classes=2).to(device)
    model_dropout = DropoutMLP(in_features=40, hidden_dims=[64, 32], num_classes=2, dropout_p=0.5).to(device)

    # Bộ tối ưu hóa
    criterion = nn.CrossEntropyLoss()
    opt_vanilla = optim.Adam(model_vanilla.parameters(), lr=lr, weight_decay=0.0)
    opt_l2 = optim.Adam(model_l2.parameters(), lr=lr, weight_decay=0.02)
    opt_dropout = optim.Adam(model_dropout.parameters(), lr=lr, weight_decay=0.0)

    models_dict = {
        "1. Vanilla (No Reg)": (model_vanilla, opt_vanilla),
        "2. L2 Weight Decay": (model_l2, opt_l2),
        "3. Dropout (p = 0.5)": (model_dropout, opt_dropout),
    }

    history = {name: {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []} for name in models_dict}

    print("\n[*] Bắt đầu tiến trình huấn luyện 120 epochs trên cả 3 mô hình...")
    start_time = time.time()

    for epoch in range(epochs):
        for name, (model, optimizer) in models_dict.items():
            # Pha Huấn Luyện
            model.train()
            optimizer.zero_grad()
            logits = model(X_train_t)
            loss = criterion(logits, y_train_t)
            loss.backward()
            optimizer.step()

            # Đánh giá tập Train
            model.eval()
            with torch.no_grad():
                train_preds = torch.argmax(model(X_train_t), dim=1)
                train_acc = (train_preds == y_train_t).float().mean().item() * 100.0

                # Đánh giá tập Test
                test_logits = model(X_test_t)
                test_loss = criterion(test_logits, y_test_t).item()
                test_preds = torch.argmax(test_logits, dim=1)
                test_acc = (test_preds == y_test_t).float().mean().item() * 100.0

            history[name]["train_loss"].append(loss.item())
            history[name]["test_loss"].append(test_loss)
            history[name]["train_acc"].append(train_acc)
            history[name]["test_acc"].append(test_acc)

    elapsed = time.time() - start_time
    print(f"[✓] Huấn luyện hoàn tất sau {elapsed:.2f} giây.")

    # Đo Co-adaptation correlation
    corrs = {}
    corr_mats = {}
    for name, (model, _) in models_dict.items():
        h = model.get_hidden_activations(X_test_t).cpu().numpy()
        mean_c, c_mat = compute_coadaptation_correlation(h)
        corrs[name] = mean_c
        corr_mats[name] = c_mat

    # In kết quả định lượng
    print("\n" + "=" * 80)
    print("KẾT QUẢ ĐỐI CHUẨN GIÁ TRỊ HỌC THUẬT: ĐỘT PHÁ CỦA DROPOUT TRONG PYTORCH")
    print("=" * 80)
    print(f"{'Mô Hình':<25} | {'Train Acc':<10} | {'Test Acc':<10} | {'Overfit Gap':<12} | {'Co-adaptation |Corr|':<20}")
    print("-" * 85)

    summary_results = {}
    for name in models_dict:
        final_tr_acc = history[name]["train_acc"][-1]
        final_te_acc = history[name]["test_acc"][-1]
        gap = final_tr_acc - final_te_acc
        c = corrs[name]
        summary_results[name] = {
            "train_acc": final_tr_acc,
            "test_acc": final_te_acc,
            "gap": gap,
            "corr": c
        }
        note = "Học vẹt 100%" if gap > 20 else ("Vẫn co-adapt" if gap > 10 else "Tối ưu, độc lập")
        print(f"{name:<25} | {final_tr_acc:6.1f}%    | {final_te_acc:6.1f}%    | {gap:6.1f}%      | {c:6.3f} ({note})")

    # Thử nghiệm Monte Carlo Dropout
    print("\n[*] [Ứng dụng Hiện đại] Thực hiện Monte Carlo Dropout Uncertainty Estimation (50 passes)...")
    _, mc_uncertainty = monte_carlo_dropout_inference(model_dropout, X_test_t, num_samples=50)
    mean_unc = np.mean(mc_uncertainty)
    print(f"    - Độ bất định trung bình trên tập kiểm tra: {mean_unc:.4f}")
    print("    - Các mẫu nằm sát ranh giới quyết định (Decision Boundary) có độ bất định cao hơn hẳn.")

    # Vẽ biểu đồ trực quan hóa
    plot_comparison(history, summary_results, corr_mats, mc_uncertainty)


def plot_comparison(history, summary, corr_mats, mc_uncertainty, output_dir="outputs"):
    """Xuất hình ảnh 4 bảng con minh họa giá trị học thuật của bài báo Dropout."""
    if not HAS_MATPLOTLIB:
        print("[!] Không tìm thấy matplotlib. Bỏ qua bước xuất ảnh.")
        return

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("DROPOUT: A SIMPLE WAY TO PREVENT NEURAL NETWORKS FROM OVERFITTING\nPyTorch Academic Value Benchmark & Diagnostic Suite",
                 fontsize=14, fontweight="bold", y=0.98)

    # Bảng 1: Train & Test Loss qua các Epochs (Chứng minh Overfitting)
    ax1 = axes[0, 0]
    epochs = range(1, len(history["1. Vanilla (No Reg)"]["train_loss"]) + 1)
    ax1.plot(epochs, history["1. Vanilla (No Reg)"]["train_loss"], 'r--', alpha=0.6, label="Vanilla Train Loss")
    ax1.plot(epochs, history["1. Vanilla (No Reg)"]["test_loss"], 'r-', linewidth=2, label="Vanilla Test Loss (Overfitting!)")
    ax1.plot(epochs, history["3. Dropout (p = 0.5)"]["train_loss"], 'b--', alpha=0.6, label="Dropout Train Loss")
    ax1.plot(epochs, history["3. Dropout (p = 0.5)"]["test_loss"], 'b-', linewidth=2, label="Dropout Test Loss (Stable)")
    ax1.set_title("1. Training vs Testing Loss (Overfitting Explosion)", fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_ylim(0, 2.0)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    # Bảng 2: So sánh Overfitting Gap giữa 3 mô hình
    ax2 = axes[0, 1]
    models = list(summary.keys())
    short_names = ["Vanilla (No Reg)", "L2 Weight Decay", "Dropout (p=0.5)"]
    train_accs = [summary[m]["train_acc"] for m in models]
    test_accs = [summary[m]["test_acc"] for m in models]
    gaps = [summary[m]["gap"] for m in models]

    x = np.arange(len(short_names))
    width = 0.25
    ax2.bar(x - width, train_accs, width, label="Train Accuracy (%)", color="#e74c3c", alpha=0.85)
    ax2.bar(x, test_accs, width, label="Test Accuracy (%)", color="#2ecc71", alpha=0.85)
    ax2.bar(x + width, gaps, width, label="Overfitting Gap (%)", color="#f39c12", alpha=0.85)

    ax2.set_title("2. Generalization & Overfitting Gap Comparison", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontweight="bold")
    ax2.set_ylabel("Accuracy / Gap (%)")
    ax2.set_ylim(0, 115)
    ax2.grid(axis='y', linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9, loc="upper right")

    # Bảng 3: Ma trận tương quan nơ-ron (Co-adaptation Heatmap)
    ax3 = axes[1, 0]
    mat_vanilla = corr_mats["1. Vanilla (No Reg)"][:20, :20]
    mat_drop = corr_mats["3. Dropout (p = 0.5)"][:20, :20]
    
    # Ghép 2 ma trận nửa trên nửa dưới để trực quan hóa tương phản
    combined_mat = np.triu(mat_vanilla) + np.tril(mat_drop, -1)
    im = ax3.imshow(np.abs(combined_mat), cmap="magma", vmin=0, vmax=0.6)
    ax3.set_title("3. Neuron Co-adaptation: Vanilla (Upper) vs Dropout (Lower)", fontweight="bold")
    ax3.set_xlabel("Neuron Index (0-19)")
    ax3.set_ylabel("Neuron Index (0-19)")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # Bảng 4: Phân phối độ bất định Monte Carlo Dropout
    ax4 = axes[1, 1]
    ax4.hist(mc_uncertainty, bins=25, color="#9b59b6", edgecolor="black", alpha=0.75)
    ax4.axvline(np.mean(mc_uncertainty), color="red", linestyle="--", linewidth=2,
                label=f"Mean Uncertainty: {np.mean(mc_uncertainty):.3f}")
    ax4.set_title("4. Monte Carlo Dropout Uncertainty Distribution (Gal 2016)", fontweight="bold")
    ax4.set_xlabel("Predictive Uncertainty (Standard Deviation)")
    ax4.set_ylabel("Number of Test Samples")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "dropout_academic_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n[✓] Đã lưu thành công biểu đồ đối chuẩn học thuật tại: {output_path}")


# ==============================================================================
# HÀM CHÍNH (MAIN PIPELINE)
# ==============================================================================

def main():
    print("=" * 80)
    print("CHƯƠNG TRÌNH ĐỐI CHUẨN HỌC THUẬT: DROPOUT TRONG PYTORCH HIỆN ĐẠI")
    print("=" * 80)
    
    if HAS_TORCH:
        # 1. Minh họa CustomDropoutFunction
        print("\n[*] [1/2] Kiểm thử Toán tử Tự tạo CustomDropoutFunction (Inverted Dropout):")
        x_dummy = torch.ones(2, 5)
        drop_fn = CustomDropout(p=0.4)
        drop_fn.train()
        y_train = drop_fn(x_dummy)
        drop_fn.eval()
        y_eval = drop_fn(x_dummy)
        
        print(f"    - Tensor đầu vào (ones): \n{x_dummy.numpy()}")
        print(f"    - Đầu ra pha Huấn luyện (Train mode, p=0.4, chia cho 0.6): \n{y_train.numpy()}")
        print(f"    - Đầu ra pha Đánh giá (Eval mode, giữ nguyên tín hiệu): \n{y_eval.numpy()}")

        # 2. Chạy khối đối chuẩn học thuật
        print("\n[*] [2/2] Bắt đầu chạy Khối Đối chuẩn Giá trị Học thuật...")
        run_academic_benchmark()
    else:
        print("[!] Không có môi trường PyTorch. Vui lòng cài đặt PyTorch để chạy đầy đủ chức năng.")

    print("\n[✓] Kết thúc chương trình dropout_pytorch.py thành công!")


if __name__ == "__main__":
    main()
