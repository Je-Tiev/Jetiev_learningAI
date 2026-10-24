"""
================================================================================
BÀI BÁO: Visualizing Data using t-SNE (van der Maaten & Hinton, JMLR 2008)
TỆP MÃ NGUỒN: tsne_pytorch.py
MÔ TẢ: Triển khai thuật toán t-SNE hiện đại tận dụng đồ thị tính toán và Autograd của PyTorch.
       Bao gồm đầy đủ các tính năng:
       1. Lớp PyTorchTSNE kế thừa cơ chế tối ưu hóa tensor và GPU/CPU.
       2. Tối ưu hóa hàm mất mát KL-Divergence trực tiếp qua torch.optim (SGD với Momentum).
       3. Kỹ thuật Early Exaggeration và Zero-Centering chuẩn hóa.
       4. Cơ chế nạp dữ liệu thông minh với Bộ sinh dữ liệu Fallback an toàn khi chạy offline.
       5. So sánh trực tiếp song song với giải thuật phân tích thành phần chính (PCA Baseline).
       6. Tự động xuất hình ảnh trực quan hóa độ phân giải cao vào thư mục outputs/.
================================================================================
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Thử nghiệm nạp matplotlib và scikit-learn nếu môi trường hỗ trợ
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ==============================================================================
# PHẦN 1: TÍNH TOÁN XÁC SUẤT BẰNG PYTORCH TENSOR
# ==============================================================================

def compute_high_dim_probabilities_torch(X, target_perplexity=30.0, tol=1e-5, max_iter=50):
    """
    Tính ma trận xác suất liên hiệp đối xứng P trên PyTorch Tensor bằng phương pháp tìm kiếm nhị phân.
    """
    N = X.shape[0]
    # Tính ma trận khoảng cách bình phương: D_ij = ||x_i - x_j||^2
    sum_X = torch.sum(X ** 2, dim=1, keepdim=True)
    D = sum_X + sum_X.t() - 2.0 * torch.mm(X, X.t())
    D.fill_diagonal_(0.0)
    D = torch.clamp(D, min=0.0)

    P = torch.zeros((N, N), dtype=torch.float32, device=X.device)
    target_H = float(np.log(target_perplexity))

    for i in range(N):
        d_row = torch.cat([D[i, :i], D[i, i+1:]])

        beta_min = -float('inf')
        beta_max = float('inf')
        beta = 1.0

        for _ in range(max_iter):
            p_row = torch.exp(-d_row * beta)
            sum_p = torch.sum(p_row)
            if sum_p < 1e-12:
                sum_p = torch.tensor(1e-12, device=X.device)

            H = torch.log(sum_p) + beta * torch.sum(d_row * p_row) / sum_p
            h_diff = float(H.item()) - target_H

            if abs(h_diff) < tol:
                break

            if h_diff > 0:
                beta_min = beta
                beta = beta * 2.0 if beta_max == float('inf') else (beta + beta_max) / 2.0
            else:
                beta_max = beta
                beta = beta / 2.0 if beta_min == -float('inf') else (beta + beta_min) / 2.0

        p_norm = p_row / sum_p
        P[i, :i] = p_norm[:i]
        P[i, i+1:] = p_norm[i:]

    # SNE Đối xứng: p_ij = (p_{j|i} + p_{i|j}) / (2N)
    P_sym = (P + P.t()) / (2.0 * N)
    P_sym.fill_diagonal_(0.0)
    P_sym = torch.clamp(P_sym, min=1e-12)
    P_sym = P_sym / torch.sum(P_sym)
    return P_sym


# ==============================================================================
# PHẦN 2: LỚP MÔ HÌNH PyTorchTSNE
# ==============================================================================

class PyTorchTSNE:
    """
    Module t-SNE triển khai bằng PyTorch:
    - Quản lý tọa độ nhúng Y như một tham số tối ưu hóa (Optimizable Tensor).
    - Tính toán phân phối t-Student và hàm mất mát KL-Divergence qua Autograd.
    - Hỗ trợ tối ưu hóa GPU nếu có CUDA.
    """
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 n_iter=1000, early_exaggeration_iter=250, exaggeration_factor=4.0,
                 device=None, random_state=42):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration_iter = early_exaggeration_iter
        self.exaggeration_factor = exaggeration_factor
        self.random_state = random_state

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def fit_transform(self, X):
        """
        Thực hiện giảm chiều dữ liệu đa chiều X thành không gian n_components.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_tensor = X.to(dtype=torch.float32, device=self.device)

        N, D = X_tensor.shape
        print(f"\n[PyTorch t-SNE] Thiết bị tính toán: {self.device}")
        print(f"Kích thước đầu vào: {N} mẫu x {D} chiều -> {self.n_components} chiều")

        start_time = time.time()

        # 1. Tính toán ma trận xác suất P
        P = compute_high_dim_probabilities_torch(X_tensor, self.perplexity)

        # 2. Khởi tạo tọa độ nhúng Y từ phân phối chuẩn nhỏ N(0, 1e-4)
        Y = torch.randn(N, self.n_components, dtype=torch.float32, device=self.device) * 1e-4
        Y.requires_grad_(True)

        # 3. Sử dụng bộ tối ưu SGD với Momentum (tương thích công thức nguyên bản của Hinton 2008)
        optimizer = optim.SGD([Y], lr=self.learning_rate, momentum=0.5)

        print("[*] Đang tối ưu hóa hàm mất mát KL-Divergence...")

        for it in range(1, self.n_iter + 1):
            # Điều chỉnh ma trận P theo giai đoạn Early Exaggeration
            current_P = P * self.exaggeration_factor if it <= self.early_exaggeration_iter else P

            # Điều chỉnh động lượng (Momentum transition: 0.5 -> 0.8)
            if it == self.early_exaggeration_iter + 1:
                for param_group in optimizer.param_groups:
                    param_group['momentum'] = 0.8

            optimizer.zero_grad()

            # Tính toán phân phối t-Student không gian thấp Q:
            # num = (1 + ||y_i - y_j||^2)^(-1)
            sum_Y = torch.sum(Y ** 2, dim=1, keepdim=True)
            dist_sq = sum_Y + sum_Y.t() - 2.0 * torch.mm(Y, Y.t())
            dist_sq.fill_diagonal_(0.0)
            dist_sq = torch.clamp(dist_sq, min=0.0)

            num = 1.0 / (1.0 + dist_sq)
            num.fill_diagonal_(0.0)
            sum_num = torch.sum(num)
            if sum_num < 1e-12:
                sum_num = torch.tensor(1e-12, device=self.device)

            Q = num / sum_num
            Q = torch.clamp(Q, min=1e-12)

            # Hàm mất mát KL-Divergence: C = sum(P * log(P / Q))
            loss = torch.sum(current_P * torch.log(current_P / Q))

            # Lan truyền ngược tự động (Autograd)
            loss.backward()
            optimizer.step()

            # Zero-centering: Trừ tâm để tọa độ luôn cân bằng quanh gốc (0, 0)
            with torch.no_grad():
                Y.sub_(Y.mean(dim=0, keepdim=True))

            if it % 100 == 0 or it == self.n_iter:
                with torch.no_grad():
                    real_kl = torch.sum(P * torch.log(P / Q)).item()
                phase = "[Early Exaggeration]" if it <= self.early_exaggeration_iter else "[Fine Tuning]"
                print(f"  > Epoch {it:4d}/{self.n_iter} | KL Loss: {real_kl:.5f} {phase}")

        elapsed = time.time() - start_time
        print(f"[*] Huấn luyện PyTorch t-SNE hoàn tất trong {elapsed:.2f} giây!\n")
        return Y.detach().cpu().numpy()


# ==============================================================================
# PHẦN 3: NẠP DỮ LIỆU & FALLBACK AN TOÀN KHI OFFLINE
# ==============================================================================

def load_data_with_fallback(n_samples=300):
    """
    Nạp dữ liệu thử nghiệm với cơ chế fallback tự động:
    1. Thử nạp bộ chữ số viết tay scikit-learn load_digits (8x8 = 64 chiều).
    2. Nếu không có internet hoặc thư viện, tự sinh tập dữ liệu 5 cụm trong không gian 50 chiều.
    """
    try:
        from sklearn.datasets import load_digits
        digits = load_digits()
        X = digits.data[:n_samples]
        y = digits.target[:n_samples]
        print(f"[*] Đã nạp thành công bộ dữ liệu Chữ số Scikit-Learn Digits ({len(X)} mẫu, 64 chiều, 10 lớp).")
        return X, y, "Scikit-Learn Hand-written Digits (64D)"
    except Exception as e:
        print(f"[!] Không thể nạp tập dữ liệu ngoài ({e}). Kích hoạt bộ sinh dữ liệu Fallback Synthetic Clusters...")
        np.random.seed(42)
        n_clusters = 5
        samples_per_cluster = n_samples // n_clusters
        n_features = 30

        X_list = []
        y_list = []
        for c in range(n_clusters):
            center = np.random.uniform(-12, 12, size=(n_features,))
            cluster_samples = center + np.random.randn(samples_per_cluster, n_features) * 1.2
            X_list.append(cluster_samples)
            y_list.append(np.full((samples_per_cluster,), c))

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        print(f"[*] Đã sinh dữ liệu Fallback an toàn: {len(X)} mẫu, {n_features} chiều, {n_clusters} cụm.")
        return X, y, "Synthetic Gaussian Clusters (30D Fallback)"


# ==============================================================================
# PHẦN 4: SO SÁNH VỚI PCA BASELINE & TRỰC QUAN HÓA (VISUALIZATION)
# ==============================================================================

def compute_pca_baseline(X, n_components=2):
    """
    Giảm chiều bằng PCA Baseline:
    Sử dụng scikit-learn PCA hoặc phân rã SVD thuần NumPy nếu không có thư viện.
    """
    if HAS_SKLEARN:
        pca = PCA(n_components=n_components)
        return pca.fit_transform(X)
    else:
        # Tự tính PCA bằng SVD (X_centered = U * S * V^T)
        X_centered = X - np.mean(X, axis=0)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        return np.dot(X_centered, Vt[:n_components].T)


def plot_and_save_comparison(X, y, Y_tsne, dataset_title, output_dir="outputs"):
    """
    Vẽ biểu đồ so sánh song song giữa PCA Baseline và t-SNE Embedding,
    sau đó tự động lưu vào thư mục outputs/.
    """
    if not HAS_MATPLOTLIB:
        print("[!] Matplotlib chưa được cài đặt trong môi trường. Bỏ qua bước xuất ảnh.")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tsne_comparison.png")

    # 1. Tính toán PCA 2D
    Y_pca = compute_pca_baseline(X, n_components=2)

    # 2. Tạo hình vẽ so sánh 2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: PCA Baseline
    scatter1 = axes[0].scatter(Y_pca[:, 0], Y_pca[:, 1], c=y, cmap="tab10", alpha=0.8, edgecolors='none', s=35)
    axes[0].set_title("PCA Baseline (Linear Projection)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: t-SNE Embedding
    scatter2 = axes[1].scatter(Y_tsne[:, 0], Y_tsne[:, 1], c=y, cmap="tab10", alpha=0.8, edgecolors='none', s=35)
    axes[1].set_title("t-SNE Embedding (Non-linear Manifold)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.colorbar(scatter2, ax=axes, orientation='horizontal', fraction=0.05, pad=0.15, label='Class Label')
    fig.suptitle(f"So Sánh Trực Quan Hóa Đa Tạp: PCA vs t-SNE\nTập dữ liệu: {dataset_title}", fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[✓] Đã lưu biểu đồ so sánh độ phân giải cao thành công tại: {output_path}")


# ==============================================================================
# HÀM CHÍNH (MAIN ENTRYPOINT)
# ==============================================================================

def main():
    print("================================================================================")
    print("TRIỂN KHAI THUẬT TOÁN t-SNE BẰNG PYTORCH (tsne_pytorch.py)")
    print("================================================================================")

    # 1. Nạp tập dữ liệu (có fallback an toàn)
    X, y, title = load_data_with_fallback(n_samples=250)

    # 2. Khởi tạo và tối ưu hóa bằng PyTorchTSNE
    tsne_model = PyTorchTSNE(
        n_components=2,
        perplexity=25.0,
        learning_rate=150.0,
        n_iter=600,
        early_exaggeration_iter=200,
        exaggeration_factor=4.0,
        random_state=42
    )

    Y_embedded = tsne_model.fit_transform(X)

    # 3. Xuất biểu đồ so sánh song song với PCA Baseline
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    plot_and_save_comparison(X, y, Y_embedded, title, output_dir=output_folder)

    print("[✓] Toàn bộ quy trình PyTorch t-SNE đã hoàn tất xuất sắc!")


if __name__ == "__main__":
    main()
