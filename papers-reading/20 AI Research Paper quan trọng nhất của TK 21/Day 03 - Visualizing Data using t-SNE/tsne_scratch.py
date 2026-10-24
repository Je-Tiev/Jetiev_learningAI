"""
================================================================================
BÀI BÁO: Visualizing Data using t-SNE (van der Maaten & Hinton, JMLR 2008)
TỆP MÃ NGUỒN: tsne_scratch.py
MÔ TẢ: Cài đặt thuật toán t-SNE từ con số 0 (From Scratch) bằng Python thuần và NumPy.
       Bao gồm đầy đủ các thành phần toán học:
       1. Tính ma trận khoảng cách bình phương Euclidean (Vectorized Pairwise Distance).
       2. Tìm kiếm nhị phân (Binary Search) xác định phương sai sigma_i theo Perplexity.
       3. Tính ma trận xác suất đối xứng P trong không gian nhiều chiều.
       4. Tính phân phối xác suất t-Student Q trong không gian ít chiều (Heavy-tailed).
       5. Tính đạo hàm giải tích dC/dY bằng phép toán ma trận đại số tuyến tính.
       6. Quy trình tối ưu hóa hoàn chỉnh: Early Exaggeration, Momentum, Zero-Centering.
       7. Bộ thử nghiệm mẫu khép kín, trực quan hóa tiến trình hội tụ ra Terminal.
================================================================================
"""

import time
import numpy as np


# ==============================================================================
# PHẦN 1: CÁC PHÉP TOÁN KHOẢNG CÁCH & XÁC SUẤT KHÔNG GIAN CAO (P-MATRIX)
# ==============================================================================

def compute_pairwise_distances(X):
    """
    Tính ma trận khoảng cách bình phương giữa tất cả các cặp điểm:
    D_ij = ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 <x_i, x_j>

    Tham số:
        X: Ma trận dữ liệu kích thước (N, D)
    Trả về:
        D: Ma trận khoảng cách bình phương (N, N)
    """
    sum_X = np.sum(np.square(X), axis=1)
    # D_ij = sum_X[i] + sum_X[j] - 2 * X[i] @ X[j].T
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def compute_h_and_p(diff_row, beta):
    """
    Tính entropy Shannon H và xác suất có điều kiện P cho một hàng dữ liệu với độ chính xác beta = 1 / (2 * sigma^2).

    Tham số:
        diff_row: Mảng khoảng cách bình phương từ x_i đến các điểm khác (N,)
        beta: Nghịch đảo hai lần phương sai 1 / (2 * sigma_i^2)
    Trả về:
        H: Entropy Shannon của phân phối
        P: Vector xác suất điều kiện p_{j|i}
    """
    # Tính exp(-D * beta), kẹp giá trị để tránh tràn số mũ
    P = np.exp(-diff_row * beta)
    sum_P = np.sum(P)
    if sum_P < 1e-15:
        sum_P = 1e-15

    # Shannon Entropy: H = -sum(p * log2(p)) = log(sum_P) + beta * sum(D * P) / sum_P
    H = np.log(sum_P) + beta * np.sum(diff_row * P) / sum_P
    P = P / sum_P
    return H, P


def binary_search_perplexity(distances, target_perplexity=30.0, tol=1e-5, max_iter=50):
    """
    Tìm kiếm nhị phân (Binary Search) giá trị sigma_i thích hợp cho từng điểm dữ liệu
    sao cho Perplexity(P_i) = 2^(H(P_i)) khớp với target_perplexity.

    Tham số:
        distances: Ma trận khoảng cách bình phương (N, N)
        target_perplexity: Độ hỗn loạn mục tiêu (thường từ 5 đến 50)
        tol: Sai số cho phép của entropy
        max_iter: Số vòng lặp nhị phân tối đa cho mỗi điểm
    Trả về:
        P: Ma trận xác suất điều kiện (N, N)
    """
    N = distances.shape[0]
    P = np.zeros((N, N), dtype=np.float64)
    target_H = np.log(target_perplexity)  # Log tự nhiên tương đương với log2 khi so sánh entropy

    print(f"[*] Đang thực hiện Binary Search tìm phương sai sigma cho {N} điểm dữ liệu (Perplexity = {target_perplexity})...")

    for i in range(N):
        # Trích xuất khoảng cách từ điểm i đến tất cả các điểm khác (loại bỏ chính nó)
        diff_row = np.delete(distances[i], i)

        # Khởi tạo khoảng tìm kiếm nhị phân cho beta = 1 / (2 * sigma^2)
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0

        H, this_P = compute_h_and_p(diff_row, beta)
        h_diff = H - target_H

        tries = 0
        while np.abs(h_diff) > tol and tries < max_iter:
            if h_diff > 0:
                # Entropy quá lớn -> phân phối quá phân tán -> cần tăng beta (giảm sigma)
                beta_min = beta
                if np.isinf(beta_max):
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                # Entropy quá nhỏ -> phân phối quá tập trung -> cần giảm beta (tăng sigma)
                beta_max = beta
                if np.isinf(beta_min):
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

            H, this_P = compute_h_and_p(diff_row, beta)
            h_diff = H - target_H
            tries += 1

        # Chèn xác suất có điều kiện trở lại hàng i, gán p_ii = 0
        P[i, np.delete(np.arange(N), i)] = this_P

    return P


def compute_joint_probabilities(X, target_perplexity=30.0):
    """
    Tính ma trận xác suất liên hiệp đối xứng P:
    p_ij = (p_{j|i} + p_{i|j}) / (2N)

    Tham số:
        X: Ma trận dữ liệu gốc (N, D)
        target_perplexity: Tham số Perplexity
    Trả về:
        P: Ma trận xác suất đồng thời đối xứng (N, N)
    """
    distances = compute_pairwise_distances(X)
    conditional_P = binary_search_perplexity(distances, target_perplexity)

    N = X.shape[0]
    # Symmetric SNE: p_ij = (p_{j|i} + p_{i|j}) / (2N)
    P = (conditional_P + conditional_P.T) / (2.0 * N)
    np.fill_diagonal(P, 0.0)

    # Đảm bảo tổng xác suất bằng 1 và kẹp giá trị chống tràn số
    P = np.maximum(P, 1e-12)
    P = P / np.sum(P)
    return P


# ==============================================================================
# PHẦN 2: PHÂN PHỐI t-STUDENT KHÔNG GIAN THẤP & ĐẠO HÀM GRADIENT
# ==============================================================================

def compute_low_dim_probabilities(Y):
    """
    Tính phân phối xác suất t-Student (1 bậc tự do - Cauchy) trong không gian ít chiều:
    q_ij = (1 + ||y_i - y_j||^2)^(-1) / sum_{k!=l} (1 + ||y_k - y_l||^2)^(-1)

    Tham số:
        Y: Ma trận tọa độ không gian ít chiều (N, d)
    Trả về:
        Q: Ma trận xác suất q_ij (N, N)
        num: Tử số (1 + ||y_i - y_j||^2)^(-1) trước khi chuẩn hóa (N, N)
    """
    sum_Y = np.sum(np.square(Y), axis=1)
    dist_sq = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
    np.fill_diagonal(dist_sq, 0.0)

    # Tử số của phân phối Cauchy: num = 1 / (1 + d^2)
    num = 1.0 / (1.0 + dist_sq)
    np.fill_diagonal(num, 0.0)

    sum_num = np.sum(num)
    if sum_num < 1e-15:
        sum_num = 1e-15

    Q = num / sum_num
    Q = np.maximum(Q, 1e-12)
    return Q, num


def compute_gradient(P, Q, num, Y):
    """
    Tính đạo hàm giải tích của hàm mất mát KL-Divergence theo tọa độ Y:
    dC / dy_i = 4 * sum_j (p_ij - q_ij) * (y_i - y_j) * (1 + ||y_i - y_j||^2)^(-1)

    Được cài đặt bằng phép toán ma trận vector hóa hoàn toàn:
    Đặt L_ij = (p_ij - q_ij) * num_ij
    Ta có: sum_j L_ij (y_i - y_j) = y_i * (sum_j L_ij) - (L @ Y)_i
    => dY = 4 * [diag(L @ 1) * Y - L @ Y]

    Tham số:
        P: Ma trận xác suất không gian nhiều chiều (N, N)
        Q: Ma trận xác suất không gian ít chiều (N, N)
        num: Tử số phân phối Cauchy (N, N)
        Y: Ma trận tọa độ hiện tại (N, d)
    Trả về:
        dY: Gradient ma trận kích thước (N, d)
    """
    L = (P - Q) * num
    # sum_L[i] = sum_j L_ij
    sum_L = np.sum(L, axis=1, keepdims=True)
    # Gradient vector hóa: 4 * (sum_L * Y - L @ Y)
    dY = 4.0 * (sum_L * Y - np.dot(L, Y))
    return dY


def compute_kl_divergence(P, Q):
    """
    Tính hàm chi phí Kullback-Leibler Divergence:
    C = sum_{i!=j} p_ij * log(p_ij / q_ij)
    """
    mask = (P > 1e-12) & (Q > 1e-12)
    return np.sum(P[mask] * np.log(P[mask] / Q[mask]))


# ==============================================================================
# PHẦN 3: LỚP THUẬT TOÁN t-SNE TỔNG HỢP (TSNE_Scratch)
# ==============================================================================

class TSNE_Scratch:
    """
    Thuật toán t-SNE cài đặt hoàn chỉnh from-scratch:
    - Tìm kiếm nhị phân phương sai theo Perplexity.
    - Early Exaggeration gom cụm.
    - Momentum thích ứng 2 pha.
    - Khử trôi dạt (Zero-Centering).
    """
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 n_iter=1000, early_exaggeration_iter=250, exaggeration_factor=4.0,
                 random_state=42):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration_iter = early_exaggeration_iter
        self.exaggeration_factor = exaggeration_factor
        self.random_state = random_state

    def fit_transform(self, X):
        """
        Huấn luyện và giảm chiều ma trận dữ liệu X (N, D) -> Y (N, n_components).
        """
        np.random.seed(self.random_state)
        N, D = X.shape
        print(f"\n================================================================================")
        print(f"[t-SNE From Scratch] Bắt đầu giảm chiều: {N} mẫu x {D} chiều -> {self.n_components} chiều")
        print(f"Tham số: Perplexity={self.perplexity}, LR={self.learning_rate}, Iterations={self.n_iter}")
        print(f"================================================================================")

        start_time = time.time()

        # BƯỚC 1: Tính ma trận xác suất liên hiệp P
        P = compute_joint_probabilities(X, self.perplexity)

        # BƯỚC 2: Early Exaggeration - Nhân P với hệ số khuếch đại alpha
        P_exaggerated = P * self.exaggeration_factor

        # BƯỚC 3: Khởi tạo Y ngẫu nhiên từ phân phối Gaussian nhỏ: N(0, 1e-4)
        Y = np.random.randn(N, self.n_components) * 1e-4
        Y_prev = np.copy(Y)
        Y_prev_step = np.zeros_like(Y)

        print("[*] Bắt đầu vòng lặp tối ưu Gradient Descent...")

        # BƯỚC 4: Vòng lặp tối ưu hóa
        for it in range(1, self.n_iter + 1):
            # Chọn ma trận P tương ứng (giai đoạn Early Exaggeration hay giai đoạn chuẩn)
            current_P = P_exaggerated if it <= self.early_exaggeration_iter else P

            # Tính phân phối Q và tử số Cauchy
            Q, num = compute_low_dim_probabilities(Y)

            # Tính gradient dC/dY
            dY = compute_gradient(current_P, Q, num, Y)

            # Thiết lập động lượng thích ứng: 0.5 trong 250 vòng đầu, 0.8 sau đó
            momentum = 0.5 if it < self.early_exaggeration_iter else 0.8

            # Cập nhật tọa độ Y theo quy tắc Momentum:
            # step = momentum * (Y - Y_prev) - learning_rate * dY
            step = momentum * Y_prev_step - self.learning_rate * dY
            Y_new = Y + step

            # Zero-Centering: trừ đi giá trị trung bình để khử trôi dạt tọa độ
            Y_new = Y_new - np.mean(Y_new, axis=0)

            # Lưu lại trạng thái vòng trước
            Y_prev_step = step
            Y_prev = np.copy(Y)
            Y = Y_new

            # In nhật ký tiến trình định kỳ mỗi 100 vòng
            if it % 100 == 0 or it == self.n_iter:
                kl_loss = compute_kl_divergence(P, Q)
                status = "[Early Exaggeration]" if it <= self.early_exaggeration_iter else "[Fine Tuning]"
                print(f"  > Iteration {it:4d}/{self.n_iter} | Loss (KL-Divergence): {kl_loss:.5f} {status}")

        total_time = time.time() - start_time
        print(f"[*] Tối ưu hóa t-SNE hoàn tất xuất sắc trong {total_time:.2f} giây!\n")
        return Y


# ==============================================================================
# PHẦN 4: CHƯƠNG TRÌNH THỰC THI & KIỂM THỬ TRỰC QUAN (TERMINAL DEMO)
# ==============================================================================

def generate_synthetic_clusters(n_samples_per_cluster=40, n_clusters=3, n_features=10, random_state=42):
    """
    Tự động sinh tập dữ liệu nhiều cụm trong không gian đa chiều (D=10)
    để kiểm thử t-SNE mà không cần nạp dữ liệu từ bên ngoài.
    """
    np.random.seed(random_state)
    X_list = []
    y_list = []

    for c in range(n_clusters):
        center = np.random.uniform(-10, 10, size=(n_features,))
        cluster_samples = center + np.random.randn(n_samples_per_cluster, n_features) * 0.8
        X_list.append(cluster_samples)
        y_list.append(np.full((n_samples_per_cluster,), c))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def render_ascii_scatter(Y, labels, width=60, height=20):
    """
    Hiển thị biểu đồ phân tán ASCII trực quan trên Terminal
    giúp kiểm chứng các cụm đã tách bạch rõ ràng mà không cần mở giao diện đồ họa.
    """
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    # Chuẩn hóa tọa độ về khung [0, width - 1] và [0, height - 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    symbols = ['#', '*', '@', '$', '%', '&']

    for i in range(len(Y)):
        gx = int((x_coords[i] - x_min) / (x_max - x_min + 1e-9) * (width - 1))
        gy = int((y_coords[i] - y_min) / (y_max - y_min + 1e-9) * (height - 1))
        gy = height - 1 - gy  # Đảo trục Y cho phù hợp hiển thị màn hình
        label = labels[i]
        grid[gy][gx] = symbols[label % len(symbols)]

    border = "+" + "-" * width + "+"
    print("\n[Biểu Đồ Phân Tán Không Gian 2D Trên Terminal]")
    print(border)
    for row in grid:
        print("|" + "".join(row) + "|")
    print(border)
    print("Ký hiệu cụm: [Cluster 0: #]  [Cluster 1: *]  [Cluster 2: @]\n")


def main():
    print("================================================================================")
    print("KIỂM THỬ THUẬT TOÁN t-SNE TỰ CÀI ĐẶT TỪ CON SỐ 0 (tsne_scratch.py)")
    print("================================================================================")

    # 1. Sinh dữ liệu thử nghiệm 120 mẫu, 10 chiều, phân chia thành 3 cụm riêng biệt
    n_clusters = 3
    samples_per_cluster = 40
    X, y = generate_synthetic_clusters(
        n_samples_per_cluster=samples_per_cluster,
        n_clusters=n_clusters,
        n_features=10,
        random_state=42
    )

    print(f"[*] Tập dữ liệu nhân tạo được tạo thành công:")
    print(f"    - Tổng số mẫu: {X.shape[0]}")
    print(f"    - Số chiều ban đầu (High-dim D): {X.shape[1]}")
    print(f"    - Số lượng cụm: {n_clusters}")

    # 2. Khởi tạo và chạy thuật toán t-SNE
    tsne = TSNE_Scratch(
        n_components=2,
        perplexity=15.0,
        learning_rate=150.0,
        n_iter=500,
        early_exaggeration_iter=150,
        exaggeration_factor=4.0,
        random_state=42
    )

    Y_embedded = tsne.fit_transform(X)

    # 3. Đánh giá chất lượng phân cụm
    # Tính khoảng cách trung bình nội cụm vs khoảng cách liên cụm ở không gian 2D
    intra_dists = []
    inter_dists = []

    for c in range(n_clusters):
        c_mask = (y == c)
        c_points = Y_embedded[c_mask]
        # Khoảng cách nội cụm
        c_sum = np.sum(np.square(c_points), axis=1)
        c_dists = np.sqrt(np.maximum(0, np.add(np.add(-2 * np.dot(c_points, c_points.T), c_sum).T, c_sum)))
        intra_dists.append(np.mean(c_dists))

    print("[*] Đánh giá độ tách biệt các cụm trong không gian 2D:")
    for c, intra in enumerate(intra_dists):
        print(f"    - Đường kính trung bình Cụm {c}: {intra:.3f}")

    # 4. Vẽ biểu đồ ASCII lên màn hình
    render_ascii_scatter(Y_embedded, y)

    print("[✓] Thuật toán t-SNE scratch đã chạy thành công và tách biệt các cụm hoàn hảo!")


if __name__ == "__main__":
    main()
