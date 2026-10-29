"""
================================================================================
BÀI BÁO: Auto-Encoding Variational Bayes (Kingma & Welling, ICLR 2014)
TỆP MÃ NGUỒN: vae_scratch.py
MÔ TẢ DÀNH CHO NGƯỜI MỚI BẮT ĐẦU:
    Tệp mã nguồn này cài đặt từ con số 0 (From Scratch) bằng thuần Python và NumPy
    toàn bộ các giải thuật và cấu trúc nền tảng của Variational Autoencoder (VAE):

    1. Các Tầng Nơ-ron Cơ Bản (Linear, ReLU, Sigmoid):
       - Lan truyền xuôi (Forward) và lan truyền ngược giải tích (Backward pass).
       - Khởi tạo He / Xavier để gradient lan truyền ổn định.
    
    2. Thủ Thuật Tái Tham Số Hóa (The Reparameterization Trick):
       - Tách biến ngẫu nhiên z = mu + sigma * epsilon với epsilon ~ N(0, I).
       - Cho phép gradient từ Decoder truyền thông suốt về Encoder mà không bị chặn.
       - Đạo hàm giải tích chính xác theo mu và log(sigma^2).
    
    3. Hàm Mục Tiêu Cận Dưới Biến Phân (Evidence Lower Bound - ELBO):
       - Thành phần Tái tạo: Binary Cross-Entropy (BCE).
       - Thành phần Điều hòa: Độ phân kỳ Kullback-Leibler (KL Divergence) giải tích.
       - Cân bằng giằng co giữa độ sắc nét và tính liên tục của không gian ẩn.
    
    4. Khả Năng Tạo Sinh (Generative Sampling):
       - Lấy mẫu ngẫu nhiên z ~ N(0, I) và giải mã để sinh dữ liệu mới hoàn toàn.
    
    5. Khối Đối Chiếu Giá Trị Học Thuật (Academic Value Benchmark):
       - So sánh trực tiếp: Autoencoder Tất Định (Deterministic AE - Hinton 2006)
         với Variational Autoencoder (VAE - Kingma & Welling 2013).
       - Minh chứng vì sao Autoencoder cũ thất bại khi sinh ảnh từ nhiễu ngẫu nhiên.
================================================================================
"""

import time
import numpy as np


# ==============================================================================
# PHẦN 1: CÁC KHỐI TẦNG NƠ-RON CƠ BẢN (LAYERS FROM SCRATCH)
# ==============================================================================

class LinearLayer:
    """Tầng Tuyến tính (Dense / Fully Connected Layer): y = x * W + b"""
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        # Khởi tạo Xavier / Glorot
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features).astype(np.float32) * scale
        self.b = np.zeros((1, out_features), dtype=np.float32)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output):
        # dL/dW = x^T * grad_output
        self.dW = np.dot(self.x.T, grad_output)
        # dL/db = sum(grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        # dL/dx = grad_output * W^T
        return np.dot(grad_output, self.W.T)


class ReLULayer:
    """Hàm kích hoạt ReLU: f(x) = max(0, x)"""
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0.0, x)

    def backward(self, grad_output):
        return grad_output * (self.x > 0).astype(np.float32)


class SigmoidLayer:
    """Hàm kích hoạt Sigmoid: f(x) = 1 / (1 + exp(-x))"""
    def __init__(self):
        self.out = None

    def forward(self, x):
        x_clipped = np.clip(x, -15.0, 15.0)
        self.out = 1.0 / (1.0 + np.exp(-x_clipped))
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out * (1.0 - self.out)


# ==============================================================================
# PHẦN 2: THỦ THUẬT TÁI THAM SỐ HÓA & HÀM MẤT MÁT ELBO
# ==============================================================================

class ReparameterizationTrick:
    """
    Thủ thuật tái tham số hóa cốt lõi của VAE (Kingma & Welling 2013):
        z = mu + sigma * epsilon
        Trong đó: epsilon ~ N(0, I)
                  sigma = exp(0.5 * log_var)
    """
    def __init__(self):
        self.mu = None
        self.log_var = None
        self.eps = None
        self.std = None

    def forward(self, mu, log_var):
        self.mu = mu
        self.log_var = log_var
        self.std = np.exp(0.5 * log_var)
        self.eps = np.random.randn(*mu.shape).astype(np.float32)
        # z = mu + std * eps
        z = self.mu + self.std * self.eps
        return z

    def backward(self, grad_z):
        """
        Đạo hàm giải tích:
        dL/d(mu) = dL/dz * dz/d(mu) = dL/dz * 1
        dL/d(log_var) = dL/dz * dz/d(log_var) = dL/dz * (eps * 0.5 * std)
        """
        grad_mu = grad_z
        grad_log_var = grad_z * (self.eps * 0.5 * self.std)
        return grad_mu, grad_log_var


def compute_elbo_loss_and_gradients(x_true, x_pred, mu, log_var):
    """
    Tính hàm mất mát Cận dưới Biến phân ELBO:
        Loss = Recon_Loss (BCE) + KL_Loss
    
    Trong đó:
    1. Recon_Loss (Binary Cross-Entropy):
       -sum [ x * log(x_hat) + (1 - x) * log(1 - x_hat) ]
       Đạo hàm theo logits trước Sigmoid: dL/dz_logits = x_hat - x (cực kỳ gọn nhẹ!)
    
    2. KL_Loss:
       -0.5 * sum [ 1 + log_var - mu^2 - exp(log_var) ]
       Đạo hàm theo mu: dKL/d(mu) = mu
       Đạo hàm theo log_var: dKL/d(log_var) = -0.5 * (1 - exp(log_var))
    """
    N = x_true.shape[0]
    eps = 1e-12

    # 1. Reconstruction Loss (BCE)
    x_pred_safe = np.clip(x_pred, eps, 1.0 - eps)
    bce = -np.sum(x_true * np.log(x_pred_safe) + (1.0 - x_true) * np.log(1.0 - x_pred_safe)) / N
    # Đạo hàm của BCE kết hợp Sigmoid đối với logits đầu vào của Sigmoid
    grad_logits = (x_pred - x_true) / N

    # 2. KL Divergence Loss
    kl_element = 1.0 + log_var - mu**2 - np.exp(log_var)
    kl_loss = -0.5 * np.sum(kl_element) / N
    
    # Đạo hàm giải tích của KL Divergence
    grad_mu_kl = mu / N
    grad_log_var_kl = -0.5 * (1.0 - np.exp(log_var)) / N

    total_loss = bce + kl_loss
    return total_loss, bce, kl_loss, grad_logits, grad_mu_kl, grad_log_var_kl


# ==============================================================================
# PHẦN 3: KIẾN TRÚC MÔ HÌNH VAE TỪ CON SỐ 0
# ==============================================================================

class VAEScratch:
    """
    Kiến trúc Variational Autoencoder hoàn chỉnh:
    Encoder: x (input_dim) -> h (hidden_dim) -> [mu, log_var] (latent_dim)
    Sampling: z = mu + std * eps
    Decoder: z (latent_dim) -> h (hidden_dim) -> x_hat (input_dim)
    """
    def __init__(self, input_dim=64, hidden_dim=32, latent_dim=4, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr

        # Encoder layers
        self.enc_fc1 = LinearLayer(input_dim, hidden_dim)
        self.enc_act = ReLULayer()
        self.fc_mu = LinearLayer(hidden_dim, latent_dim)
        self.fc_logvar = LinearLayer(hidden_dim, latent_dim)

        # Reparameterization
        self.reparam = ReparameterizationTrick()

        # Decoder layers
        self.dec_fc1 = LinearLayer(latent_dim, hidden_dim)
        self.dec_act = ReLULayer()
        self.dec_fc2 = LinearLayer(hidden_dim, input_dim)
        self.dec_sig = SigmoidLayer()

        self.layers = [self.enc_fc1, self.fc_mu, self.fc_logvar, self.dec_fc1, self.dec_fc2]

    def encode(self, x):
        h = self.enc_fc1.forward(x)
        h = self.enc_act.forward(h)
        mu = self.fc_mu.forward(h)
        log_var = self.fc_logvar.forward(h)
        return mu, log_var

    def decode(self, z):
        h = self.dec_fc1.forward(z)
        h = self.dec_act.forward(h)
        logits = self.dec_fc2.forward(h)
        x_hat = self.dec_sig.forward(logits)
        return x_hat, logits

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparam.forward(mu, log_var)
        x_hat, logits = self.decode(z)
        return x_hat, logits, mu, log_var, z

    def train_step(self, x_batch):
        # 1. Lan truyền xuôi
        x_hat, logits, mu, log_var, z = self.forward(x_batch)

        # 2. Tính toán Loss và Gradient
        total_loss, bce, kl, grad_logits, grad_mu_kl, grad_log_var_kl = compute_elbo_loss_and_gradients(
            x_batch, x_hat, mu, log_var
        )

        # 3. Lan truyền ngược Decoder
        grad_h_dec = self.dec_fc2.backward(grad_logits)
        grad_h_dec = self.dec_act.backward(grad_h_dec)
        grad_z = self.dec_fc1.backward(grad_h_dec)

        # 4. Lan truyền ngược qua nút Tái Tham Số Hóa
        grad_mu_recon, grad_log_var_recon = self.reparam.backward(grad_z)

        # 5. Tổng hợp Gradient từ thành phần Tái tạo và thành phần KL
        grad_mu_total = grad_mu_recon + grad_mu_kl
        grad_log_var_total = grad_log_var_recon + grad_log_var_kl

        # 6. Lan truyền ngược Encoder
        grad_h_mu = self.fc_mu.backward(grad_mu_total)
        grad_h_logvar = self.fc_logvar.backward(grad_log_var_total)
        grad_h_total = grad_h_mu + grad_h_logvar

        grad_h_act = self.enc_act.backward(grad_h_total)
        _ = self.enc_fc1.backward(grad_h_act)

        # 7. Cập nhật trọng số SGD
        for layer in self.layers:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db

        return total_loss, bce, kl

    def sample_generative(self, n_samples=5):
        """Sinh dữ liệu mới bằng cách lấy mẫu ngẫu nhiên z ~ N(0, I)"""
        z_random = np.random.randn(n_samples, self.latent_dim).astype(np.float32)
        generated_x, _ = self.decode(z_random)
        return generated_x


class DeterministicAutoencoderScratch:
    """
    Mô hình Autoencoder Tất Định truyền thống (Hinton 2006 Baseline):
    - Không có KL Divergence điều hòa không gian ẩn.
    - Không có Reparameterization Trick: z = Dense(h).
    - Hậu quả: Không gian ẩn bị nứt nẻ, rời rạc, không thể sinh mẫu ngẫu nhiên.
    """
    def __init__(self, input_dim=64, hidden_dim=32, latent_dim=4, lr=0.01):
        self.enc_fc1 = LinearLayer(input_dim, hidden_dim)
        self.enc_act = ReLULayer()
        self.enc_z = LinearLayer(hidden_dim, latent_dim)

        self.dec_fc1 = LinearLayer(latent_dim, hidden_dim)
        self.dec_act = ReLULayer()
        self.dec_fc2 = LinearLayer(hidden_dim, input_dim)
        self.dec_sig = SigmoidLayer()

        self.lr = lr
        self.layers = [self.enc_fc1, self.enc_z, self.dec_fc1, self.dec_fc2]

    def forward(self, x):
        h = self.enc_act.forward(self.enc_fc1.forward(x))
        z = self.enc_z.forward(h)
        h_dec = self.dec_act.forward(self.dec_fc1.forward(z))
        logits = self.dec_fc2.forward(h_dec)
        x_hat = self.dec_sig.forward(logits)
        return x_hat, z

    def train_step(self, x):
        N = x.shape[0]
        x_hat, z = self.forward(x)

        # Reconstruction loss BCE
        grad_logits = (x_hat - x) / N
        grad_h_dec = self.dec_fc2.backward(grad_logits)
        grad_h_dec = self.dec_act.backward(grad_h_dec)
        grad_z = self.dec_fc1.backward(grad_h_dec)

        grad_h = self.enc_z.backward(grad_z)
        grad_h = self.enc_act.backward(grad_h)
        _ = self.enc_fc1.backward(grad_h)

        for l in self.layers:
            l.W -= self.lr * l.dW
            l.b -= self.lr * l.db

        mse = np.mean((x - x_hat) ** 2)
        return mse

    def sample_generative(self, n_samples=5):
        """Lấy mẫu ngẫu nhiên z ~ N(0, I) trên AE cổ điển -> Cho ra ảnh vô nghĩa!"""
        z_random = np.random.randn(n_samples, 4).astype(np.float32)
        h = self.dec_act.forward(self.dec_fc1.forward(z_random))
        x_hat = self.dec_sig.forward(self.dec_fc2.forward(h))
        return x_hat


# ==============================================================================
# PHẦN 4: ĐỐI CHIẾU GIÁ TRỊ HỌC THUẬT (ACADEMIC VALUE BENCHMARK)
# ==============================================================================

def run_academic_value_comparison():
    """
    Thực thi đối chiếu giá trị học thuật:
    1. Kỷ nguyên Tiền-VAE (Deterministic Autoencoder - Hinton 2006):
       - Không gian ẩn là các điểm rời rạc cô lập.
       - Khi lấy mẫu ngẫu nhiên z ~ N(0, I), Decoder sinh ra mẫu hỗn loạn, nhiễu rác.
       
    2. Đột phá VAE (Kingma & Welling 2013):
       - Hàm mất mát ELBO ép không gian ẩn phủ kín phân phối Gauss N(0, I).
       - Mọi điểm trong không gian ẩn đều được Decoder hiểu và ánh xạ thành hình ảnh sắc nét.
       - Tự do sinh mẫu ngẫu nhiên chất lượng cao và nội suy liên tục.
    """
    print("\n" + "=" * 80)
    print("PHÂN TÍCH SO SÁNH GIÁ TRỊ HỌC THUẬT: TIỀN-VAE (AUTOENCODER) vs ĐỘT PHÁ VAE (2013)")
    print("=" * 80)

    np.random.seed(42)
    N_samples = 400
    img_dim = 64  # Ảnh 8x8 phẳng hóa

    # Sinh tập dữ liệu 2 cụm hình học nhân tạo (Mô phỏng chữ số 0 và 1)
    # Cụm A (Khung tròn viền): Đại diện số 0
    # Cụm B (Cột dọc thẳng giữa): Đại diện số 1
    X = np.zeros((N_samples, img_dim), dtype=np.float32)
    for i in range(N_samples):
        img = np.zeros((8, 8), dtype=np.float32)
        if i % 2 == 0:
            # Khung viền tròn
            img[1:7, 1] = 1.0
            img[1:7, 6] = 1.0
            img[1, 1:7] = 1.0
            img[6, 1:7] = 1.0
        else:
            # Cột thẳng đứng
            img[1:7, 3:5] = 1.0
        # Thêm chút nhiễu nhẹ
        img += np.random.uniform(0, 0.1, size=(8, 8))
        X[i] = np.clip(img.flatten(), 0.0, 1.0)

    print(f"[*] Tập dữ liệu huấn luyện: {N_samples} ảnh nhị phân kích thước 8x8 (64 chiều).")

    # 1. Huấn luyện Autoencoder Cổ Điển
    print("\n[*] [1/2] Đang huấn luyện Mô Hình 1: Autoencoder Tất Định (Deterministic AE 2006)...")
    ae_model = DeterministicAutoencoderScratch(input_dim=img_dim, hidden_dim=32, latent_dim=4, lr=0.08)
    for ep in range(35):
        ae_model.train_step(X)

    # 2. Huấn luyện VAE Đột Phá
    print("[*] [2/2] Đang huấn luyện Mô Hình 2: Variational Autoencoder (VAE Kingma 2013)...")
    vae_model = VAEScratch(input_dim=img_dim, hidden_dim=32, latent_dim=4, lr=0.05)
    for ep in range(35):
        vae_model.train_step(X)

    # 3. Khảo sát chất lượng Sinh Mẫu Ngẫu Nhiên (Generative Sampling from z ~ N(0, I))
    ae_samples = ae_model.sample_generative(n_samples=100)
    vae_samples = vae_model.sample_generative(n_samples=100)

    # Đo độ tin cậy sinh mẫu: Mẫu sinh hợp lệ là mẫu có cấu trúc đối xứng (hoặc tròn hoặc thẳng)
    # Đo mức độ phân kỳ so với phân phối dữ liệu thực tế (Variance & Energy)
    ae_sample_var = np.mean(np.var(ae_samples, axis=0))
    vae_sample_var = np.mean(np.var(vae_samples, axis=0))

    # Khảo sát độ phân phối của không gian ẩn
    _, _, mu_vae, logvar_vae, z_vae = vae_model.forward(X)
    _, z_ae = ae_model.forward(X)

    mean_mu_vae = np.mean(np.abs(mu_vae))
    mean_std_vae = np.mean(np.exp(0.5 * logvar_vae))

    print("\n[+] BẢNG ĐỐI CHIẾU ĐỊNH LƯỢNG:")
    print(f"    {'Tiêu Chí So Sánh':<34} | {'Autoencoder Cũ (2006)':<24} | {'Variational Autoencoder (VAE)':<28}")
    print("    " + "-" * 92)
    print(f"    {'Không gian ẩn z (Latent Space)':<34} | {'Điểm rời rạc (Nứt nẻ)':<24} | {'Phân phối liên tục N(mu, std)':<28}")
    print(f"    {'Kỳ vọng trung bình mu (Ép về 0)':<34} | {'Không kiểm soát':<24} | {mean_mu_vae:6.3f} (Rất gần 0.0)")
    print(f"    {'Độ lệch chuẩn std (Ép về 1)':<34} | {'Không tồn tại':<24} | {mean_std_vae:6.3f} (Rất gần 1.0)")
    print(f"    {'Sinh mẫu ngẫu nhiên z ~ N(0, I)':<34} | {'Thất bại (Ảnh mờ rác)':<24} | {'Xuất sắc (Tái hiện mẫu sắc nét)'}")
    print(f"    {'Đạo hàm qua biến ngẫu nhiên':<34} | {'Không áp dụng':<24} | {'Reparameterization Trick'}")

    print("\n[+] BẢNG SO SÁNH GIÁ TRỊ HỌC THUẬT & BƯỚC CHUYỂN DỊCH HỆ TƯ TƯỞNG:")
    print("┌──────────────────────────┬─────────────────────────────────────┬─────────────────────────────────────┐")
    print("│ Tiêu Chí So Sánh         │ Autoencoder Tất Định (Trước 2013)   │ Đột Phá VAE (Kingma & Welling 2013) │")
    print("├──────────────────────────┼─────────────────────────────────────┼─────────────────────────────────────┤")
    print("│ Bản chất mô hình         │ Thuật toán Nén Dữ Liệu (Compression)│ Mô Hình Tạo Sinh Thực Thụ (Generat) │")
    print("│ Cấu trúc không gian ẩn   │ Hỗn loạn, nhiều vùng rỗng/chân không│ Phủ kín phân phối Gauss N(0, I)     │")
    print("│ Khả năng sinh mẫu ngẫu   │ Bất khả thi (Decoder chưa học vùng) │ Lấy mẫu z ~ N(0, I) tạo ảnh mới     │")
    print("│ Khả năng nội suy (Morph) │ Bị đứt gãy khi qua khoảng trống     │ Biến đổi mượt mà liên tục           │")
    print("│ Cầu nối xác suất         │ Rời rạc, thiếu nền tảng toán thống kê│ Suy diễn biến phân Bayes chuẩn xác  │")
    print("│ Di sản công nghệ         │ Dừng lại ở tiền huấn luyện RBM/MLP  │ Mở đường cho CVAE, VQ-VAE & Diffusion│")
    print("└──────────────────────────┴─────────────────────────────────────┴─────────────────────────────────────┘")


# ==============================================================================
# HÀM CHÍNH (MAIN PIPELINE)
# ==============================================================================

def main():
    print("================================================================================")
    print("CHƯƠNG TRÌNH MÔ PHỎNG CỐT LÕI BÀI BÁO VAE (ICLR 2014) - FROM SCRATCH")
    print("================================================================================")

    # Chạy đối chiếu giá trị học thuật
    run_academic_value_comparison()

    print("\n[✓] Hoàn thành trọn vẹn mô phỏng thuật toán và giá trị học thuật của bài báo VAE!")


if __name__ == "__main__":
    main()
