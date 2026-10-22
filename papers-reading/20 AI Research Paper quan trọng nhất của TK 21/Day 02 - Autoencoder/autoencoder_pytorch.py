"""
================================================================================
BÀI BÁO: Reducing the Dimensionality of Data with Neural Networks (Hinton & Salakhutdinov, Science 2006)
TỆP MÃ NGUỒN: autoencoder_pytorch.py
MÔ TẢ: Cài đặt mô hình Deep Autoencoder hiện đại bằng thư viện PyTorch.
       Bao gồm đầy đủ các tính năng:
       1. Deep MLP Autoencoder (mô phỏng kiến trúc 784 -> 1000 -> 500 -> 250 -> 30/2 của bài báo).
       2. Convolutional Autoencoder (biến thể hiện đại cho thị giác máy tính).
       3. Denoising Autoencoder (khử nhiễu ảnh).
       4. Cơ chế nạp dữ liệu MNIST kèm Fallback thông minh nếu không có kết nối internet.
       5. Trực quan hóa và so sánh trực tiếp với thuật toán PCA Baseline (tái hiện Figure 3 của bài báo).
       6. Tự động lưu biểu đồ kết quả vào thư mục `outputs/`.
================================================================================
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Thử import matplotlib và scikit-learn để trực quan hóa
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
# PHẦN 1: ĐỊNH NGHĨA CÁC KIẾN TRÚC MÔ HÌNH NƠ-RON
# ==============================================================================

class DeepAutoencoder(nn.Module):
    """
    Mô hình Deep Autoencoder theo sát kiến trúc trong bài báo Hinton 2006:
    Encoder:  784 -> 1000 -> 500 -> 250 -> latent_dim (mặc định 2 hoặc 30)
    Decoder:  latent_dim -> 250 -> 500 -> 1000 -> 784
    """
    def __init__(self, input_dim=784, latent_dim=2):
        super(DeepAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder (Bộ mã hóa)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(inplace=True),
            # Tầng Code Layer ở nút cổ chai: Trong bài báo dùng Linear units để biểu diễn liên tục
            nn.Linear(250, latent_dim)
        )

        # Decoder (Bộ giải mã)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(inplace=True),
            nn.Linear(250, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, input_dim),
            nn.Sigmoid()  # Giữ giá trị tái tạo trong đoạn [0, 1] như cường độ pixel
        )

    def encode(self, x):
        """Trích xuất vector tiềm ẩn z từ ảnh đầu vào"""
        return self.encoder(x)

    def decode(self, z):
        """Tái tạo lại ảnh từ vector tiềm ẩn z"""
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class ConvAutoencoder(nn.Module):
    """
    Mô hình Convolutional Autoencoder (CAE) hiện đại:
    Tận dụng tính chất bất biến tịnh tiến của phép tích chập (Convolution),
    giúp giảm đáng kể số lượng tham số so với MLP và giữ nguyên cấu trúc không gian 2D của ảnh.
    """
    def __init__(self, latent_dim=16):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: 1x28x28 -> 16x14x14 -> 32x7x7 -> latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 14, 14]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, 7, 7]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.fc_encode = nn.Linear(32 * 7 * 7, latent_dim)

        # Decoder: latent_dim -> 32x7x7 -> 16x14x14 -> 1x28x28
        self.fc_decode = nn.Linear(latent_dim, 32 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, 16, 14, 14]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.size(0)
        # Nén
        feat = self.encoder_conv(x.view(B, 1, 28, 28))
        z = self.fc_encode(feat.view(B, -1))
        # Giải nén
        h = self.fc_decode(z).view(B, 32, 7, 7)
        recon_2d = self.decoder_conv(h)
        return recon_2d.view(B, -1), z


# ==============================================================================
# PHẦN 2: CHUẨN BỊ DỮ LIỆU & FALLBACK OFFLINE AN TOÀN
# ==============================================================================

def get_mnist_or_synthetic_data(num_samples=2500):
    """
    Nạp dữ liệu MNIST từ torchvision nếu có internet.
    Nếu tải lỗi hoặc offline, tự động sinh tập dữ liệu chữ số nhân tạo (Toy Digits)
    để đảm bảo chương trình luôn chạy mượt mà 100% không bị ngắt quãng.
    """
    print("\n[Dữ liệu] Đang chuẩn bị dữ liệu...")
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        # Lấy subset mẫu để huấn luyện demo nhanh chóng
        indices = torch.randperm(len(dataset))[:num_samples]
        images = dataset.data[indices].float() / 255.0
        images = images.view(-1, 784)
        labels = dataset.targets[indices]
        print(f" -> Tải thành công tập dữ liệu MNIST gốc từ web! Lấy mẫu: {num_samples} ảnh.")
        return images, labels
    except Exception as e:
        print(f" -> Không thể tải MNIST từ web ({e}). Chuyển sang chế độ Fallback sinh dữ liệu chữ số tổng hợp...")
        return generate_synthetic_mnist(num_samples)


def generate_synthetic_mnist(num_samples=2500):
    """
    Sinh tập dữ liệu ảnh 28x28 tổng hợp gồm 10 loại hình thái (tương đương 10 nhãn chữ số 0-9):
    Bao gồm các hình tròn (0), nét thẳng đứng (1), đường cong zic-zac (2, 3), chữ thập (4), v.v.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    images = torch.zeros((num_samples, 784), dtype=torch.float32)
    labels = torch.zeros(num_samples, dtype=torch.long)

    for i in range(num_samples):
        digit = i % 10
        labels[i] = digit
        img = np.zeros((28, 28), dtype=np.float32)

        # Vẽ các nét đặc trưng cho từng loại chữ số
        if digit == 0:
            # Hình ellipse / vòng tròn
            rr, cc = np.ogrid[:28, :28]
            mask = ((rr - 14)**2 / 64 + (cc - 14)**2 / 36 <= 1.0) & ((rr - 14)**2 / 36 + (cc - 14)**2 / 16 >= 0.8)
            img[mask] = 1.0
        elif digit == 1:
            # Nét thẳng đứng ở giữa
            img[4:24, 13:16] = 1.0
        elif digit == 2:
            img[5:8, 8:20] = 1.0
            for r in range(8, 20):
                c = int(20 - (r - 8) * 1.0)
                img[r, max(0, c-1):min(28, c+2)] = 1.0
            img[20:23, 8:21] = 1.0
        elif digit == 3:
            img[5:8, 8:20] = 1.0
            img[13:16, 12:20] = 1.0
            img[21:24, 8:20] = 1.0
            img[5:24, 18:21] = 1.0
        elif digit == 4:
            img[5:15, 8:11] = 1.0
            img[14:17, 8:22] = 1.0
            img[5:24, 17:20] = 1.0
        elif digit == 5:
            img[5:8, 8:20] = 1.0
            img[8:14, 8:11] = 1.0
            img[13:16, 8:20] = 1.0
            img[15:21, 17:20] = 1.0
            img[21:24, 8:20] = 1.0
        elif digit == 6:
            img[5:24, 8:11] = 1.0
            img[13:16, 8:20] = 1.0
            img[21:24, 8:20] = 1.0
            img[14:23, 17:20] = 1.0
        elif digit == 7:
            img[5:8, 8:21] = 1.0
            for r in range(8, 24):
                c = int(20 - (r - 8) * 0.7)
                img[r, max(0, c-1):min(28, c+2)] = 1.0
        elif digit == 8:
            rr, cc = np.ogrid[:28, :28]
            mask1 = ((rr - 9)**2 + (cc - 14)**2 <= 25) & ((rr - 9)**2 + (cc - 14)**2 >= 9)
            mask2 = ((rr - 19)**2 + (cc - 14)**2 <= 36) & ((rr - 19)**2 + (cc - 14)**2 >= 12)
            img[mask1 | mask2] = 1.0
        elif digit == 9:
            img[5:8, 8:20] = 1.0
            img[13:16, 8:20] = 1.0
            img[5:14, 8:11] = 1.0
            img[5:24, 17:20] = 1.0

        # Thêm chút nhiễu nhẹ ngẫu nhiên
        noise = np.random.uniform(0.0, 0.1, size=(28, 28))
        img = np.clip(img + noise, 0.0, 1.0)
        images[i] = torch.from_numpy(img.flatten())

    print(f" -> Khởi tạo thành công bộ dữ liệu tổng hợp {num_samples} ảnh 28x28!")
    return images, labels


# ==============================================================================
# PHẦN 3: HUẤN LUYỆN MÔ HÌNH VỚI PYTORCH
# ==============================================================================

def train_autoencoder(model, train_loader, epochs=20, lr=0.001, device="cpu", noise_factor=0.0):
    """
    Hàm huấn luyện Autoencoder tiêu chuẩn bằng PyTorch:
    Hỗ trợ cả chế độ Autoencoder thông thường và Denoising Autoencoder (thêm nhiễu noise_factor).
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "="*70)
    print(f" BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH: {model.__class__.__name__} ({device})")
    print(f" Cấu hình: Epochs={epochs}, LR={lr}, Noise={noise_factor}")
    print("="*70)

    start_time = time.time()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)

            # Thêm nhiễu nếu là Denoising Autoencoder
            if noise_factor > 0.0:
                noisy_x = batch_x + noise_factor * torch.randn_like(batch_x)
                noisy_x = torch.clamp(noisy_x, 0.0, 1.0)
                input_x = noisy_x
            else:
                input_x = batch_x

            optimizer.zero_grad()
            recon_x, _ = model(input_x)
            loss = criterion(recon_x, batch_x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        scheduler.step()
        epoch_loss = total_loss / len(train_loader.dataset)
        history.append(epoch_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:2d}/{epochs:2d}] - MSE Loss: {epoch_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    total_time = time.time() - start_time
    print(f"===> Hoàn thành huấn luyện trong {total_time:.2f}s! MSE Loss cuối cùng: {history[-1]:.6f}")
    return history


# ==============================================================================
# PHẦN 4: SO SÁNH VỚI PCA BASELINE & TRỰC QUAN HÓA KẾT QUẢ
# ==============================================================================

def compare_with_pca(X_train_np, X_test_np, latent_dim=2):
    """
    Chạy thuật toán PCA (Principal Component Analysis) làm đường cơ sở (Baseline)
    để so sánh với Deep Autoencoder theo đúng thí nghiệm trong bài báo Hinton 2006.
    """
    if not HAS_SKLEARN:
        print("[PCA] Không tìm thấy thư viện scikit-learn, bỏ qua bước so sánh PCA.")
        return None, None

    print(f"\n[Baseline] Đang huấn luyện mô hình PCA với {latent_dim} thành phần chính...")
    pca = PCA(n_components=latent_dim)
    pca.fit(X_train_np)

    # Nén và Tái tạo bằng PCA
    test_pca_latents = pca.transform(X_test_np)
    test_pca_recon = pca.inverse_transform(test_pca_latents)

    pca_mse = np.mean((X_test_np - test_pca_recon) ** 2)
    print(f" -> Sai số tái tạo bình phương trung bình của PCA (MSE): {pca_mse:.6f}")
    return test_pca_latents, test_pca_recon


def visualize_and_save_results(model, X_test, y_test, pca_recon=None, output_dir="./outputs", device="cpu"):
    """
    Xuất các biểu đồ trực quan hóa:
    1. So sánh ảnh gốc vs Tái tạo bởi Autoencoder vs Tái tạo bởi PCA.
    2. Biểu đồ phân cụm không gian tiềm ẩn 2D (Latent Space Visualization - Figure 3).
    """
    if not HAS_MATPLOTLIB:
        print("[Visualization] Không tìm thấy thư viện matplotlib, bỏ qua lưu hình ảnh.")
        return

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        test_tensor = X_test[:10].to(device)
        recon_tensor, _ = model(test_tensor)
        orig_imgs = test_tensor.cpu().numpy()
        recon_imgs = recon_tensor.cpu().numpy()

        # Toàn bộ test set để vẽ không gian 2D
        all_test_tensor = X_test.to(device)
        _, all_latents = model(all_test_tensor)
        latents_np = all_latents.cpu().numpy()

    # --------------------------------------------------------------------------
    # BIỂU ĐỒ 1: SO SÁNH TÁI TẠO ẢNH (ORIGINAL VS AUTOENCODER VS PCA)
    # --------------------------------------------------------------------------
    num_cols = 10
    num_rows = 3 if pca_recon is not None else 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 2 * num_rows))
    fig.suptitle("So Sánh Khả Năng Tái Tạo Ảnh: Gốc vs Deep Autoencoder vs PCA", fontsize=14, fontweight="bold")

    for i in range(num_cols):
        # Ảnh gốc
        axes[0, i].imshow(orig_imgs[i].reshape(28, 28), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Ảnh Gốc", fontsize=10, loc="left")

        # Autoencoder tái tạo
        axes[1, i].imshow(recon_imgs[i].reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Autoencoder", fontsize=10, loc="left")

        # PCA tái tạo (nếu có)
        if pca_recon is not None:
            axes[2, i].imshow(pca_recon[i].reshape(28, 28), cmap="gray")
            axes[2, i].axis("off")
            if i == 0:
                axes[2, i].set_title("PCA Baseline", fontsize=10, loc="left")

    plt.tight_layout()
    reconstruction_path = os.path.join(output_dir, "reconstruction_comparison.png")
    plt.savefig(reconstruction_path, dpi=200)
    plt.close()
    print(f" -> Đã lưu biểu đồ so sánh tái tạo tại: {reconstruction_path}")

    # --------------------------------------------------------------------------
    # BIỂU ĐỒ 2: KHÔNG GIAN TIỀM ẨN 2 CHIỀU (TÁI HIỆN HÌNH 3 TRONG BÀI BÁO)
    # --------------------------------------------------------------------------
    if latents_np.shape[1] == 2:
        plt.figure(figsize=(9, 8))
        labels_np = y_test.numpy()
        scatter = plt.scatter(latents_np[:, 0], latents_np[:, 1], c=labels_np, cmap="tab10", alpha=0.75, s=16)
        plt.colorbar(scatter, ticks=range(10), label="Nhãn Chữ Số (0-9)")
        plt.title("Biểu Diễn Không Gian Tiềm Ẩn 2D (Tái Hiện Figure 3 - Hinton 2006)", fontsize=13, fontweight="bold")
        plt.xlabel("Thành phần tiềm ẩn 1 ($z_1$)")
        plt.ylabel("Thành phần tiềm ẩn 2 ($z_2$)")
        plt.grid(True, linestyle="--", alpha=0.4)

        latent_path = os.path.join(output_dir, "latent_space_2d.png")
        plt.savefig(latent_path, dpi=200)
        plt.close()
        print(f" -> Đã lưu biểu đồ không gian tiềm ẩn 2D tại: {latent_path}")


# ==============================================================================
# HÀM MAIN: THỰC THI CHƯƠNG TRÌNH PYTORCH
# ==============================================================================

if __name__ == "__main__":
    print("="*75)
    print(" KHỞI ĐỘNG CHƯƠNG TRÌNH: DEEP AUTOENCODER BẰNG PYTORCH")
    print("="*75)

    # 1. Thiết bị tính toán (GPU CUDA nếu có, ngược lại CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" * Thiết bị tính toán được sử dụng: {device}")

    # 2. Chuẩn bị tập dữ liệu
    all_images, all_labels = get_mnist_or_synthetic_data(num_samples=2000)
    split_idx = int(0.8 * len(all_images))

    train_x, test_x = all_images[:split_idx], all_images[split_idx:]
    train_y, test_y = all_labels[:split_idx], all_labels[split_idx:]

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 3. Khởi tạo mô hình Deep Autoencoder (nén 784 về 2 chiều như Figure 3 của bài báo)
    latent_dimension = 2
    model = DeepAutoencoder(input_dim=784, latent_dim=latent_dimension)
    print(f"\n * Đã khởi tạo DeepAutoencoder với Bottleneck = {latent_dimension} chiều.")

    # 4. Huấn luyện mô hình
    history = train_autoencoder(model, train_loader, epochs=20, lr=0.002, device=device)

    # 5. Đánh giá sai số trên tập kiểm thử
    model.eval()
    with torch.no_grad():
        test_recon, _ = model(test_x.to(device))
        ae_mse = nn.MSELoss()(test_recon, test_x.to(device)).item()
    print(f"\n * Sai số tái tạo bình phương trung bình của Deep Autoencoder (MSE): {ae_mse:.6f}")

    # 6. Chạy mô hình Baseline PCA để đối sánh
    pca_latents, pca_recon = compare_with_pca(train_x.numpy(), test_x.numpy(), latent_dim=latent_dimension)

    # 7. Trực quan hóa và lưu hình ảnh kết quả
    output_directory = os.path.join(os.path.dirname(__file__), "outputs")
    visualize_and_save_results(model, test_x, test_y, pca_recon=pca_recon, output_dir=output_directory, device=device)

    print("\n" + "="*75)
    print(" HOÀN TẤT THỰC THI PIPELINE PYTORCH THÀNH CÔNG!")
    print("="*75)
