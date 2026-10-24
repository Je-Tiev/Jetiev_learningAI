"""
================================================================================
BÀI BÁO: ImageNet Classification with Deep Convolutional Neural Networks (Krizhevsky et al., NeurIPS 2012)
TỆP MÃ NGUỒN: alexnet_pytorch.py
MÔ TẢ: Triển khai kiến trúc AlexNet kinh điển và tối ưu hóa hiện đại bằng PyTorch:
       1. Định nghĩa chuẩn xác kiến trúc AlexNet gốc (Conv1-5, LRN, Overlapping MaxPool, FC6-8, Dropout).
       2. Tùy chọn chuyển đổi linh hoạt giữa chuẩn hóa LRN cổ điển và Batch Normalization hiện đại.
       3. Bộ nạp dữ liệu hình ảnh thông minh kèm Fallback an toàn khi chạy offline.
       4. Trực quan hóa 96 bộ lọc tích chập tầng Conv1 (tái hiện Figure 3 lịch sử của bài báo).
       5. Huấn luyện SGD với Momentum và xuất biểu đồ đánh giá vào thư mục outputs/.
================================================================================
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ==============================================================================
# PHẦN 1: ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH ALEXNET BẰNG PYTORCH
# ==============================================================================

class AlexNet(nn.Module):
    """
    Kiến trúc mạng nơ-ron tích chập sâu AlexNet kinh điển (Krizhevsky et al., 2012):
    - 5 tầng Convolutional + 3 tầng Fully Connected.
    - LocalResponseNorm (LRN) theo công thức gốc (size=5, alpha=1e-4, beta=0.75, k=2).
    - Overlapping Max Pooling (size=3, stride=2).
    - Dropout p=0.5 ở FC6 và FC7.
    """
    def __init__(self, num_classes=10, use_lrn=True, in_channels=3):
        super(AlexNet, self).__init__()
        self.use_lrn = use_lrn

        norm_layer1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0) if use_lrn else nn.BatchNorm2d(96)
        norm_layer2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0) if use_lrn else nn.BatchNorm2d(256)

        self.features = nn.Sequential(
            # Tầng Conv 1: 96 filters 11x11, stride 4
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            norm_layer1,
            nn.MaxPool2d(kernel_size=3, stride=2),  # Overlapping Pooling

            # Tầng Conv 2: 256 filters 5x5, pad 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            norm_layer2,
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Tầng Conv 3: 384 filters 3x3, pad 1
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Tầng Conv 4: 384 filters 3x3, pad 1
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Tầng Conv 5: 256 filters 3x3, pad 1
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Thích ứng kích thước không gian ra chuẩn 6x6
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        feat_flat = torch.flatten(feat, 1)
        logits = self.classifier(feat_flat)
        return logits


# ==============================================================================
# PHẦN 2: BỘ NẠP DỮ LIỆU & FALLBACK OFFLINE
# ==============================================================================

def load_data_with_fallback(n_samples=200, img_size=64, num_classes=10):
    """
    Nạp dữ liệu huấn luyện kiểm thử:
    Thử tải tập dữ liệu CIFAR-10 chuẩn, nếu offline thì tự sinh tensor ảnh RGB phân cụm nhân tạo.
    """
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        # Lấy subset nhỏ để demo nhanh
        indices = list(range(n_samples))
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=32, shuffle=True)
        print(f"[*] Đã nạp thành công bộ dữ liệu CIFAR-10 cục bộ ({n_samples} mẫu).")
        return loader, "CIFAR-10"
    except Exception as e:
        print(f"[!] Không thể nạp CIFAR-10 ({e}). Kích hoạt bộ sinh ảnh Fallback Synthetic Tensors...")
        torch.manual_seed(42)
        X = torch.randn(n_samples, 3, img_size, img_size) * 0.5
        y = torch.randint(0, num_classes, (n_samples,))

        # Thêm mẫu đặc trưng khác biệt theo từng lớp
        for i in range(n_samples):
            cls = y[i].item()
            X[i, cls % 3, 10:25, 10:25] += 1.5

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        print(f"[*] Đã sinh dữ liệu Fallback an toàn: {n_samples} ảnh RGB ({img_size}x{img_size}), {num_classes} lớp.")
        return loader, "Synthetic RGB Images (Fallback)"


# ==============================================================================
# PHẦN 3: TRỰC QUAN HÓA BỘ LỌC CONV1 & HUẤN LUYỆN (VISUALIZATION)
# ==============================================================================

def plot_conv1_filters(model, output_dir="outputs"):
    """
    Tái hiện lại Figure 3 lịch sử của bài báo AlexNet:
    Trích xuất và vẽ lưới 96 bộ lọc của tầng Conv1 (11x11x3) thể hiện các bộ dò cạnh và màu sắc.
    """
    if not HAS_MATPLOTLIB:
        print("[!] Matplotlib chưa cài đặt. Bỏ qua xuất ảnh.")
        return

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "alexnet_filters.png")

    # Lấy trọng số Conv1: (96, 3, 11, 11)
    conv1_weights = model.features[0].weight.data.clone().cpu().numpy()
    n_filters = conv1_weights.shape[0]  # 96

    # Chuẩn hóa từng bộ lọc về khoảng [0, 1] để hiển thị ảnh RGB
    fig, axes = plt.subplots(8, 12, figsize=(14, 9))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(n_filters):
        row = i // 12
        col = i % 12
        ax = axes[row, col]
        filt = conv1_weights[i].transpose(1, 2, 0)  # (11, 11, 3)
        # Min-max scale
        filt_min, filt_max = filt.min(), filt.max()
        filt_norm = (filt - filt_min) / (filt_max - filt_min + 1e-9)
        ax.imshow(filt_norm)
        ax.axis('off')

    fig.suptitle("96 Bộ Lọc Tích Chập Tầng Conv1 (11x11x3) Của AlexNet\n(Tái Hiện Figure 3 - Krizhevsky et al., 2012)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[✓] Đã lưu biểu đồ trực quan hóa 96 bộ lọc Conv1 thành công tại: {out_file}")


def plot_training_curve(losses, accuracies, output_dir="outputs"):
    """Vẽ và lưu đồ thị suy giảm mất mát và độ chính xác trong quá trình huấn luyện"""
    if not HAS_MATPLOTLIB:
        return

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "alexnet_training.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(losses, color='crimson', lw=2, marker='o')
    ax1.set_title("Đường Cong Mất Mát (Cross-Entropy Loss)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(accuracies, color='teal', lw=2, marker='s')
    ax2.set_title("Độ Chính Xác Huấn Luyện (Accuracy %)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Tiến Trình Huấn Luyện Mạng AlexNet (SGD Momentum = 0.9)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[✓] Đã lưu đồ thị huấn luyện thành công tại: {out_file}")


# ==============================================================================
# HÀM CHÍNH (MAIN PIPELINE)
# ==============================================================================

def main():
    print("================================================================================")
    print("TRIỂN KHAI KIẾN TRÚC MẠNG NƠ-RON SÂU ALEXNET BẰNG PYTORCH")
    print("================================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị tính toán: {device}")

    num_classes = 10
    model = AlexNet(num_classes=num_classes, use_lrn=True).to(device)

    # Đếm số lượng tham số
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv_params = sum(p.numel() for p in model.features.parameters() if p.requires_grad)
    fc_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)

    print(f"[*] Thống kê kiến trúc AlexNet:")
    print(f"    - Tổng số tham số:        {total_params:,}")
    print(f"    - Tham số tầng Conv (1-5): {conv_params:,} ({conv_params/total_params*100:.1f}%)")
    print(f"    - Tham số tầng FC (6-8):   {fc_params:,} ({fc_params/total_params*100:.1f}%)")

    # 1. Nạp dữ liệu kiểm thử
    train_loader, dataset_name = load_data_with_fallback(n_samples=256, img_size=64, num_classes=num_classes)

    # 2. Thiết lập tối ưu hóa SGD với Momentum & Weight Decay chuẩn AlexNet
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    print(f"\n[*] Bắt đầu huấn luyện mô hình AlexNet trên tập dữ liệu {dataset_name}...")
    epochs = 6
    loss_history = []
    acc_history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        print(f"  > Epoch {epoch:2d}/{epochs} | Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc:5.1f}%")

    elapsed = time.time() - start_time
    print(f"[*] Huấn luyện hoàn thành trong {elapsed:.2f} giây.")

    # 3. Xuất biểu đồ bộ lọc Conv1 và đường cong huấn luyện
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    plot_conv1_filters(model, output_dir=out_dir)
    plot_training_curve(loss_history, acc_history, output_dir=out_dir)

    print("\n[✓] Toàn bộ quy trình kiểm chứng và trực quan hóa AlexNet bằng PyTorch đã hoàn tất xuất sắc!")


if __name__ == "__main__":
    main()
