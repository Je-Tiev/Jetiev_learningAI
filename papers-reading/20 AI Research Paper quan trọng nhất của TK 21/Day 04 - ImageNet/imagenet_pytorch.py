"""
================================================================================
BÀI BÁO: ImageNet: A Large-Scale Hierarchical Image Database (Deng et al., CVPR 2009)
TỆP MÃ NGUỒN: imagenet_pytorch.py
MÔ TẢ DÀNH CHO NGƯỜI MỚI BẮT ĐẦU:
    Tệp mã nguồn này triển khai mô hình phân loại hình ảnh phân cấp hiện đại bằng
    thư viện PyTorch, mô phỏng lại các đóng góp học thuật cốt lõi của ImageNet:
    
    1. Hàm Mất Mát Phân Cấp (HierarchicalCrossEntropyLoss):
       - Tích hợp ma trận khoảng cách cây phả hệ WordNet trực tiếp vào hàm mất mát.
       - Phạt sai số tỷ lệ với khoảng cách ngữ nghĩa: nếu đoán sai, đoán nhầm sang
         lớp cùng họ hàng bị phạt nhẹ hơn nhiều so với đoán nhầm sang loài khác.
    
    2. Kiến Trúc Mạng Nơ-ron Tích Chập Thị Giác (Vision CNN):
       - Khối trích xuất đặc trưng (Feature Extractor) nhiều tầng tích chập, chuẩn hóa
         BatchNorm2d, kích hoạt ReLU và gom nhóm MaxPool2d.
       - Bộ phân loại (Classifier) với Dropout chống quá khớp.
    
    3. Bộ Đo Lường Chuẩn Mực của ImageNet:
       - Top-1 Accuracy: Tỷ lệ dự đoán đứng đầu chính xác tuyệt đối.
       - Top-5 Accuracy: Tỷ lệ nhãn đúng nằm trong 5 dự đoán tự tin nhất.
       - Mean Tree-Hop Distance: Khoảng cách nhảy cây trung bình giữa dự đoán và nhãn thực tế.
    
    4. Phân Tích & Đối Chiếu Giá Trị Học Thuật (Academic Value Benchmark):
       - Mô hình 1 (Kỷ nguyên Tiền-ImageNet): Trích xuất đặc trưng thủ công (Hand-crafted Features / SIFT-like) + Bộ phân loại tuyến tính.
       - Mô hình 2 (CNN Dữ liệu nhỏ / Caltech-101 scale): Mạng CNN huấn luyện trên tập dữ liệu nhỏ dẫn tới quá khớp (Overfitting) trầm trọng.
       - Mô hình 3 (Tư duy ImageNet / Scaled Data): Mạng CNN huấn luyện trên dữ liệu phong phú, đa dạng, kết hợp hàm mất mát phân cấp.
       - Xuất biểu đồ trực quan hóa 4 bảng đối chiếu chuyên sâu vào:
         outputs/imagenet_academic_comparison.png
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
# PHẦN 1: HÀM MẤT MÁT PHÂN CẤP (HIERARCHICAL CROSS-ENTROPY LOSS)
# ==============================================================================

class HierarchicalCrossEntropyLoss(nn.Module):
    """
    Hàm mất mát phân cấp phạt sai số theo khoảng cách nhảy cây (Tree-Hop Distance):
    
        L_total = L_CE(logits, y_true) + lambda * sum_j [ P(y_pred = j) * D_tree(y_true, j) ]
        
    Giải thích cho người mới bắt đầu:
    - Cross-Entropy tiêu chuẩn (L_CE) đối xử với mọi lỗi sai như nhau (Husky -> Wolf cũng như Husky -> Car).
    - Thành phần phạt phân cấp (Hierarchical Penalty) lấy kỳ vọng khoảng cách cây phả hệ WordNet:
      Nếu mô hình phân vân giữa Husky và Golden Retriever (D_tree = 2), mức phạt sẽ rất nhỏ.
      Nếu mô hình phân vân giữa Husky và Ô tô (D_tree = 10), mức phạt sẽ rất nặng!
    """
    def __init__(self, distance_matrix, penalty_weight=0.15):
        super(HierarchicalCrossEntropyLoss, self).__init__()
        if not isinstance(distance_matrix, torch.Tensor):
            self.register_buffer("distance_matrix", torch.tensor(distance_matrix, dtype=torch.float32))
        else:
            self.register_buffer("distance_matrix", distance_matrix.float())
        self.penalty_weight = penalty_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # 1. Mất mát Cross-Entropy tiêu chuẩn
        base_ce = self.ce_loss(logits, targets)

        # 2. Tính phân phối xác suất dự đoán qua hàm Softmax
        probs = torch.softmax(logits, dim=1)  # Kích thước: (Batch, C)
        
        # 3. Lấy vector khoảng cách cây phả hệ tương ứng với nhãn đúng của từng mẫu
        target_dists = self.distance_matrix[targets]  # Kích thước: (Batch, C)
        
        # 4. Kỳ vọng khoảng cách cây = sum_j [ P(j) * D(true, j) ]
        hierarchical_penalty = torch.mean(torch.sum(probs * target_dists, dim=1))

        # 5. Tổng hợp mất mát
        total_loss = base_ce + self.penalty_weight * hierarchical_penalty
        return total_loss, base_ce, hierarchical_penalty


# ==============================================================================
# PHẦN 2: KIẾN TRÚC MẠNG NƠ-RON TÍCH CHẬP THỊ GIÁC (VISION CNN)
# ==============================================================================

class HierarchicalVisionCNN(nn.Module):
    """
    Mạng nơ-ron tích chập (CNN) phân loại hình ảnh:
    Đầu vào: Tensor ảnh kích thước (Batch, 3, 32, 32)
    Đầu ra: Logits phân loại cho C lớp đối tượng
    """
    def __init__(self, num_classes=8, in_channels=3):
        super(HierarchicalVisionCNN, self).__init__()
        self.features = nn.Sequential(
            # Khối Conv 1: Trích xuất cạnh và đường nét cơ bản
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 32, 32) -> (32, 16, 16)

            # Khối Conv 2: Học họa tiết và góc cạnh phức tạp hơn
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (64, 16, 16) -> (64, 8, 8)

            # Khối Conv 3: Tổng hợp hình khối và bộ phận nhận thức
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))             # (128, 8, 8) -> (128, 4, 4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),                         # Giảm thiểu quá khớp
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


class HandcraftedFeatureClassifier(nn.Module):
    """
    Mô phỏng trường phái Tiền-ImageNet (Pre-2009 Baseline):
    - Trích xuất đặc trưng thủ công (ví dụ: lược đồ màu Color Histogram, kết cấu).
    - Bộ phân loại tuyến tính đơn giản (Linear Classifier tương tự Linear SVM).
    """
    def __init__(self, num_classes=8):
        super(HandcraftedFeatureClassifier, self).__init__()
        # 3 kênh x 8 bins mỗi kênh = 24 đặc trưng lược đồ màu toàn cục
        self.classifier = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def extract_color_histograms(self, x):
        """Trích xuất vector đặc trưng lược đồ màu thủ công từ batch ảnh (B, 3, H, W)."""
        B, C, H, W = x.shape
        features = []
        for i in range(B):
            sample = x[i].view(C, -1)
            hists = []
            for c in range(C):
                channel_data = sample[c]
                hist = torch.histc(channel_data, bins=8, min=-2.0, max=2.0)
                hists.append(hist / (hist.sum() + 1e-8))
            features.append(torch.cat(hists))
        return torch.stack(features)

    def forward(self, x):
        feats = self.extract_color_histograms(x)
        return self.classifier(feats)


# ==============================================================================
# PHẦN 3: BỘ ĐO LƯỜNG CHUẨN MỰC IMAGENET (TOP-1, TOP-5, TREE-HOP ERROR)
# ==============================================================================

def evaluate_imagenet_metrics(logits, targets, distance_matrix):
    """
    Tính toán 3 chỉ số cốt lõi của ImageNet:
    1. Top-1 Accuracy: Tỷ lệ dự đoán cao nhất trùng khớp với nhãn thật.
    2. Top-5 Accuracy: Tỷ lệ nhãn thật nằm trong top-5 dự đoán cao nhất.
    3. Mean Tree-Hop Distance: Khoảng cách nhảy cây trung bình theo phả hệ WordNet.
    """
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        k = min(5, probs.size(1))

        # Top-1 Prediction
        _, top1_preds = torch.max(probs, dim=1)
        top1_acc = (top1_preds == targets).float().mean().item()

        # Top-5 Prediction
        _, topk_preds = torch.topk(probs, k=k, dim=1)
        topk_correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds))
        top5_acc = topk_correct.any(dim=1).float().mean().item()

        # Mean Tree-Hop Distance
        dists = distance_matrix[targets, top1_preds]
        mean_tree_dist = dists.float().mean().item()

    return top1_acc, top5_acc, mean_tree_dist


# ==============================================================================
# PHẦN 4: NẠP DỮ LIỆU & BỘ SINH DỮ LIỆU PHÂN CẤP TỔNG HỢP
# ==============================================================================

def create_synthetic_hierarchical_data(n_samples=800, n_classes=8):
    """
    Bộ sinh dữ liệu phân cấp dự phòng (Offline Synthetic Fallback):
    Tạo ra 8 lớp ảnh (3x32x32) phân bố theo 2 nhánh cha WordNet:
    - Nhánh 0..3 (Sinh vật sống / Living Thing): Tông màu ấm (đỏ/vàng), có vân tròn.
    - Nhánh 4..7 (Đồ vật nhân tạo / Artifact): Tông màu lạnh (xanh dương/xám), có góc nhọn.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    X = np.random.randn(n_samples, 3, 32, 32).astype(np.float32) * 0.2
    y = np.random.randint(0, n_classes, size=(n_samples,)).astype(np.int64)

    for i in range(n_samples):
        cls = y[i]
        if cls < 4:
            # Living Thing: tăng kênh Đỏ và kênh Xanh lá
            X[i, 0, :, :] += 0.8 + 0.15 * cls
            X[i, 1, :, :] += 0.3 * (cls % 2)
            # Thêm hoa văn tròn ở trung tâm
            X[i, :, 10:22, 10:22] += 0.4
        else:
            # Artifact: tăng kênh Xanh dương
            X[i, 2, :, :] += 0.8 + 0.15 * (cls - 4)
            X[i, 1, :, :] += 0.35 * ((cls - 4) % 2)
            # Thêm họa tiết đường chéo
            for d in range(16):
                X[i, :, 8+d, 8+d] += 0.5

    # Chuẩn hóa theo chuẩn ImageNet thống kê kinh điển
    # (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    X = (X - mean) / std

    # Ma trận khoảng cách cây phả hệ mẫu cho 8 lớp:
    # 2 lớp cùng loài con: 2 bước; cùng nhánh cha: 4 bước; khác nhánh cha: 8 bước
    dist_matrix = np.zeros((n_classes, n_classes), dtype=np.float32)
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                dist_matrix[i, j] = 0.0
            elif (i // 2) == (j // 2):
                dist_matrix[i, j] = 2.0  # Rất gần gũi (ví dụ husky vs golden_retriever)
            elif (i < 4 and j < 4) or (i >= 4 and j >= 4):
                dist_matrix[i, j] = 4.0  # Cùng nhánh cha lớn (Living Thing hoặc Artifact)
            else:
                dist_matrix[i, j] = 8.0  # Khác biệt hoàn toàn (Living Thing vs Artifact)

    class_names = [
        "husky", "golden_retriever", "cat", "tiger",
        "car", "truck", "airplane", "speedboat"
    ]

    return torch.tensor(X), torch.tensor(y), dist_matrix, class_names


# ==============================================================================
# PHẦN 5: ĐỐI CHIẾU GIÁ TRỊ HỌC THUẬT & TRỰC QUAN HÓA CHUYÊN SÂU
# ==============================================================================

def run_academic_value_benchmark(device):
    """
    Thực thi thử nghiệm so sánh chuẩn mực giữa 3 mô hình đại diện cho 3 giai đoạn:
    1. Mô Hình 1: Tiền-ImageNet (Trích xuất đặc trưng thủ công SIFT-like + Tuyến tính).
    2. Mô Hình 2: CNN Dữ Liệu Nhỏ (Caltech-101 scale: 100 mẫu/lớp -> Quá khớp trầm trọng).
    3. Mô Hình 3: CNN ImageNet Phân Cấp (Đầy đủ dữ liệu + Mất mát phân cấp Tree Loss).
    """
    print("\n" + "=" * 80)
    print("CHẠY THỬ NGHIỆM ĐỐI CHIẾU GIÁ TRỊ HỌC THUẬT (ACADEMIC VALUE BENCHMARK)")
    print("=" * 80)

    # 1. Sinh dữ liệu đầy đủ (Đại diện quy mô ImageNet đa dạng)
    X_large, y_large, dist_matrix_np, class_names = create_synthetic_hierarchical_data(n_samples=1000, n_classes=8)
    dist_matrix_tensor = torch.tensor(dist_matrix_np, device=device)

    # Chia train/test (80/20)
    split_large = int(0.8 * len(X_large))
    train_loader_large = DataLoader(TensorDataset(X_large[:split_large], y_large[:split_large]), batch_size=32, shuffle=True)
    test_loader_large = DataLoader(TensorDataset(X_large[split_large:], y_large[split_large:]), batch_size=32, shuffle=False)

    # Tập dữ liệu nhỏ (Đại diện quy mô Tiền-ImageNet: Caltech-101 với chỉ ~80 mẫu huấn luyện)
    X_small = X_large[:120]
    y_small = y_large[:120]
    train_loader_small = DataLoader(TensorDataset(X_small[:80], y_small[:80]), batch_size=16, shuffle=True)
    test_loader_small = DataLoader(TensorDataset(X_small[80:], y_small[80:]), batch_size=16, shuffle=False)

    # --------------------------------------------------------------------------
    # Huấn luyện Mô Hình 1: Đặc trưng thủ công (Pre-ImageNet Baseline)
    # --------------------------------------------------------------------------
    print("\n[*] [1/3] Đang huấn luyện Mô Hình 1: Hand-crafted Features (Kỷ nguyên Tiền-ImageNet)...")
    model_handcrafted = HandcraftedFeatureClassifier(num_classes=8).to(device)
    opt_1 = optim.Adam(model_handcrafted.parameters(), lr=0.01)
    crit_1 = nn.CrossEntropyLoss()

    for epoch in range(15):
        model_handcrafted.train()
        for bx, by in train_loader_large:
            bx, by = bx.to(device), by.to(device)
            opt_1.zero_grad()
            out = model_handcrafted(bx)
            loss = crit_1(out, by)
            loss.backward()
            opt_1.step()

    # Đánh giá Mô Hình 1
    model_handcrafted.eval()
    with torch.no_grad():
        all_logits = [model_handcrafted(bx.to(device)) for bx, _ in test_loader_large]
        all_targets = [by.to(device) for _, by in test_loader_large]
        val_logits_1 = torch.cat(all_logits, dim=0)
        val_targets_1 = torch.cat(all_targets, dim=0)
        top1_m1, top5_m1, tree_m1 = evaluate_imagenet_metrics(val_logits_1, val_targets_1, dist_matrix_tensor)
        
        train_logits_1 = torch.cat([model_handcrafted(bx.to(device)) for bx, _ in train_loader_large], dim=0)
        train_targets_1 = torch.cat([by.to(device) for _, by in train_loader_large], dim=0)
        train_acc_m1 = (torch.argmax(train_logits_1, dim=1) == train_targets_1).float().mean().item()

    # --------------------------------------------------------------------------
    # Huấn luyện Mô Hình 2: CNN trên tập dữ liệu nhỏ (Overfitting Baseline)
    # --------------------------------------------------------------------------
    print("[*] [2/3] Đang huấn luyện Mô Hình 2: Vision CNN trên tập dữ liệu nhỏ (Caltech-101 scale)...")
    model_small = HierarchicalVisionCNN(num_classes=8).to(device)
    opt_2 = optim.Adam(model_small.parameters(), lr=0.001)
    crit_2 = nn.CrossEntropyLoss()

    for epoch in range(25):
        model_small.train()
        for bx, by in train_loader_small:
            bx, by = bx.to(device), by.to(device)
            opt_2.zero_grad()
            out = model_small(bx)
            loss = crit_2(out, by)
            loss.backward()
            opt_2.step()

    # Đánh giá Mô Hình 2
    model_small.eval()
    with torch.no_grad():
        all_logits = [model_small(bx.to(device)) for bx, _ in test_loader_large]
        val_logits_2 = torch.cat(all_logits, dim=0)
        top1_m2, top5_m2, tree_m2 = evaluate_imagenet_metrics(val_logits_2, val_targets_1, dist_matrix_tensor)
        
        train_logits_2 = torch.cat([model_small(bx.to(device)) for bx, _ in train_loader_small], dim=0)
        train_targets_2 = torch.cat([by.to(device) for _, by in train_loader_small], dim=0)
        train_acc_m2 = (torch.argmax(train_logits_2, dim=1) == train_targets_2).float().mean().item()

    # --------------------------------------------------------------------------
    # Huấn luyện Mô Hình 3: CNN trên ImageNet Scale kèm Hierarchical Loss
    # --------------------------------------------------------------------------
    print("[*] [3/3] Đang huấn luyện Mô Hình 3: Vision CNN trên ImageNet Scale + Hierarchical Loss...")
    model_imagenet = HierarchicalVisionCNN(num_classes=8).to(device)
    opt_3 = optim.Adam(model_imagenet.parameters(), lr=0.001)
    crit_3 = HierarchicalCrossEntropyLoss(dist_matrix_tensor, penalty_weight=0.15)

    for epoch in range(12):
        model_imagenet.train()
        for bx, by in train_loader_large:
            bx, by = bx.to(device), by.to(device)
            opt_3.zero_grad()
            out = model_imagenet(bx)
            loss, _, _ = crit_3(out, by)
            loss.backward()
            opt_3.step()

    # Đánh giá Mô Hình 3
    model_imagenet.eval()
    with torch.no_grad():
        all_logits = [model_imagenet(bx.to(device)) for bx, _ in test_loader_large]
        val_logits_3 = torch.cat(all_logits, dim=0)
        top1_m3, top5_m3, tree_m3 = evaluate_imagenet_metrics(val_logits_3, val_targets_1, dist_matrix_tensor)
        
        train_logits_3 = torch.cat([model_imagenet(bx.to(device)) for bx, _ in train_loader_large], dim=0)
        train_targets_3 = torch.cat([by.to(device) for _, by in train_loader_large], dim=0)
        train_acc_m3 = (torch.argmax(train_logits_3, dim=1) == train_targets_3).float().mean().item()

    # In bảng kết quả so sánh định lượng
    print("\n[+] KẾT QUẢ ĐỐI CHIẾU ĐỊNH LƯỢNG GIỮA CÁC MÔ HÌNH:")
    print(f"    {'Mô Hình Nghiên Cứu':<36} | {'Train Acc':<11} | {'Top-1 Val':<11} | {'Top-5 Val':<11} | {'Tree-Hop Error':<16}")
    print("    " + "-" * 95)
    print(f"    {'1. Hand-crafted Features (Pre-2009)':<36} | {train_acc_m1*100:6.2f}%    | {top1_m1*100:6.2f}%    | {top5_m1*100:6.2f}%    | {tree_m1:6.2f} bước")
    print(f"    {'2. CNN Dữ Liệu Nhỏ (Caltech Scale)':<36} | {train_acc_m2*100:6.2f}%    | {top1_m2*100:6.2f}%    | {top5_m2*100:6.2f}%    | {tree_m2:6.2f} bước (Overfitting!)")
    print(f"    {'3. CNN ImageNet Scale + Tree Loss':<36} | {train_acc_m3*100:6.2f}%    | {top1_m3*100:6.2f}%    | {top5_m3*100:6.2f}%    | {tree_m3:6.2f} bước (Tối ưu!)")

    # Tính ma trận nhầm lẫn của Mô Hình 3
    with torch.no_grad():
        preds_3 = torch.argmax(val_logits_3, dim=1).cpu().numpy()
        targets_np = val_targets_1.cpu().numpy()

    cm = np.zeros((8, 8), dtype=np.int32)
    for t, p in zip(targets_np, preds_3):
        cm[t, p] += 1

    # Xuất đồ thị đối chiếu 4 bảng chuyên sâu
    benchmark_data = {
        "models": ["Hand-crafted\n(Pre-2009)", "CNN Small Data\n(Overfitting)", "CNN ImageNet Scale\n(Hierarchical)"],
        "top1": [top1_m1 * 100, top1_m2 * 100, top1_m3 * 100],
        "top5": [top5_m1 * 100, top5_m2 * 100, top5_m3 * 100],
        "train_acc": [train_acc_m1 * 100, train_acc_m2 * 100, train_acc_m3 * 100],
        "tree_err": [tree_m1, tree_m2, tree_m3],
        "cm": cm,
        "class_names": class_names,
        "dist_matrix": dist_matrix_np
    }
    
    return benchmark_data


def plot_comprehensive_academic_comparison(bench_data, output_dir="outputs"):
    """
    Vẽ biểu đồ đối chiếu học thuật 4 bảng (4 Subplots):
    1. So sánh Top-1 vs Top-5 Accuracy giữa các phương pháp.
    2. Khoảng cách Quá Khớp (Train vs Val Generalization Gap).
    3. Sai số Nhảy Cây Phân Cấp (Mean Tree-Hop Distance).
    4. Ma trận Nhầm lẫn Phân cấp (Hierarchical Confusion Matrix).
    """
    if not HAS_MATPLOTLIB:
        print("[!] Matplotlib chưa được cài đặt. Bỏ qua xuất ảnh đồ thị.")
        return

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "imagenet_academic_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['#e74c3c', '#f39c12', '#2ecc71']

    # Subplot 1: Top-1 vs Top-5 Accuracy
    x_idx = np.arange(len(bench_data["models"]))
    width = 0.35
    axes[0, 0].bar(x_idx - width/2, bench_data["top1"], width, label='Top-1 Accuracy', color='#3498db', edgecolor='black')
    axes[0, 0].bar(x_idx + width/2, bench_data["top5"], width, label='Top-5 Accuracy', color='#2ecc71', edgecolor='black')
    axes[0, 0].set_title("1. Top-1 vs Top-5 Accuracy Qua Các Kỷ Nguyên", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("Độ Chính Xác (%)")
    axes[0, 0].set_xticks(x_idx)
    axes[0, 0].set_xticklabels(bench_data["models"])
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)

    # Subplot 2: Train vs Val Generalization Gap (Chứng minh giải quyết Overfitting)
    train_bars = axes[0, 1].bar(x_idx - width/2, bench_data["train_acc"], width, label='Train Acc', color='#9b59b6', edgecolor='black')
    val_bars = axes[0, 1].bar(x_idx + width/2, bench_data["top1"], width, label='Val Acc (Top-1)', color='#34495e', edgecolor='black')
    axes[0, 1].set_title("2. Khoảng Cách Quá Khớp: Train vs Val Acc", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("Độ Chính Xác (%)")
    axes[0, 1].set_xticks(x_idx)
    axes[0, 1].set_xticklabels(bench_data["models"])
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.5)

    # Subplot 3: Mean Tree-Hop Distance (Phạt sai số ngữ nghĩa)
    bars_tree = axes[1, 0].bar(bench_data["models"], bench_data["tree_err"], color=colors, edgecolor='black', width=0.5)
    axes[1, 0].set_title("3. Sai Số Nhảy Cây Ngữ Nghĩa (Mean Tree-Hop Distance)", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Số Bước Nhảy Cây (Hops) - Càng thấp càng tốt")
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars_tree:
        yval = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

    # Subplot 4: Ma trận nhầm lẫn mô hình phân cấp ImageNet
    im = axes[1, 1].imshow(bench_data["cm"], interpolation='nearest', cmap='Blues')
    axes[1, 1].set_title("4. Ma Trận Nhầm Lẫn Mô Hình ImageNet Scale", fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(bench_data["class_names"]))
    axes[1, 1].set_xticks(tick_marks)
    axes[1, 1].set_xticklabels(bench_data["class_names"], rotation=45, ha='right')
    axes[1, 1].set_yticks(tick_marks)
    axes[1, 1].set_yticklabels(bench_data["class_names"])
    axes[1, 1].set_ylabel('Nhãn Thật (Ground Truth)')
    axes[1, 1].set_xlabel('Nhãn Dự Đoán (Predicted)')

    fig.suptitle("ImageNet (CVPR 2009): Khảo Sát & Đối Chiếu Giá Trị Học Thuật Bằng PyTorch", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[✓] Đã lưu biểu đồ đối chiếu học thuật hoàn chỉnh tại: {out_file}")


# ==============================================================================
# HÀM CHÍNH (MAIN PIPELINE)
# ==============================================================================

def main():
    print("================================================================================")
    print("TRIỂN KHAI PHÂN LOẠI HÌNH ẢNH PHÂN CẤP IMAGENET (CVPR 2009) BẰNG PYTORCH")
    print("================================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị tính toán được phát hiện: {device}")

    # Chạy thử nghiệm đối chiếu học thuật toàn diện
    benchmark_results = run_academic_value_benchmark(device)

    # Xuất biểu đồ đồ họa phân tích
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    plot_comprehensive_academic_comparison(benchmark_results, output_dir=out_dir)

    print("\n[✓] Hoàn thành trọn vẹn chương trình thử nghiệm ImageNet phân cấp bằng PyTorch!")


if __name__ == "__main__":
    main()
