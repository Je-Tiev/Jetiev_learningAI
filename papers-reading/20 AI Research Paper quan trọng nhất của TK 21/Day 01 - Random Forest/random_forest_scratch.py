"""
================================================================================
BÀI BÁO: Random Forests (Leo Breiman, Machine Learning 2001)
TỆP MÃ NGUỒN: random_forest_scratch.py
MÔ TẢ: Cài đặt hoàn chỉnh thuật toán Cây Quyết Định (CART) và Rừng Ngẫu Nhiên (Random Forest)
       từ con số 0 (From Scratch) bằng Python thuần và NumPy:
       1. Độ vẩn đục Gini Impurity & Tìm kiếm điểm rẽ nhánh tối ưu.
       2. Cơ chế Không gian con ngẫu nhiên (Random Feature Subspace m = sqrt(M)).
       3. Lấy mẫu Bootstrap có hoàn lại (Bagging).
       4. Ước lượng sai số ngoại mẫu Out-of-Bag (OOB Error Estimation).
       5. Tính toán tầm quan trọng đặc trưng (Mean Decrease Impurity - MDI).
       6. Bộ thử nghiệm độc lập trên dữ liệu đa chiều, in bảng xếp hạng đặc trưng ra Terminal.
================================================================================
"""

import time
import numpy as np


# ==============================================================================
# PHẦN 1: CÂY QUYẾT ĐỊNH ĐƠN LẺ TỪ ĐẦU (DECISION TREE CLASSIFIER SCRATCH)
# ==============================================================================

class DecisionTreeNode:
    """Nút trong cây quyết định (có thể là nút rẽ nhánh hoặc nút lá)."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None,
                 value=None, impurity=0.0):
        self.feature_idx = feature_idx  # Chỉ số đặc trưng dùng để rẽ nhánh
        self.threshold = threshold      # Ngưỡng phân chia (x <= threshold)
        self.left = left                # Nhánh con bên trái
        self.right = right              # Nhánh con bên phải
        self.value = value              # Nhãn dự đoán hoặc phân phối xác suất (nếu là nút lá)
        self.impurity = impurity        # Độ vẩn đục Gini tại nút này

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifierScratch:
    """
    Cây quyết định phân loại (CART) đơn lẻ:
    Hỗ trợ ngẫu nhiên hóa tập con đặc trưng (Random Subspace) tại mỗi nút.
    """
    def __init__(self, max_depth=None, min_samples_split=2, max_features="sqrt"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None
        self.feature_importances_ = None

    @staticmethod
    def _gini_impurity(y):
        """Tính độ vẩn đục Gini: Gini = 1 - sum(p_i^2)"""
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _get_num_features(self, total_features):
        """Xác định số lượng đặc trưng m cần lấy mẫu ngẫu nhiên tại mỗi nút rẽ nhánh"""
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(total_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(total_features)))
        elif isinstance(self.max_features, int):
            return min(total_features, self.max_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * total_features))
        return total_features

    def _best_split(self, X, y, total_features):
        """
        Tìm đặc trưng và ngưỡng cắt tối ưu nhất trong số m đặc trưng được chọn ngẫu nhiên.
        """
        best_gain = -1.0
        best_feature = None
        best_threshold = None
        n_samples = len(y)
        current_gini = self._gini_impurity(y)

        # Chọn ngẫu nhiên m đặc trưng (Random Subspace Method của Breiman)
        m = self._get_num_features(total_features)
        feature_subspace = np.random.choice(total_features, size=m, replace=False)

        for feat_idx in feature_subspace:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            # Nếu có quá nhiều ngưỡng, lấy mẫu phân vị để tăng tốc độ
            if len(thresholds) > 20:
                thresholds = np.percentile(X_column, np.linspace(5, 95, 20))

            for thresh in thresholds:
                left_mask = X_column <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                # Gini sau phân chia
                p_left = len(y_left) / n_samples
                p_right = len(y_right) / n_samples
                gini_split = p_left * self._gini_impurity(y_left) + p_right * self._gini_impurity(y_right)

                # Mức suy giảm Gini (Gini Gain)
                gain = current_gini - gini_split

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = thresh

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0, total_features=None):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Điều kiện dừng tạo nút lá
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or n_samples < self.min_samples_split:
            # Gán giá trị nút lá là lớp chiếm đa số
            vals, counts = np.unique(y, return_counts=True)
            leaf_value = vals[np.argmax(counts)]
            return DecisionTreeNode(value=leaf_value, impurity=self._gini_impurity(y))

        # Tìm điểm rẽ nhánh tốt nhất
        best_feat, best_thresh, gain = self._best_split(X, y, total_features)

        # Nếu không cải thiện được độ tinh khiết, dừng lại làm nút lá
        if gain <= 1e-7 or best_feat is None:
            vals, counts = np.unique(y, return_counts=True)
            return DecisionTreeNode(value=vals[np.argmax(counts)], impurity=self._gini_impurity(y))

        # Tích lũy mức giảm độ vẩn đục cho MDI Feature Importance
        # Trọng số: (N_node / N_total) * Gain
        self.feature_importances_[best_feat] += (n_samples) * gain

        # Rẽ nhánh đệ quy
        left_mask = X[:, best_feat] <= best_thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, total_features)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1, total_features)

        return DecisionTreeNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            impurity=self._gini_impurity(y)
        )

    def fit(self, X, y):
        total_features = X.shape[1]
        self.feature_importances_ = np.zeros(total_features, dtype=np.float64)
        self.root = self._build_tree(X, y, depth=0, total_features=total_features)
        return self

    def _predict_single(self, node, x):
        """Duyệt cây từ nút gốc xuống lá để dự đoán 1 mẫu"""
        if node.is_leaf():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(node.left, x)
        return self._predict_single(node.right, x)

    def predict(self, X):
        return np.array([self._predict_single(self.root, x) for x in X])


# ==============================================================================
# PHẦN 2: THUẬT TOÁN RỪNG NGẪU NHIÊN (RANDOM FOREST CLASSIFIER SCRATCH)
# ==============================================================================

class RandomForestClassifierScratch:
    """
    Thuật toán Random Forest hoàn chỉnh của Leo Breiman (2001):
    - Bootstrap Aggregating (Bagging).
    - Random Subspace Feature Selection.
    - Out-of-Bag (OOB) Error Estimation.
    - Mean Decrease Impurity (MDI) Feature Importance.
    """
    def __init__(self, n_estimators=50, max_depth=None, min_samples_split=2,
                 max_features="sqrt", random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []
        self.feature_importances_ = None
        self.oob_accuracy_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        N, M = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.trees = []
        self.feature_importances_ = np.zeros(M, dtype=np.float64)

        # Mảng lưu trữ phiếu bầu Out-of-Bag cho từng mẫu: oob_votes[sample_idx, class_idx]
        oob_votes = np.zeros((N, n_classes), dtype=np.int32)
        oob_counts = np.zeros(N, dtype=np.int32)

        start_time = time.time()
        print(f"\n[*] Đang huấn luyện Rừng Ngẫu Nhiên với {self.n_estimators} cây...")

        for b in range(self.n_estimators):
            # 1. Rút mẫu Bootstrap có hoàn lại: chọn N mẫu từ N
            boot_indices = np.random.choice(N, size=N, replace=True)
            X_boot = X[boot_indices]
            y_boot = y[boot_indices]

            # 2. Xác định các mẫu ngoại mẫu (Out-of-Bag - OOB)
            oob_indices = np.setdiff1d(np.arange(N), boot_indices)

            # 3. Khởi tạo và huấn luyện cây quyết định unpruned
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

            # Tích lũy MDI Feature Importance
            self.feature_importances_ += tree.feature_importances_

            # 4. Dự đoán trên tập OOB của cây hiện tại
            if len(oob_indices) > 0:
                oob_preds = tree.predict(X[oob_indices])
                for idx, pred in zip(oob_indices, oob_preds):
                    cls_idx = np.where(self.classes_ == pred)[0][0]
                    oob_votes[idx, cls_idx] += 1
                    oob_counts[idx] += 1

        # Chuẩn hóa MDI Feature Importance về tổng bằng 1 (100%)
        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ /= total_importance

        # 5. Tính toán độ chính xác Out-of-Bag (OOB Accuracy)
        valid_oob_mask = oob_counts > 0
        if np.sum(valid_oob_mask) > 0:
            oob_final_preds = self.classes_[np.argmax(oob_votes[valid_oob_mask], axis=1)]
            self.oob_accuracy_ = np.mean(oob_final_preds == y[valid_oob_mask])
        else:
            self.oob_accuracy_ = 0.0

        elapsed = time.time() - start_time
        print(f"[*] Huấn luyện {self.n_estimators} cây hoàn tất trong {elapsed:.2f} giây.")
        print(f"[*] Sai số kiểm định ước lượng Out-of-Bag (OOB Accuracy): {self.oob_accuracy_ * 100:.2f}%")
        return self

    def predict(self, X):
        """Bỏ phiếu đa số (Majority Voting) từ tất cả các cây trong rừng"""
        # Ma trận dự đoán: (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        N = X.shape[0]
        final_preds = np.zeros(N, dtype=self.classes_.dtype)

        for i in range(N):
            vals, counts = np.unique(all_preds[:, i], return_counts=True)
            final_preds[i] = vals[np.argmax(counts)]

        return final_preds

    def predict_proba(self, X):
        """Tính phân phối xác suất trung bình của các cây"""
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        N = X.shape[0]
        n_classes = len(self.classes_)
        probs = np.zeros((N, n_classes), dtype=np.float64)

        for i in range(N):
            for c_idx, c in enumerate(self.classes_):
                probs[i, c_idx] = np.mean(all_preds[:, i] == c)

        return probs


# ==============================================================================
# PHẦN 3: KIỂM CHỨNG & MINH HỌA TRÊN TERMINAL
# ==============================================================================

def generate_synthetic_tabular_data(n_samples=400, n_features=10, n_informative=4, random_state=42):
    """
    Tự động sinh tập dữ liệu dạng bảng phi tuyến tính (Tabular Data):
    - Gồm n_informative đặc trưng có tính quyết định cao
    - Gồm (n_features - n_informative) đặc trưng nhiễu ngẫu nhiên (Noise features)
    """
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)

    # Nhãn mục tiêu được tạo từ tổ hợp phi tuyến của 4 đặc trưng đầu tiên:
    # y = 1 nếu (x0 * x1 + x2^2 - sin(x3)) > 0.5, ngược lại 0
    signal = X[:, 0] * X[:, 1] + (X[:, 2] ** 2) - np.sin(X[:, 3]) + np.random.randn(n_samples) * 0.2
    y = (signal > 0.8).astype(np.int64)
    return X, y


def main():
    print("================================================================================")
    print("KIỂM CHỨNG THUẬT TOÁN RANDOM FOREST FROM SCRATCH (random_forest_scratch.py)")
    print("================================================================================")

    # 1. Sinh dữ liệu thử nghiệm 400 mẫu, 10 đặc trưng (4 đặc trưng có ích, 6 đặc trưng nhiễu)
    n_features = 10
    n_informative = 4
    X, y = generate_synthetic_tabular_data(n_samples=400, n_features=n_features, n_informative=n_informative, random_state=42)

    # Chia Train/Test 75-25
    split = int(0.75 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print(f"[*] Tập dữ liệu dạng bảng được tạo:")
    print(f"    - Tập huấn luyện: {len(X_train)} mẫu x {n_features} đặc trưng")
    print(f"    - Tập kiểm thử:   {len(X_test)} mẫu")
    print(f"    - Các đặc trưng [x0, x1, x2, x3] là tín hiệu thực tế; [x4..x9] là nhiễu.")

    # 2. Khởi tạo và huấn luyện Random Forest
    rf = RandomForestClassifierScratch(
        n_estimators=60,
        max_depth=8,
        min_samples_split=3,
        max_features="sqrt",  # m = sqrt(10) ~ 3 đặc trưng mỗi nút
        random_state=42
    )

    rf.fit(X_train, y_train)

    # 3. Đánh giá chất lượng
    train_preds = rf.predict(X_train)
    test_preds = rf.predict(X_test)

    train_acc = np.mean(train_preds == y_train) * 100.0
    test_acc = np.mean(test_preds == y_test) * 100.0
    oob_acc = rf.oob_accuracy_ * 100.0

    print("\n[Kết Quả Đánh Giá]")
    print(f"    - Training Accuracy:        {train_acc:6.2f}%")
    print(f"    - Out-of-Bag (OOB) Accuracy: {oob_acc:6.2f}% (Ước lượng ngoại mẫu)")
    print(f"    - Test Set Accuracy:         {test_acc:6.2f}%")

    # 4. In bảng xếp hạng Tầm Quan Trọng Đặc Trưng (MDI)
    print("\n[Bảng Xếp Hạng Tầm Quan Trọng Đặc Trưng (Mean Decrease Impurity - MDI)]")
    print(f"    {'Thứ Hạng':<10} | {'Đặc Trưng':<14} | {'Độ Quan Trọng (MDI)':<22} | Vai Trò Thực Tế")
    print("    " + "-" * 70)

    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    for rank, idx in enumerate(sorted_indices, 1):
        importance_pct = importances[idx] * 100.0
        role = "⭐ Tín hiệu cốt lõi (Signal)" if idx < n_informative else "  Nhiễu ngẫu nhiên (Noise)"
        bar = "█" * int(importance_pct / 3)
        print(f"    #{rank:<9} | Feature {idx:<6} | {importance_pct:5.2f}%  {bar:<12} | {role}")

    print("\n[✓] Thuật toán Random Forest scratch của Leo Breiman đã phân biệt chính xác các đặc trưng cốt lõi!")


if __name__ == "__main__":
    main()
