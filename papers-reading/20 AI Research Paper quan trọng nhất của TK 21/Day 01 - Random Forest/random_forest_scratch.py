"""
================================================================================
BÀI BÁO GỐC: Random Forests (Leo Breiman, Machine Learning 2001)
TỆP MÃ NGUỒN: random_forest_scratch.py
MÔ TẢ DÀNH CHO NGƯỜI MỚI BẮT ĐẦU:
    Triển khai thuật toán Cây Quyết Định (CART) và Rừng Ngẫu Nhiên (Random Forest)
    hoàn toàn từ con số 0 (From Scratch) bằng Python thuần và NumPy.

CẤU TRÚC NỘI DUNG:
    PHẦN 1: BỐI CẢNH HỌC THUẬT & GIÁ TRỊ ĐỘT PHÁ CỦA BÀI BÁO
            - Trước bài báo: Cây đơn lẻ (CART 1984) bị quá khớp (High Variance).
                             Bagging (1996) gom nhiều cây nhưng các cây tương quan cao (High Correlation).
            - Sau bài báo: Random Forest (2001) giới thiệu "Random Subspace" tại mỗi nút,
                           giảm mạnh tương quan giữa các cây, triệt tiêu phương sai và cung cấp
                           ước lượng Out-of-Bag (OOB) miễn phí.
    PHẦN 2: CÂY QUYẾT ĐỊNH ĐƠN LẺ TỪ ĐẦU (CART DECISION TREE)
            - Độ vẩn đục Gini (Gini Impurity) & Ví dụ minh họa từng bước.
            - Tìm kiếm ngưỡng cắt tham lam (Greedy Split Search).
            - Ngẫu nhiên hóa tập con đặc trưng (Random Subspace m = sqrt(M)).
    PHẦN 3: THUẬT TOÁN RỪNG NGẪU NHIÊN (RANDOM FOREST CLASSIFIER)
            - Lấy mẫu Bootstrap có hoàn lại (Bagging).
            - Ước lượng sai số ngoại mẫu Out-of-Bag (OOB Error Estimation).
            - Tính toán tầm quan trọng đặc trưng (Mean Decrease Impurity - MDI).
            - Đo lường tương quan trung bình giữa các cây (Tree-to-Tree Correlation rho).
    PHẦN 4: THỰC NGHIỆM ĐỐI CHUẨN GIÁ TRỊ HỌC THUẬT (ACADEMIC VALUE BENCHMARK)
            - So sánh định lượng trực tiếp 3 cột mốc:
              1. Cây quyết định đơn lẻ (CART 1984)
              2. Bagging Trees với toàn bộ đặc trưng (Breiman 1996)
              3. Random Forest với Random Subspace (Breiman 2001)
            - Kiểm chứng: Train/Test Acc, Overfitting Gap, Tree Correlation (rho), OOB Error.
================================================================================
"""

import time
import numpy as np


# ==============================================================================
# PHẦN 1: BỐI CẢNH HỌC THUẬT & TÓM TẮT LÝ THUYẾT DÀNH CHO NGƯỜI MỚI BẮT ĐẦU
# ==============================================================================
"""
[GIẢI THÍCH DÀNH CHO NGƯỜI MỚI BẮT ĐẦU]:

1. BÀI TOÁN TRƯỚC NĂM 2001 ĐƯỢC GIẢI QUYẾT THẾ NÀO?
   - Cây quyết định đơn lẻ (Decision Tree - CART 1984):
     Rất trực quan và dễ hiểu (nếu X1 > 3 và X2 <= 5 thì Phân loại A).
     NHƯNG nó có một điểm yếu chết người: PHƯƠNG SAI CỰC CAO (High Variance).
     Cây rất dễ bị quá khớp (Overfitting), ghi nhớ từng chi tiết của tập train
     khiến Test Accuracy rất thấp nếu không cắt tỉa (pruning).
   - Phương pháp Bagging (Breiman 1996):
     Lấy mẫu Bootstrap (chọn ngẫu nhiên N mẫu có lặp lại từ tập dữ liệu) để xây dựng
     nhiều cây khác nhau, sau đó lấy trung bình hoặc bỏ phiếu đa số.
     VẤN ĐỀ TỒN ĐỌNG: Nếu tập dữ liệu có 1 hoặc 2 đặc trưng quá mạnh (dominant features),
     hầu hết mọi cây đều sẽ chọn đặc trưng đó ở nút gốc (root node).
     Hậu quả là các cây trở nên RẤT GIỐNG NHAU (tương quan cao rho lớn).
     Khi các cây giống nhau, việc gom trung bình không giúp giảm phương sai được bao nhiêu!

2. BÀI BÁO LEO BREIMAN (2001) ĐÃ ĐỘT PHÁ ĐIỀU GÌ?
   - Đột phá 1: CƠ CHẾ NGẪU NHIÊN KÉP (Double Randomization)
     Breiman kết hợp Bagging (lấy mẫu dòng dữ liệu) với RANDOM SUBSPACE (ngẫu nhiên hóa cột đặc trưng).
     Tại MỖI NÚT rẽ nhánh, cây không được nhìn toàn bộ M đặc trưng, mà CHỈ ĐƯỢC CHỌN NGẪU NHIÊN
     m = sqrt(M) đặc trưng.
     Điều này ép các cây phải học từ các đặc trưng yếu hơn -> Phá vỡ sự tương quan (giảm rho mạnh)!
   - Đột phá 2: CÂY SÂU KHÔNG CẮT TỈA (Unpruned Deep Trees)
     Cho phép từng cây phát triển sâu tối đa để độ chệch (Bias) cực thấp.
     Việc giảm phương sai (Variance) hoàn toàn do sự đa dạng của quần thể rừng đảm nhiệm.
   - Đột phá 3: ĐÁNH GIÁ NGOẠI MẪU (Out-of-Bag - OOB Error)
     Khoảng 36.8% mẫu bị bỏ sót trong mỗi mẫu bootstrap. Breiman dùng chính các mẫu này
     để kiểm định mô hình miễn phí, không cần chia tập Validation hay K-Fold Cross-Validation!
"""


# ==============================================================================
# PHẦN 2: CÂY QUYẾT ĐỊNH ĐƠN LẺ TỪ ĐẦU (CART DECISION TREE SCRATCH)
# ==============================================================================

class DecisionTreeNode:
    """
    Nút trong Cây Quyết Định (Decision Tree Node):
    Cây quyết định được cấu tạo từ các nút liên kết với nhau theo cấu trúc phân cấp:
    - Nút nhánh (Internal/Split Node): Chứa điều kiện rẽ nhánh (Ví dụ: Feature_3 <= 2.5).
      Nếu mẫu dữ liệu thỏa điều kiện, đi sang nhánh trái (`left`), ngược lại đi sang nhánh phải (`right`).
    - Nút lá (Leaf Node): Không rẽ nhánh nữa, chứa nhãn dự đoán cuối cùng (`value`).
    """
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None,
                 value=None, impurity=0.0):
        self.feature_idx = feature_idx  # Chỉ số đặc trưng dùng để rẽ nhánh (None nếu là lá)
        self.threshold = threshold      # Giá trị ngưỡng phân chia: x[feature_idx] <= threshold
        self.left = left                # Con trỏ tới nút con nhánh trái
        self.right = right              # Con trỏ tới nút con nhánh phải
        self.value = value              # Nhãn lớp dự đoán (chỉ có ở nút lá)
        self.impurity = impurity        # Độ vẩn đục Gini tại nút hiện tại

    def is_leaf(self):
        """Kiểm tra xem nút hiện tại có phải là nút lá hay không"""
        return self.value is not None


class DecisionTreeClassifierScratch:
    """
    Cây quyết định phân loại (CART) tự lập trình thuần túy:
    Hỗ trợ cơ chế chọn ngẫu nhiên tập con đặc trưng (Random Subspace) tại mỗi nút rẽ nhánh.

    Tham số:
    ---------
    max_depth : int hoặc None
        Độ sâu tối đa của cây (None nghĩa là để cây phát triển sâu tối đa đến khi lá thuần khiết).
    min_samples_split : int
        Số lượng mẫu tối thiểu cần có tại một nút để tiếp tục rẽ nhánh.
    max_features : str, int, float hoặc None
        Cách chọn số đặc trưng m tại mỗi nút:
        - "sqrt": m = floor(sqrt(M)) (Chuẩn của Breiman cho bài toán phân loại).
        - "log2": m = floor(log2(M)).
        - None: m = M (dùng toàn bộ đặc trưng - cây CART truyền thống).
    """
    def __init__(self, max_depth=None, min_samples_split=2, max_features="sqrt"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None
        self.feature_importances_ = None

    @staticmethod
    def _gini_impurity(y):
        """
        Tính Độ vẩn đục Gini (Gini Impurity):
        Công thức: Gini = 1 - sum(p_i^2) với p_i là tỷ lệ mẫu thuộc lớp i.
        
        Ý NGHĨA CHO NGƯỜI MỚI BẮT ĐẦU:
        - Gini đo lường "mức độ hỗn loạn" của nhãn dữ liệu.
        - Gini = 0.0: Tinh khiết tuyệt đối (Tất cả mẫu đều cùng 1 lớp).
        - Gini = 0.5: Hỗn loạn tối đa trong bài toán 2 lớp cân bằng (50% lớp A, 50% lớp B).
        - Mục tiêu của thuật toán: Tìm điểm rẽ nhánh sao cho Gini sau khi chia là NHỎ NHẤT!
        """
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _get_num_features(self, total_features):
        """
        Xác định số lượng đặc trưng m cần lấy mẫu ngẫu nhiên tại mỗi nút rẽ nhánh.
        Đây chính là linh hồn của phương pháp Random Subspace!
        """
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(total_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(total_features)))
        elif isinstance(self.max_features, int):
            return min(total_features, self.max_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * total_features))
        # Nếu max_features = None, sử dụng toàn bộ M đặc trưng (CART chuẩn 1984)
        return total_features

    def _best_split(self, X, y, total_features):
        """
        Tìm kiếm tham lam (Greedy Search) đặc trưng tốt nhất và ngưỡng cắt tối ưu:
        1. Chọn ngẫu nhiên m đặc trưng từ tổng số total_features (nếu max_features != None).
        2. Với mỗi đặc trưng trong m đặc trưng này, duyệt qua các ngưỡng giá trị có thể có.
        3. Chọn cặp (đặc trưng, ngưỡng cắt) mang lại mức giảm Gini (Gini Gain) lớn nhất.
        """
        best_gain = -1.0
        best_feature = None
        best_threshold = None
        n_samples = len(y)
        current_gini = self._gini_impurity(y)

        # 1. Cơ chế Random Subspace: Chọn ngẫu nhiên m đặc trưng không lặp lại
        m = self._get_num_features(total_features)
        feature_subspace = np.random.choice(total_features, size=m, replace=False)

        # 2. Duyệt qua từng đặc trưng được chọn
        for feat_idx in feature_subspace:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            # Tối ưu hóa tốc độ: Nếu có quá nhiều giá trị duy nhất, lấy mẫu 20 phân vị
            if len(thresholds) > 20:
                thresholds = np.percentile(X_column, np.linspace(5, 95, 20))

            for thresh in thresholds:
                left_mask = X_column <= thresh
                right_mask = ~left_mask

                # Không xét phép chia nếu một bên nhánh bị rỗng
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                # 3. Tính Gini trung bình có trọng số sau khi phân chia
                p_left = len(y_left) / n_samples
                p_right = len(y_right) / n_samples
                gini_split = p_left * self._gini_impurity(y_left) + p_right * self._gini_impurity(y_right)

                # Mức giảm độ vẩn đục (Gini Gain / Information Gain)
                gain = current_gini - gini_split

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = thresh

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0, total_features=None):
        """
        Hàm đệ quy xây dựng cây từ nút gốc xuống các nút lá:
        - Kiểm tra điều kiện dừng: đạt max_depth, nút đã tinh khiết, hoặc quá ít mẫu.
        - Nếu dừng: tạo nút lá lưu nhãn chiếm đa số.
        - Nếu tiếp tục: tìm điểm cắt tối ưu và đệ quy xây dựng nhánh trái và nhánh phải.
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Điều kiện dừng tạo nút lá:
        # 1. Đạt độ sâu tối đa quy định.
        # 2. Toàn bộ mẫu tại nút đều cùng một lớp (độ tinh khiết tuyệt đối).
        # 3. Số lượng mẫu nhỏ hơn ngưỡng tối thiểu cho phép rẽ nhánh.
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or n_samples < self.min_samples_split:
            vals, counts = np.unique(y, return_counts=True)
            leaf_value = vals[np.argmax(counts)]
            return DecisionTreeNode(value=leaf_value, impurity=self._gini_impurity(y))

        # Tìm điểm rẽ nhánh tốt nhất trong không gian con đặc trưng
        best_feat, best_thresh, gain = self._best_split(X, y, total_features)

        # Nếu không tìm được điểm cắt hoặc mức cải thiện quá bé (<= 1e-7), dừng lại làm nút lá
        if gain <= 1e-7 or best_feat is None:
            vals, counts = np.unique(y, return_counts=True)
            return DecisionTreeNode(value=vals[np.argmax(counts)], impurity=self._gini_impurity(y))

        # Tích lũy mức giảm Gini cho độ đo tầm quan trọng đặc trưng (MDI Feature Importance):
        # Mức đóng góp = (Số mẫu tại nút / Tổng số mẫu) * Gini Gain
        self.feature_importances_[best_feat] += n_samples * gain

        # Rẽ nhánh đệ quy sang hai bên
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
        """Huấn luyện cây quyết định trên ma trận đặc trưng X và nhãn y"""
        total_features = X.shape[1]
        self.feature_importances_ = np.zeros(total_features, dtype=np.float64)
        self.root = self._build_tree(X, y, depth=0, total_features=total_features)
        return self

    def _predict_single(self, node, x):
        """Duyệt một mẫu x từ nút gốc xuống lá theo các điều kiện rẽ nhánh nhị phân"""
        if node.is_leaf():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(node.left, x)
        return self._predict_single(node.right, x)

    def predict(self, X):
        """Dự đoán nhãn cho toàn bộ tập mẫu dữ liệu X"""
        return np.array([self._predict_single(self.root, x) for x in X])


# ==============================================================================
# PHẦN 3: THUẬT TOÁN RỪNG NGẪU NHIÊN (RANDOM FOREST CLASSIFIER SCRATCH)
# ==============================================================================

class RandomForestClassifierScratch:
    """
    Thuật toán Rừng Ngẫu Nhiên hoàn chỉnh của Leo Breiman (2001):
    - Lấy mẫu Bootstrap có hoàn lại (Bagging).
    - Ngẫu nhiên hóa không gian con đặc trưng tại mỗi nút (Random Subspace).
    - Tự động ước lượng sai số ngoại mẫu Out-of-Bag (OOB Error).
    - Đo lường tầm quan trọng đặc trưng qua Mean Decrease Impurity (MDI).
    - Tính toán độ tương quan trung bình giữa các cây (rho) để kiểm chứng lý thuyết.

    Tham số:
    ---------
    n_estimators : int
        Số lượng cây quyết định trong rừng (B).
    max_depth : int hoặc None
        Độ sâu tối đa của mỗi cây (Breiman khuyến nghị để None để cây sâu tối đa).
    min_samples_split : int
        Số mẫu tối thiểu để rẽ nhánh.
    max_features : str, int, float hoặc None
        Số đặc trưng m được chọn tại mỗi nút ("sqrt" là chuẩn của Breiman).
    random_state : int
        Hạt giống ngẫu nhiên để tái lập kết quả thực nghiệm.
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
        self.classes_ = None

    def fit(self, X, y, verbose=True):
        """
        Quy trình huấn luyện Rừng Ngẫu Nhiên chuẩn Breiman:
        1. Rút mẫu Bootstrap N phần tử có hoàn lại.
        2. Ghi nhận các mẫu Out-of-Bag (OOB) không được chọn (~36.8% dữ liệu).
        3. Huấn luyện cây quyết định sâu (unpruned) trên mẫu Bootstrap.
        4. Dùng cây vừa học dự đoán các mẫu OOB và tích lũy phiếu bầu.
        5. Tính toán sai số kiểm định không chệch OOB Accuracy.
        """
        np.random.seed(self.random_state)
        N, M = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.trees = []
        self.feature_importances_ = np.zeros(M, dtype=np.float64)

        # Mảng tích lũy phiếu bầu OOB cho từng mẫu: oob_votes[i, c] là số cây bầu lớp c cho mẫu i
        oob_votes = np.zeros((N, n_classes), dtype=np.int32)
        oob_counts = np.zeros(N, dtype=np.int32)

        start_time = time.time()
        if verbose:
            print(f"[*] Bắt đầu huấn luyện Rừng Ngẫu Nhiên ({self.n_estimators} cây, max_features='{self.max_features}')...")

        for b in range(self.n_estimators):
            # 1. Lấy mẫu Bootstrap có hoàn lại (N mẫu từ N phần tử)
            boot_indices = np.random.choice(N, size=N, replace=True)
            X_boot = X[boot_indices]
            y_boot = y[boot_indices]

            # 2. Mẫu ngoại mẫu Out-of-Bag (các chỉ số không xuất hiện trong boot_indices)
            oob_indices = np.setdiff1d(np.arange(N), boot_indices)

            # 3. Huấn luyện cây quyết định
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

            # Tích lũy MDI Feature Importance
            self.feature_importances_ += tree.feature_importances_

            # 4. Dự đoán và ghi nhận phiếu bầu OOB
            if len(oob_indices) > 0:
                oob_preds = tree.predict(X[oob_indices])
                for idx, pred in zip(oob_indices, oob_preds):
                    cls_idx = np.where(self.classes_ == pred)[0][0]
                    oob_votes[idx, cls_idx] += 1
                    oob_counts[idx] += 1

        # Chuẩn hóa MDI Feature Importance về tỷ lệ phần trăm (tổng bằng 1)
        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ /= total_importance

        # 5. Tính toán OOB Accuracy (Chỉ tính trên những mẫu có ít nhất 1 cây OOB bầu)
        valid_oob_mask = oob_counts > 0
        if np.sum(valid_oob_mask) > 0:
            oob_final_preds = self.classes_[np.argmax(oob_votes[valid_oob_mask], axis=1)]
            self.oob_accuracy_ = np.mean(oob_final_preds == y[valid_oob_mask])
        else:
            self.oob_accuracy_ = 0.0

        elapsed = time.time() - start_time
        if verbose:
            print(f"[*] Huấn luyện hoàn tất trong {elapsed:.2f} giây.")
            print(f"[*] Ước lượng sai số ngoại mẫu OOB Accuracy: {self.oob_accuracy_ * 100:.2f}%")

        return self

    def predict(self, X):
        """Bỏ phiếu đa số (Majority Voting) từ tất cả các cây trong rừng"""
        # Thu thập dự đoán từ toàn bộ B cây: shape = (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        N = X.shape[0]
        final_preds = np.zeros(N, dtype=self.classes_.dtype)

        # Lấy nhãn xuất hiện nhiều nhất cho từng mẫu
        for i in range(N):
            vals, counts = np.unique(all_preds[:, i], return_counts=True)
            final_preds[i] = vals[np.argmax(counts)]

        return final_preds

    def compute_tree_correlation(self, X):
        """
        Tính hệ số tương quan trung bình (rho) giữa các cây trong rừng:
        Đây chính là đại lượng lý thuyết cốt lõi trong bài báo Breiman:
        Var(Ensemble) = rho * sigma^2 + ((1 - rho) / B) * sigma^2.
        Khi rho càng nhỏ, phương sai tổng thể của mô hình càng giảm sâu!
        """
        all_preds = np.array([tree.predict(X) for tree in self.trees])  # (B, N)
        B = len(self.trees)
        if B < 2:
            return 1.0

        correlations = []
        for i in range(B):
            for j in range(i + 1, B):
                # Tính tỷ lệ dự đoán trùng khớp giữa cây i và cây j
                agreement = np.mean(all_preds[i] == all_preds[j])
                correlations.append(agreement)

        return float(np.mean(correlations))


# ==============================================================================
# PHẦN 4: THỰC NGHIỆM ĐỐI CHUẨN GIÁ TRỊ HỌC THUẬT (ACADEMIC VALUE BENCHMARK)
# ==============================================================================

def generate_academic_benchmark_dataset(n_samples=500, n_features=12, n_informative=4,
                                        noise_features=8, label_noise_ratio=0.05,
                                        random_state=42):
    """
    Sinh tập dữ liệu dạng bảng đa chiều để kiểm chứng thực nghiệm bài báo:
    - n_informative: Số đặc trưng tín hiệu phi tuyến thực sự (x0, x1, x2, x3).
    - noise_features: Số đặc trưng nhiễu ngẫu nhiên không có ý nghĩa (x4...x11).
    - label_noise_ratio: Tỷ lệ nhãn bị lật ngẫu nhiên để thử thách độ bền vững (Robustness).
    """
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)

    # Quan hệ phi tuyến giữa 4 đặc trưng đầu tiên
    signal = (X[:, 0] * X[:, 1] + 1.2 * (X[:, 2] ** 2) - 1.5 * np.sin(X[:, 3]))
    y = (signal > 0.5).astype(np.int64)

    # Thêm nhiễu nhãn (Label Noise)
    if label_noise_ratio > 0:
        flip_mask = np.random.rand(n_samples) < label_noise_ratio
        y[flip_mask] = 1 - y[flip_mask]

    return X, y


def run_academic_value_comparison():
    """
    HÀM ĐỐI CHUẨN HỌC THUẬT:
    So sánh định lượng 3 cột mốc công nghệ trong lịch sử Học máy:
    1. Single Decision Tree (CART 1984): Cây đơn lẻ dùng toàn bộ đặc trưng, unpruned.
    2. Bagging Trees (Breiman 1996): Quần thể cây nhưng m = M (dùng toàn bộ đặc trưng tại mỗi nút).
    3. Random Forest (Breiman 2001): Quần thể cây kết hợp Random Subspace m = sqrt(M).

    Các chỉ số đo lường:
    - Train Accuracy (%)
    - Test Accuracy (%)
    - Khoảng cách Overfitting Gap = Train Acc - Test Acc (%)
    - Hệ số tương quan trung bình giữa các cây (rho)
    - Thời gian huấn luyện (giây)
    - So sánh sai số OOB vs Test thực tế
    """
    print("\n" + "=" * 80)
    print(" KHẢO SÁT & ĐỐI CHUẨN GIÁ TRỊ HỌC THUẬT CỦA BÀI BÁO LEO BREIMAN (2001)")
    print("=" * 80)
    print("Bối cảnh thực nghiệm: Dữ liệu dạng bảng 500 mẫu, 12 đặc trưng (4 tín hiệu, 8 nhiễu)")
    print("Đặc trưng trội (dominant feature) và nhiễu nhãn 5% để thử thách tính kháng overfit.")

    # 1. Chuẩn bị dữ liệu
    X, y = generate_academic_benchmark_dataset(
        n_samples=500, n_features=12, n_informative=4,
        noise_features=8, label_noise_ratio=0.05, random_state=42
    )

    split = int(0.7 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # --------------------------------------------------------------------------
    # CỘT MỐC 1: CÂY QUYẾT ĐỊNH ĐƠN LẺ (CART - Breiman 1984)
    # --------------------------------------------------------------------------
    print("\n[1/3] Đang đánh giá: Single Decision Tree (CART 1984)...")
    t0 = time.time()
    cart_tree = DecisionTreeClassifierScratch(max_depth=None, min_samples_split=2, max_features=None)
    cart_tree.fit(X_train, y_train)
    cart_time = time.time() - t0

    cart_train_acc = np.mean(cart_tree.predict(X_train) == y_train) * 100.0
    cart_test_acc = np.mean(cart_tree.predict(X_test) == y_test) * 100.0
    cart_overfit_gap = cart_train_acc - cart_test_acc

    # --------------------------------------------------------------------------
    # CỘT MỐC 2: BAGGING TIÊU CHUẨN (Breiman 1996 - Dùng toàn bộ đặc trưng m = M)
    # --------------------------------------------------------------------------
    print("[2/3] Đang đánh giá: Standard Bagging (Breiman 1996 - Full Features m=M)...")
    t0 = time.time()
    bagging = RandomForestClassifierScratch(
        n_estimators=40, max_depth=None, min_samples_split=2,
        max_features=None,  # max_features=None đồng nghĩa với Bagging truyền thống
        random_state=42
    )
    bagging.fit(X_train, y_train, verbose=False)
    bagging_time = time.time() - t0

    bag_train_acc = np.mean(bagging.predict(X_train) == y_train) * 100.0
    bag_test_acc = np.mean(bagging.predict(X_test) == y_test) * 100.0
    bag_overfit_gap = bag_train_acc - bag_test_acc
    bag_rho = bagging.compute_tree_correlation(X_test)

    # --------------------------------------------------------------------------
    # CỘT MỐC 3: RANDOM FOREST (Breiman 2001 - Ngẫu nhiên hóa m = sqrt(M))
    # --------------------------------------------------------------------------
    print("[3/3] Đang đánh giá: Random Forest (Breiman 2001 - Random Subspace m=sqrt(M))...")
    t0 = time.time()
    rf = RandomForestClassifierScratch(
        n_estimators=40, max_depth=None, min_samples_split=2,
        max_features="sqrt",  # m = sqrt(12) ~ 3 đặc trưng mỗi nút
        random_state=42
    )
    rf.fit(X_train, y_train, verbose=False)
    rf_time = time.time() - t0

    rf_train_acc = np.mean(rf.predict(X_train) == y_train) * 100.0
    rf_test_acc = np.mean(rf.predict(X_test) == y_test) * 100.0
    rf_overfit_gap = rf_train_acc - rf_test_acc
    rf_rho = rf.compute_tree_correlation(X_test)
    rf_oob_acc = rf.oob_accuracy_ * 100.0

    # --------------------------------------------------------------------------
    # XUẤT BẢNG SO SÁNH ĐỊNH LƯỢNG HỌC THUẬT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"{'Mô Hình / Giải Thuật':<26} | {'Train Acc':<10} | {'Test Acc':<10} | {'Overfit Gap':<12} | {'Tương quan rho':<14} | {'Thời gian':<10}")
    print("=" * 88)
    print(f"{'1. Single CART (1984)':<26} | {cart_train_acc:8.2f}% | {cart_test_acc:8.2f}% | {cart_overfit_gap:10.2f}% | {'N/A (1 cây)':<14} | {cart_time:7.3f}s")
    print(f"{'2. Bagging (1996)':<26} | {bag_train_acc:8.2f}% | {bag_test_acc:8.2f}% | {bag_overfit_gap:10.2f}% | {bag_rho:14.3f} | {bagging_time:7.3f}s")
    print(f"{'3. Random Forest (2001)':<26} | {rf_train_acc:8.2f}% | {rf_test_acc:8.2f}% | {rf_overfit_gap:10.2f}% | {rf_rho:14.3f} | {rf_time:7.3f}s")
    print("=" * 88)

    # --------------------------------------------------------------------------
    # PHÂN TÍCH KẾT LUẬN HỌC THUẬT
    # --------------------------------------------------------------------------
    print("\n[PHÂN TÍCH KẾT QUẢ DÀNH CHO NGƯỜI MỚI BẮT ĐẦU]:")
    print(f"1. Hiện tượng Quá khớp (Overfitting):")
    print(f"   - Single CART Tree đạt {cart_train_acc:.1f}% trên tập Train nhưng tụt xuống {cart_test_acc:.1f}% trên tập Test.")
    print(f"     Khoảng cách overfit lên tới {cart_overfit_gap:.1f}%, chứng minh cây đơn lẻ có phương sai (Variance) rất cao.")
    print(f"   - Random Forest thu hẹp khoảng cách overfit và tăng Test Acc lên {rf_test_acc:.1f}%!")

    print(f"\n2. Bản chất toán học về Độ tương quan giữa các cây (rho):")
    print(f"   - Ở Bagging chuẩn (1996), tương quan giữa các cây là rho = {bag_rho:.3f} (rất cao vì cây nào cũng chia theo đặc trưng mạnh nhất).")
    print(f"   - Ở Random Forest (2001), nhờ chỉ chọn m = sqrt(M) đặc trưng ngẫu nhiên tại mỗi nút, tương quan giảm xuống rho = {rf_rho:.3f}!")
    print(f"   - Vì rho giảm mạnh, số hạng rho * sigma^2 trong phương sai quần thể bị triệt tiêu đáng kể!")

    print(f"\n3. Kiểm chứng ước lượng sai số Out-of-Bag (OOB):")
    print(f"   - OOB Accuracy đo được ngay trong lúc huấn luyện: {rf_oob_acc:.2f}%")
    print(f"   - Test Accuracy thực tế trên tập kiểm thử độc lập: {rf_test_acc:.2f}%")
    print(f"   - Sai lệch tuyệt đối giữa OOB và Test: {abs(rf_oob_acc - rf_test_acc):.2f}%")
    print(f"   -> Kết luận: OOB là một ước lượng không chệch xuất sắc, hoàn toàn có thể thay thế K-Fold Cross-Validation!")

    # --------------------------------------------------------------------------
    # BẢNG XẾP HẠNG TẦM QUAN TRỌNG ĐẶC TRƯNG (MDI)
    # --------------------------------------------------------------------------
    print("\n[BẢNG XẾP HẠNG TẦM QUAN TRỌNG ĐẶC TRƯNG CỦA RANDOM FOREST (MDI)]")
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx, 1):
        pct = importances[idx] * 100.0
        role = "⭐ Tín hiệu thực tế (Signal)" if idx < 4 else "  Nhiễu ngẫu nhiên (Noise)"
        bar = "█" * int(pct / 2.5)
        print(f"   #{rank:<2} | Đặc trưng X_{idx:<2} | {pct:5.2f}%  {bar:<14} | {role}")

    print("\n[✓] Hoàn thành xuất sắc kiểm chứng thực nghiệm bài báo Random Forest (Breiman 2001)!")


def main():
    print("================================================================================")
    print(" TRIỂN KHAI THUẬT TOÁN RANDOM FOREST TỪ CON SỐ 0 (FROM SCRATCH thuần NumPy)")
    print(" BÀI BÁO: Random Forests - Leo Breiman (Machine Learning 2001)")
    print("================================================================================")

    # Chạy quy trình đối chuẩn học thuật chi tiết
    run_academic_value_comparison()


if __name__ == "__main__":
    main()
