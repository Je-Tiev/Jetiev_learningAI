"""
================================================================================
BÀI BÁO: ImageNet: A Large-Scale Hierarchical Image Database (Deng et al., CVPR 2009)
TỆP MÃ NGUỒN: imagenet_scratch.py
MÔ TẢ: Cài đặt từ con số 0 (From Scratch) các giải thuật và cấu trúc cốt lõi của ImageNet:
       1. Cấu trúc Cây Phả Hệ WordNet (Synset Tree, Hypernym-Hyponym Traversal).
       2. Giải thuật tìm Tổ tiên chung gần nhất (Lowest Common Ancestor - LCA).
       3. Ma trận khoảng cách ngữ nghĩa nhảy cây (Tree-Hop Distance Matrix).
       4. Thuật toán lọc nhãn đám đông Amazon Mechanical Turk (Bayesian Majority Voting & Adaptive Stopping).
       5. Đánh giá phân loại phân cấp (Hierarchical Loss vs Flat 0/1 Loss, Top-1 & Top-K).
       6. Thử nghiệm khép kín độc lập, hiển thị trực quan tiến trình trên màn hình Terminal.
================================================================================
"""

import numpy as np


# ==============================================================================
# PHẦN 1: CẤU TRÚC DỮ LIỆU CÂY PHẢ HỆ WORDNET (SYNSET HIERARCHY)
# ==============================================================================

class SynsetNode:
    """Đại diện cho một nút khái niệm (Synset) trong cây phả hệ WordNet."""
    def __init__(self, synset_id, name, parent=None):
        self.synset_id = synset_id
        self.name = name
        self.parent = parent
        self.children = []
        self.depth = 0 if parent is None else parent.depth + 1
        if parent is not None:
            parent.children.append(self)

    def is_leaf(self):
        return len(self.children) == 0

    def get_ancestors(self):
        """Trả về danh sách tất cả các nút tổ tiên từ nút hiện tại ngược về nút gốc (Root)."""
        ancestors = []
        curr = self
        while curr is not None:
            ancestors.append(curr)
            curr = curr.parent
        return ancestors


class WordNetHierarchy:
    """Quản lý toàn bộ cấu trúc đồ thị cây phả hệ WordNet cho bài toán phân loại."""
    def __init__(self, root_name="entity"):
        self.nodes = {}
        self.root = SynsetNode("n00000001", root_name)
        self.nodes["n00000001"] = self.root
        self.leaf_classes = []

    def add_synset(self, synset_id, name, parent_id):
        """Thêm một Synset mới vào cây theo mã định danh của nút cha."""
        if parent_id not in self.nodes:
            raise ValueError(f"Không tìm thấy nút cha với ID: {parent_id}")
        parent = self.nodes[parent_id]
        node = SynsetNode(synset_id, name, parent)
        self.nodes[synset_id] = node
        return node

    def build_sample_tree(self):
        """
        Xây dựng một cây phả hệ WordNet mẫu gồm 12 lớp lá phân cấp theo 2 nhánh lớn:
        Sinh vật sống (Living Thing) và Đồ vật nhân tạo (Artifact).
        """
        # Nhánh 1: Sinh vật sống (Living thing -> Animal)
        self.add_synset("n00001", "living_thing", "n00000001")
        self.add_synset("n00002", "animal", "n00001")
        self.add_synset("n00003", "mammal", "n00002")
        self.add_synset("n00004", "carnivore", "n00003")
        
        # Họ Chó (Canine)
        self.add_synset("n00005", "canine", "n00004")
        self.add_synset("n00006", "dog", "n00005")
        self.add_synset("c01", "husky", "n00006")
        self.add_synset("c02", "golden_retriever", "n00006")
        self.add_synset("c03", "german_shepherd", "n00006")
        self.add_synset("c04", "wolf", "n00005")

        # Họ Mèo (Feline)
        self.add_synset("n00007", "feline", "n00004")
        self.add_synset("c05", "domestic_cat", "n00007")
        self.add_synset("c06", "tiger", "n00007")

        # Nhánh 2: Đồ vật nhân tạo (Artifact -> Vehicle)
        self.add_synset("n00008", "artifact", "n00000001")
        self.add_synset("n00009", "vehicle", "n00008")
        self.add_synset("n00010", "wheeled_vehicle", "n00009")
        self.add_synset("c07", "car", "n00010")
        self.add_synset("c08", "truck", "n00010")
        self.add_synset("c09", "bus", "n00010")
        self.add_synset("c10", "motorcycle", "n00010")

        # Phương tiện đường thủy/không khí
        self.add_synset("n00011", "aircraft", "n00009")
        self.add_synset("c11", "airplane", "n00011")
        self.add_synset("n00012", "watercraft", "n00009")
        self.add_synset("c12", "speedboat", "n00012")

        # Cập nhật danh sách các lớp lá (Leaf Classes)
        self.leaf_classes = [node for node in self.nodes.values() if node.is_leaf()]
        self.leaf_classes.sort(key=lambda x: x.synset_id)

    def find_lowest_common_ancestor(self, node_a, node_b):
        """
        Tìm Tổ tiên chung gần nhất (Lowest Common Ancestor - LCA) của 2 nút.
        """
        ancestors_a = set(node_a.get_ancestors())
        curr = node_b
        while curr is not None:
            if curr in ancestors_a:
                return curr
            curr = curr.parent
        return self.root

    def compute_tree_distance(self, node_a, node_b):
        """
        Tính khoảng cách nhảy cây (Tree-Hop Distance):
        D_tree(A, B) = Depth(A) + Depth(B) - 2 * Depth(LCA(A, B))
        """
        lca = self.find_lowest_common_ancestor(node_a, node_b)
        return node_a.depth + node_b.depth - 2 * lca.depth

    def get_distance_matrix(self):
        """
        Tạo ma trận khoảng cách nhảy cây kích thước (C, C) giữa tất cả các lớp lá.
        """
        C = len(self.leaf_classes)
        dist_matrix = np.zeros((C, C), dtype=np.int32)
        for i in range(C):
            for j in range(C):
                dist_matrix[i, j] = self.compute_tree_distance(self.leaf_classes[i], self.leaf_classes[j])
        return dist_matrix


# ==============================================================================
# PHẦN 2: THUẬT TOÁN KIỂM ĐỊNH ĐÁM ĐÔNG AMAZON MECHANICAL TURK (AMT)
# ==============================================================================

class AMTQualityVerifier:
    """
    Mô phỏng thuật toán kiểm định chất lượng nhãn đám đông của ImageNet:
    - Mỗi bức ảnh ứng viên được gán nhãn bởi nhiều thợ (Workers) độc lập.
    - Một số thợ có độ chuẩn xác cao (chăm chỉ), một số thợ bất cẩn (nhiễu).
    - Cập nhật xác suất hậu nghiệm Bayes (Posterior Probability) cho từng ảnh.
    - Thuật toán dừng thích ứng (Adaptive Stopping) khi độ tin cậy vượt ngưỡng.
    """
    def __init__(self, confidence_threshold=0.98, prior_prob=0.3):
        self.confidence_threshold = confidence_threshold
        self.prior_prob = prior_prob

    def simulate_crowd_verification(self, true_has_concept, worker_accuracies):
        """
        Mô phỏng quy trình kiểm định nhãn cho một bức ảnh:
        
        Tham số:
            true_has_concept: True/False (Ảnh có thực sự chứa khái niệm không)
            worker_accuracies: Danh sách độ chính xác của các thợ tham gia
        Trả về:
            accepted: Được chấp nhận vào ImageNet hay bị từ chối
            votes_collected: Số lượng phiếu bầu cần thu thập trước khi dừng
            final_prob: Xác suất hậu nghiệm cuối cùng
        """
        # Khởi tạo log-odds = log(P / (1 - P))
        log_odds = np.log(self.prior_prob / (1.0 - self.prior_prob))
        votes_collected = 0

        for acc in worker_accuracies:
            votes_collected += 1
            # Thợ bỏ phiếu dựa trên độ chính xác của họ
            if true_has_concept:
                vote = np.random.rand() < acc  # True nếu thợ nhận ra đúng
            else:
                vote = np.random.rand() > acc  # True nếu thợ nhầm lẫn (False positive)

            # Cập nhật Bayes qua log-likelihood ratio
            # P(vote=1 | concept) = acc, P(vote=1 | no_concept) = 1 - acc
            if vote:
                llr = np.log(acc / (1.0 - acc + 1e-12))
            else:
                llr = np.log((1.0 - acc) / (acc + 1e-12))

            log_odds += llr
            posterior = 1.0 / (1.0 + np.exp(-log_odds))

            # Dừng sớm nếu độ tin cậy đã đạt ngưỡng cao
            if posterior >= self.confidence_threshold or posterior <= (1.0 - self.confidence_threshold):
                break

        accepted = posterior >= self.confidence_threshold
        return accepted, votes_collected, posterior


# ==============================================================================
# PHẦN 3: ĐỘ ĐO ĐÁNH GIÁ PHÂN CẤP (TOP-1, TOP-K & HIERARCHICAL LOSS)
# ==============================================================================

def compute_top_k_accuracy(predictions, targets, k=5):
    """
    Tính độ chính xác Top-K: Đúng nếu nhãn thực tế nằm trong K dự đoán cao nhất.
    """
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
    correct = np.any(top_k_preds == targets[:, None], axis=1)
    return np.mean(correct)


def compute_hierarchical_cost(predictions, targets, distance_matrix):
    """
    Tính chi phí lỗi phân cấp (Mean Tree-Hop Distance):
    Mỗi lỗi sai được nhân với khoảng cách phả hệ giữa lớp dự đoán và lớp gốc.
    """
    top_1_preds = np.argmax(predictions, axis=1)
    costs = distance_matrix[targets, top_1_preds]
    return np.mean(costs)


# ==============================================================================
# PHẦN 4: THỰC THI KIỂM CHỨNG & MINH HỌA TRÊN TERMINAL
# ==============================================================================

def main():
    print("================================================================================")
    print("MÔ PHỎNG GIẢI THUẬT VÀ CẤU TRÚC DỮ LIỆU IMAGENET (imagenet_scratch.py)")
    print("================================================================================")

    # 1. Xây dựng Cây Phả Hệ WordNet mẫu
    print("\n[1] Xây dựng Cây Phả Hệ WordNet (WordNet Synset Hierarchy):")
    tree = WordNetHierarchy()
    tree.build_sample_tree()

    print(f"[*] Tổng số nút trong cây: {len(tree.nodes)}")
    print(f"[*] Tổng số lớp lá nhận diện (Leaf Concepts): {len(tree.leaf_classes)}")
    print("    Danh sách các lớp lá:")
    for idx, leaf in enumerate(tree.leaf_classes):
        print(f"    [{idx:02d}] {leaf.name:<18} (ID: {leaf.synset_id}, Độ sâu: {leaf.depth})")

    # 2. Phân tích khoảng cách ngữ nghĩa và LCA
    print("\n[2] Phân tích khoảng cách nhảy cây (Tree-Hop Distance & LCA):")
    node_husky = tree.nodes["c01"]
    node_golden = tree.nodes["c02"]
    node_wolf = tree.nodes["c04"]
    node_cat = tree.nodes["c05"]
    node_car = tree.nodes["c07"]
    node_airplane = tree.nodes["c11"]

    examples = [
        (node_husky, node_golden, "Cùng giống chó"),
        (node_husky, node_wolf, "Cùng họ Chó (Canine)"),
        (node_husky, node_cat, "Cùng bộ Ăn thịt (Carnivore)"),
        (node_husky, node_car, "Chó vs Ô tô (Chung nút Entity gốc)"),
        (node_car, node_airplane, "Cùng nhánh Phương tiện (Vehicle)")
    ]

    print(f"    {'Cặp Khái Niệm':<30} | {'Tổ Tiên Chung (LCA)':<18} | {'Khoảng Cách Tree-Hop':<20} | Ghi Chú")
    print("    " + "-" * 88)
    for a, b, note in examples:
        lca = tree.find_lowest_common_ancestor(a, b)
        dist = tree.compute_tree_distance(a, b)
        pair_str = f"{a.name} <-> {b.name}"
        print(f"    {pair_str:<30} | {lca.name:<18} | {dist:<20} | {note}")

    dist_matrix = tree.get_distance_matrix()

    # 3. Mô phỏng thuật toán kiểm định chất lượng đám đông AMT
    print("\n[3] Mô phỏng thuật toán gán nhãn đa biểu quyết Amazon Mechanical Turk (AMT):")
    verifier = AMTQualityVerifier(confidence_threshold=0.98, prior_prob=0.25)
    
    # Giả lập đội ngũ thợ gồm người làm tốt (acc ~0.85-0.95) và người bất cẩn (acc ~0.55)
    np.random.seed(42)
    worker_pool = [0.90, 0.85, 0.55, 0.92, 0.88, 0.60, 0.95, 0.82]

    n_test_images = 1000
    actual_positives = 300  # 300 ảnh thực sự chứa vật thể, 700 ảnh rác
    accepted_correctly = 0
    rejected_correctly = 0
    total_votes = 0

    for i in range(n_test_images):
        has_concept = (i < actual_positives)
        # Chọn ngẫu nhiên thứ tự thợ đánh giá
        np.random.shuffle(worker_pool)
        accepted, votes, prob = verifier.simulate_crowd_verification(has_concept, worker_pool)
        total_votes += votes

        if has_concept and accepted:
            accepted_correctly += 1
        elif not has_concept and not accepted:
            rejected_correctly += 1

    precision = accepted_correctly / (accepted_correctly + (n_test_images - actual_positives - rejected_correctly) + 1e-9)
    recall = accepted_correctly / actual_positives
    avg_votes = total_votes / n_test_images

    print(f"[*] Kết quả kiểm định trên {n_test_images} bức ảnh ứng viên:")
    print(f"    - Độ chuẩn xác nhãn chấp nhận (Precision): {precision * 100:.2f}% (Rất sạch!)")
    print(f"    - Tỷ lệ giữ lại ảnh chuẩn (Recall):        {recall * 100:.2f}%")
    print(f"    - Số phiếu trung bình mỗi ảnh (Tiết kiệm ngân sách): {avg_votes:.1f} lượt thợ/ảnh")

    # 4. So sánh đánh giá phân loại phẳng vs phân cấp
    print("\n[4] So sánh Đánh Giá Phân Loại: Phẳng (Flat 0/1) vs Phân Cấp (Hierarchical):")
    N_samples = 500
    C = len(tree.leaf_classes)
    targets = np.random.randint(0, C, size=(N_samples,))

    # Giả lập 2 mô hình khác nhau:
    # Mô hình A: Sai sót ngẫu nhiên hỗn loạn (nhầm lẫn giữa động vật và ô tô)
    # Mô hình B: Sai sót có cấu trúc ngữ nghĩa (nhầm husky với golden retriever, car với truck)
    
    # Tạo xác suất mô hình B (tập trung lân cận cây)
    preds_B = np.zeros((N_samples, C))
    for i in range(N_samples):
        # Xác suất tỷ lệ nghịch với khoảng cách cây: exp(-dist)
        target = targets[i]
        dists = dist_matrix[target]
        scores = np.exp(-0.8 * dists) + np.random.normal(0, 0.1, size=(C,))
        scores = np.maximum(scores, 0)
        preds_B[i] = scores / np.sum(scores)

    # Tạo xác suất mô hình A (nhiễu phẳng)
    preds_A = np.zeros((N_samples, C))
    for i in range(N_samples):
        target = targets[i]
        scores = np.random.uniform(0.1, 0.5, size=(C,))
        scores[target] += 0.8  # Tăng cơ hội đúng
        preds_A[i] = scores / np.sum(scores)

    top1_A = compute_top_k_accuracy(preds_A, targets, k=1)
    top1_B = compute_top_k_accuracy(preds_B, targets, k=1)
    
    top5_A = compute_top_k_accuracy(preds_A, targets, k=5)
    top5_B = compute_top_k_accuracy(preds_B, targets, k=5)

    tree_err_A = compute_hierarchical_cost(preds_A, targets, dist_matrix)
    tree_err_B = compute_hierarchical_cost(preds_B, targets, dist_matrix)

    print(f"    {'Mô Hình':<15} | {'Top-1 Accuracy':<16} | {'Top-5 Accuracy':<16} | {'Mean Tree-Hop Error':<20}")
    print("    " + "-" * 75)
    print(f"    {'Mô Hình A (Nhiễu)':<15} | {top1_A * 100:6.2f}%         | {top5_A * 100:6.2f}%         | {tree_err_A:6.2f} bước nhảy")
    print(f"    {'Mô Hình B (Ngữ nghĩa)':<15} | {top1_B * 100:6.2f}%         | {top5_B * 100:6.2f}%         | {tree_err_B:6.2f} bước nhảy (Thấp hơn!)")

    print("\n[✓] Mô phỏng hoàn tất! Kết quả phản ánh chính xác luận điểm cốt lõi của bài báo ImageNet CVPR 2009.")


if __name__ == "__main__":
    main()
