"""
================================================================================
BÀI BÁO: Efficient Estimation of Word Representations in Vector Space (Mikolov et al., ICLR 2013)
TỆP MÃ NGUỒN: word2vec_scratch.py
MÔ TẢ DÀNH CHO NGƯỜI MỚI BẮT ĐẦU:
    Tệp mã nguồn này cài đặt từ con số 0 (From Scratch) bằng thuần Python và NumPy
    toàn bộ các giải thuật và cấu trúc nền tảng của mô hình Word2Vec (Skip-gram với Negative Sampling):

    1. Quản lý Từ Vựng & Phân Phối Lấy Mẫu (Vocabulary & Noise Distribution):
       - Ánh xạ hai chiều Từ <-> Chỉ số nguyên (w2i và i2w).
       - Bảng phân phối xác suất lấy mẫu âm lũy thừa 3/4: P_n(w) ~ U(w)^(3/4).
       - Bộ trượt cửa sổ ngữ cảnh (Context Window Generator).

    2. Thuật Toán Lấy Mẫu Tiêu Cực (Negative Sampling - NEG):
       - Biến bài toán phân loại Softmax V chiều đắt đỏ O(V) thành các bài toán
         hồi quy Logistic nhị phân gọn nhẹ O(K) với K mẫu âm tính.
       - Cài đặt hàm mất mát nhị phân Binary Logistic Loss và hàm Sigmoid chống tràn số.
    
    3. Lan Truyền Ngược Giải Tích (Analytical Gradient Updates):
       - Tự đạo hàm và cập nhật ma trận nhúng đầu vào (W_in) và ma trận nhúng đầu ra (W_out).

    4. Đại Số Ngữ Nghĩa (Word Analogy Arithmetic):
       - Đo độ tương đồng Cosine giữa các vector từ vựng.
       - Giải bài toán tương đồng kinh điển: v(King) - v(Man) + v(Woman) ~= v(Queen).

    5. Khối Đối Chiếu Giá Trị Học Thuật (Academic Value Benchmark):
       - So sánh trực tiếp: Kỷ nguyên Tiền-Word2Vec (One-Hot & Ma trận đồng xuất hiện SVD)
         với Không gian vector liên tục dày đặc của Word2Vec.
       - Chứng minh tại sao One-Hot thất bại trong việc nắm bắt quan hệ ngữ nghĩa.
================================================================================
"""

import time
import numpy as np


# ==============================================================================
# PHẦN 1: QUẢN LÝ TỪ VỰNG & PHÂN PHỐI LẤY MẪU TIÊU CỰC (VOCABULARY)
# ==============================================================================

class Vocabulary:
    """
    Quản lý từ điển, chuyển đổi văn bản và sinh bảng phân phối lấy mẫu âm 3/4.
    """
    def __init__(self, corpus_text, min_count=1):
        self.min_count = min_count
        self.w2i = {}
        self.i2w = {}
        self.word_counts = {}
        self.total_words = 0
        self.vocab_size = 0
        self.sampling_table = []

        self._build_vocab(corpus_text)
        self._build_negative_sampling_table()

    def _build_vocab(self, corpus_text):
        """Tách từ, đếm tần suất và thiết lập ánh xạ từ vựng."""
        words = corpus_text.lower().replace(".", " ").replace(",", " ").split()
        for w in words:
            self.word_counts[w] = self.word_counts.get(w, 0) + 1
            self.total_words += 1

        # Lọc các từ thỏa mãn tần suất tối thiểu
        idx = 0
        for w, c in self.word_counts.items():
            if c >= self.min_count:
                self.w2i[w] = idx
                self.i2w[idx] = w
                idx += 1

        self.vocab_size = len(self.w2i)

    def _build_negative_sampling_table(self, table_size=100000):
        """
        Xây dựng bảng lấy mẫu Unigram phân phối lũy thừa 3/4 theo Mikolov et al.:
        P_n(w) = U(w)^0.75 / sum_w' (U(w')^0.75)
        """
        powered_counts = np.array([self.word_counts[self.i2w[i]] ** 0.75 for i in range(self.vocab_size)])
        probs = powered_counts / np.sum(powered_counts)

        # Tạo bảng tra cứu nhanh O(1)
        cumulative = np.cumsum(probs)
        table = np.zeros(table_size, dtype=np.int32)
        idx = 0
        for i in range(table_size):
            r = (i + 1) / table_size
            while idx < self.vocab_size - 1 and r > cumulative[idx]:
                idx += 1
            table[i] = idx

        self.sampling_table = table

    def get_negative_samples(self, target_idx, k=5):
        """Lấy ngẫu nhiên K mẫu âm tính từ bảng phân phối, loại trừ từ mục tiêu."""
        neg_samples = []
        while len(neg_samples) < k:
            rand_idx = self.sampling_table[np.random.randint(len(self.sampling_table))]
            if rand_idx != target_idx:
                neg_samples.append(rand_idx)
        return neg_samples


def generate_skipgram_training_pairs(corpus_text, vocab, window_size=2):
    """
    Sinh các cặp huấn luyện (Từ trung tâm, Từ ngữ cảnh) trong cửa sổ trượt:
    [w_{t-C}, ..., w_{t-1}, w_t (trung tâm), w_{t+1}, ..., w_{t+C}]
    """
    words = corpus_text.lower().replace(".", " ").replace(",", " ").split()
    word_indices = [vocab.w2i[w] for w in words if w in vocab.w2i]
    pairs = []

    for center_pos, center_idx in enumerate(word_indices):
        # Xác định phạm vi cửa sổ ngữ cảnh
        start = max(0, center_pos - window_size)
        end = min(len(word_indices), center_pos + window_size + 1)

        for context_pos in range(start, end):
            if context_pos != center_pos:
                context_idx = word_indices[context_pos]
                pairs.append((center_idx, context_idx))

    return pairs


# ==============================================================================
# PHẦN 2: MÔ HÌNH WORD2VEC SKIP-GRAM TỪ CON SỐ 0
# ==============================================================================

class Word2VecScratch:
    """
    Triển khai mô hình Skip-Gram với Negative Sampling bằng thuần NumPy:
    
    Ma trận trọng số:
    - W_in  (Vocab_size, Embed_dim): Ma trận nhúng biểu diễn từ trung tâm.
    - W_out (Vocab_size, Embed_dim): Ma trận nhúng biểu diễn từ ngữ cảnh.
    """
    def __init__(self, vocab_size, embed_dim=16, learning_rate=0.025):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = learning_rate

        # Khởi tạo ma trận ngẫu nhiên đều nhỏ theo bài báo gốc [-0.5/D, 0.5/D]
        bound = 0.5 / embed_dim
        self.W_in = np.random.uniform(-bound, bound, (vocab_size, embed_dim)).astype(np.float32)
        self.W_out = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    @staticmethod
    def sigmoid(z):
        """Hàm kích hoạt Sigmoid có kẹp giá trị chống tràn số học (overflow)."""
        z_clipped = np.clip(z, -15.0, 15.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def train_pair(self, center_idx, context_idx, neg_indices):
        """
        Huấn luyện một cặp từ (center, context) cùng với K mẫu âm tính:
        
        Mục tiêu cực đại hóa:
            log sigma(v'_context^T * v_center) + sum_k [ log sigma(-v'_neg^T * v_center) ]
            
        Đạo hàm giải tích:
            dL/d(dot_pos) = sigma(v'_context^T * v_center) - 1
            dL/d(dot_neg) = sigma(v'_neg^T * v_center)
        """
        v_center = self.W_in[center_idx]  # Shape: (Embed_dim,)

        # 1. Đánh giá Mẫu Dương (Positive Sample)
        v_pos = self.W_out[context_idx]
        dot_pos = np.dot(v_pos, v_center)
        sig_pos = self.sigmoid(dot_pos)
        grad_pos = sig_pos - 1.0  # (sigma - 1)
        loss = -np.log(sig_pos + 1e-12)

        # 2. Đánh giá Các Mẫu Âm (Negative Samples)
        v_negs = self.W_out[neg_indices]  # Shape: (K, Embed_dim)
        dot_negs = np.dot(v_negs, v_center)
        sig_negs = self.sigmoid(dot_negs)
        grad_negs = sig_negs  # (sigma - 0)
        loss += -np.sum(np.log(1.0 - sig_negs + 1e-12))

        # 3. Tích lũy Gradient để cập nhật vector từ trung tâm W_in
        grad_center = grad_pos * v_pos + np.sum(grad_negs[:, None] * v_negs, axis=0)

        # 4. Cập nhật ma trận đầu ra W_out
        self.W_out[context_idx] -= self.lr * (grad_pos * v_center)
        for i, neg_idx in enumerate(neg_indices):
            self.W_out[neg_idx] -= self.lr * (grad_negs[i] * v_center)

        # 5. Cập nhật ma trận đầu vào W_in
        self.W_in[center_idx] -= self.lr * grad_center

        return loss

    def get_word_vector(self, word_idx):
        """Trả về vector biểu diễn cuối cùng (thường lấy W_in hoặc tổng W_in + W_out)."""
        return self.W_in[word_idx]


# ==============================================================================
# PHẦN 3: ĐỘ TƯƠNG ĐỒNG COSINE & BỘ GIẢI ĐẠI SỐ NGỮ NGHĨA
# ==============================================================================

def cosine_similarity(v1, v2):
    """Tính độ tương đồng Cosine giữa 2 vector: cos(u, v) = (u . v) / (||u|| * ||v||)"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def solve_analogy(w1, w2, w3, model, vocab, top_k=3):
    """
    Giải bài toán suy luận tương đồng ngữ nghĩa:
    'w1 đối với w2 thì tương tự như w3 đối với TỪ NÀO?'
    Vector mục tiêu: v_target = v(w2) - v(w1) + v(w3)
    (Ví dụ: King - Man + Woman = Queen)
    """
    for w in [w1, w2, w3]:
        if w not in vocab.w2i:
            return [(f"Từ '{w}' không có trong từ điển", 0.0)]

    idx1 = vocab.w2i[w1]
    idx2 = vocab.w2i[w2]
    idx3 = vocab.w2i[w3]

    v1 = model.get_word_vector(idx1)
    v2 = model.get_word_vector(idx2)
    v3 = model.get_word_vector(idx3)

    target_vec = v2 - v1 + v3

    # Tìm các từ có độ tương đồng Cosine gần nhất
    exclude_indices = {idx1, idx2, idx3}
    scores = []

    for i in range(vocab.vocab_size):
        if i not in exclude_indices:
            sim = cosine_similarity(target_vec, model.get_word_vector(i))
            scores.append((vocab.i2w[i], sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ==============================================================================
# PHẦN 4: ĐỐI CHIẾU GIÁ TRỊ HỌC THUẬT (ACADEMIC VALUE BENCHMARK)
# ==============================================================================

def run_academic_value_comparison():
    """
    Đối chiếu toàn diện 2 phương pháp tiếp cận:
    1. Kỷ nguyên Tiền-Word2Vec (One-Hot Encoding & SVD Co-occurrence):
       - Đặc điểm: Vector rời rạc, số chiều bằng kích thước từ điển V (rất thưa).
       - Tích vô hướng giữa 2 từ bất kỳ luôn bằng 0 (dot(cat, dog) = 0).
       - Hoàn toàn không nắm bắt được mối quan hệ tương quan ngữ nghĩa.
       
    2. Đột phá Word2Vec (Mikolov et al., 2013 - Dense Continuous Embedding):
       - Vector liên tục, dày đặc, số chiều nhỏ (D << V).
       - Tự động học quan hệ ngữ nghĩa qua khoảng cách hình học góc Cosine.
       - Cho phép thực hiện các phép toán đại số vector vi diệu (King - Man + Woman = Queen).
    """
    print("\n" + "=" * 80)
    print("PHÂN TÍCH SO SÁNH GIÁ TRỊ HỌC THUẬT: TIỀN-WORD2VEC vs ĐỘT PHÁ WORD2VEC (2013)")
    print("=" * 80)

    # Đoạn văn bản mẫu phong phú phản ánh các mối quan hệ ngữ nghĩa rõ rệt
    corpus = """
    king rules the royal empire with the queen
    queen is the royal woman ruler beside the king
    man is a royal king who rules the court
    woman is a royal queen who rules the court
    apple is a fresh sweet fruit from the garden
    orange is a juicy citrus fruit from the garden
    banana is a yellow sweet fruit from the garden
    paris is the beautiful capital city of france
    rome is the historic capital city of italy
    hanoi is the charming capital city of vietnam
    tokyo is the modern capital city of japan
    france has the capital city of paris
    italy has the capital city of rome
    vietnam has the capital city of hanoi
    japan has the capital city of tokyo
    """

    vocab = Vocabulary(corpus, min_count=1)
    print(f"[*] Kích thước từ vựng (Vocabulary Size): V = {vocab.vocab_size} từ.")

    # --------------------------------------------------------------------------
    # 1. Phương pháp Tiền-Word2Vec: One-Hot Encoding
    # --------------------------------------------------------------------------
    one_hot_matrix = np.eye(vocab.vocab_size)
    king_idx = vocab.w2i["king"]
    queen_idx = vocab.w2i["queen"]
    apple_idx = vocab.w2i["apple"]

    cos_onehot_king_queen = cosine_similarity(one_hot_matrix[king_idx], one_hot_matrix[queen_idx])
    cos_onehot_king_apple = cosine_similarity(one_hot_matrix[king_idx], one_hot_matrix[apple_idx])

    # --------------------------------------------------------------------------
    # 2. Phương pháp Word2Vec: Skip-Gram với Negative Sampling
    # --------------------------------------------------------------------------
    pairs = generate_skipgram_training_pairs(corpus, vocab, window_size=3)
    embed_dim = 16
    model = Word2VecScratch(vocab_size=vocab.vocab_size, embed_dim=embed_dim, learning_rate=0.05)

    np.random.seed(42)
    epochs = 40
    print(f"[*] Huấn luyện Word2Vec Skip-gram trên {len(pairs)} cặp từ ({epochs} epochs)...")

    for ep in range(epochs):
        np.random.shuffle(pairs)
        total_loss = 0.0
        for center, context in pairs:
            negs = vocab.get_negative_samples(context, k=5)
            loss = model.train_pair(center, context, negs)
            total_loss += loss

    # Tính độ tương đồng sau khi huấn luyện Word2Vec
    v_king = model.get_word_vector(king_idx)
    v_queen = model.get_word_vector(queen_idx)
    v_apple = model.get_word_vector(apple_idx)

    cos_w2v_king_queen = cosine_similarity(v_king, v_queen)
    cos_w2v_king_apple = cosine_similarity(v_king, v_apple)

    # Thử nghiệm phép suy luận tương đồng: king - man + woman = ?
    analogy_results = solve_analogy("man", "king", "woman", model, vocab, top_k=3)

    print("\n[+] BẢNG ĐỐI CHIẾU ĐỊNH LƯỢNG:")
    print(f"    {'Tiêu Chí So Sánh':<32} | {'Tiền-Word2Vec (One-Hot)':<25} | {'Word2Vec (Skip-Gram NEG)':<28}")
    print("    " + "-" * 90)
    print(f"    {'Số chiều biểu diễn (Dimensions)':<32} | {vocab.vocab_size:<25} | {embed_dim} (Nén gọn {vocab.vocab_size/embed_dim:.1f}x)")
    print(f"    {'Độ tương đồng: king <-> queen':<32} | {cos_onehot_king_queen:6.2f} (Trực giao = 0.0)    | {cos_w2v_king_queen:6.2f} (Rất gần gũi!)")
    print(f"    {'Độ tương đồng: king <-> apple':<32} | {cos_onehot_king_apple:6.2f} (Trực giao = 0.0)    | {cos_w2v_king_apple:6.2f} (Xa lạ)")
    top_pred, top_score = analogy_results[0]
    print(f"    {'Phép toán: King - Man + Woman':<32} | {'Không thể thực hiện':<25} | ~= {top_pred} (Cosine: {top_score:.2f})")

    print("\n[+] BẢNG SO SÁNH GIÁ TRỊ HỌC THUẬT & BƯỚC NGOẶT LỊCH SỬ:")
    print("┌──────────────────────────┬─────────────────────────────────────┬─────────────────────────────────────┐")
    print("│ Tiêu Chí So Sánh         │ Kỷ Nguyên Tiền-Word2Vec (Trước 2013)│ Đột Phá Word2Vec (Mikolov et al.)   │")
    print("├──────────────────────────┼─────────────────────────────────────┼─────────────────────────────────────┤")
    print("│ Bản chất biểu diễn       │ Rời rạc, thưa thớt (One-Hot, BoW)   │ Dày đặc, liên tục (Dense Vector)    │")
    print("│ Khoảng cách ngữ nghĩa    │ Mọi từ trực giao nhau (dot = 0)     │ Cosine phản ánh đúng độ tương đồng  │")
    print("│ Đại số ngữ nghĩa         │ Bất khả thi                         │ Tuyệt vời (King - Man + Woman=Queen)│")
    print("│ Độ phức tạp mỗi bước     │ O(V) hoặc O(V^2) cực kỳ đắt đỏ      │ O(K) với K mẫu âm (Negative Sample) │")
    print("│ Khả năng mở rộng (Scale) │ Giới hạn ở tập dữ liệu nhỏ          │ Huấn luyện hàng tỷ từ trong vài giờ │")
    print("│ Di sản lịch sử           │ Mô hình n-gram thống kê rời rạc     │ Nền móng cho GloVe, BERT, và LLMs   │")
    print("└──────────────────────────┴─────────────────────────────────────┴─────────────────────────────────────┘")


# ==============================================================================
# HÀM CHÍNH (MAIN PIPELINE)
# ==============================================================================

def main():
    print("================================================================================")
    print("CHƯƠNG TRÌNH MÔ PHỎNG CỐT LÕI BÀI BÁO WORD2VEC (ICLR 2013) - FROM SCRATCH")
    print("================================================================================")

    # Chạy đối chiếu giá trị học thuật
    run_academic_value_comparison()

    print("\n[✓] Hoàn thành trọn vẹn mô phỏng thuật toán và giá trị học thuật của bài báo Word2Vec (ICLR 2013)!")


if __name__ == "__main__":
    main()
