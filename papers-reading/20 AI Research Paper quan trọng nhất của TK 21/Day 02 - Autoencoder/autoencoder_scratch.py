"""
================================================================================
BÀI BÁO: Reducing the Dimensionality of Data with Neural Networks (Hinton & Salakhutdinov, Science 2006)
TỆP MÃ NGUỒN: autoencoder_scratch.py
MÔ TẢ: Cài đặt mô hình Deep Autoencoder từ con số 0 (From Scratch) bằng Python thuần và NumPy.
       Bao gồm đầy đủ:
       1. Các hàm kích hoạt & Đạo hàm giải tích (Sigmoid, ReLU, Linear).
       2. Các hàm mất mát (MSE, Binary Cross-Entropy).
       3. Tầng kết nối đầy đủ (Dense Layer) với Backpropagation thủ công.
       4. Cỗ máy Boltzmann Giới hạn (Restricted Boltzmann Machine - RBM) với giải thuật Contrastive Divergence (CD-1).
       5. Quy trình 3 bước chuẩn của bài báo: Greedy Layer-Wise Pre-training -> Unrolling -> Fine-tuning.
================================================================================
"""

import numpy as np
import time


# ==============================================================================
# PHẦN 1: CÁC HÀM TOÁN HỌC & ĐẠO HÀM (ACTIVATION FUNCTIONS)
# ==============================================================================

def sigmoid(x):
    """
    Hàm kích hoạt Sigmoid: sigma(x) = 1 / (1 + exp(-x))
    Kẹp giá trị đầu vào để tránh tràn số học (numerical overflow).
    """
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def sigmoid_derivative(output):
    """
    Đạo hàm của hàm Sigmoid theo giá trị đầu ra y = sigmoid(x):
    d(sigma)/dx = sigma(x) * (1 - sigma(x)) = y * (1 - y)
    """
    return output * (1.0 - output)


def relu(x):
    """Hàm kích hoạt ReLU: f(x) = max(0, x)"""
    return np.maximum(0, x)


def relu_derivative(output):
    """Đạo hàm của ReLU: 1 nếu output > 0, ngược lại 0"""
    return (output > 0).astype(np.float64)


def linear(x):
    """Hàm đồng nhất (Linear / Identity): f(x) = x"""
    return x


def linear_derivative(output):
    """Đạo hàm của hàm Linear: luôn bằng 1"""
    return np.ones_like(output)


# ==============================================================================
# PHẦN 2: CÁC HÀM MẤT MÁT (LOSS FUNCTIONS)
# ==============================================================================

class MSELoss:
    """Mean Squared Error Loss (Sai số bình phương trung bình)"""
    @staticmethod
    def forward(y_pred, y_true):
        # L = 0.5 * mean(sum((y_pred - y_true)^2))
        return 0.5 * np.mean(np.sum((y_pred - y_true) ** 2, axis=1))

    @staticmethod
    def gradient(y_pred, y_true):
        # dL/dy_pred = (y_pred - y_true) / batch_size
        batch_size = y_pred.shape[0]
        return (y_pred - y_true) / batch_size


class BCELoss:
    """Binary Cross-Entropy Loss (dùng cho ảnh nhị phân hoặc pixel [0, 1])"""
    @staticmethod
    def forward(y_pred, y_true):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        loss = - np.mean(np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred), axis=1))
        return loss

    @staticmethod
    def gradient(y_pred, y_true):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        batch_size = y_pred.shape[0]
        # Khi kết hợp với đầu ra Sigmoid, dL/dz = (y_pred - y_true) / batch_size
        return ((y_pred - y_true) / (y_pred * (1.0 - y_pred))) / batch_size


# ==============================================================================
# PHẦN 3: TẦNG NƠ-RON LIÊN KẾT ĐẦY ĐỦ (DENSE LAYER VỚI BACKPROP)
# ==============================================================================

class DenseLayer:
    """
    Tầng nơ-ron truyền thẳng (Fully Connected Layer):
    z = X * W + b
    a = activation(z)
    """
    def __init__(self, in_features, out_features, activation="sigmoid"):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation

        # Khởi tạo trọng số Xavier / Glorot
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros((1, out_features))

        # Lưu trữ trạng thái cho lan truyền ngược (Backpropagation)
        self.input_cache = None
        self.output_cache = None

        # Bộ đệm cập nhật Gradient (cho thuật toán Momentum)
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)

        # Chọn hàm kích hoạt
        if activation == "sigmoid":
            self.act_fn = sigmoid
            self.act_deriv = sigmoid_derivative
        elif activation == "relu":
            self.act_fn = relu
            self.act_deriv = relu_derivative
        elif activation == "linear":
            self.act_fn = linear
            self.act_deriv = linear_derivative
        else:
            raise ValueError(f"Activation {activation} không được hỗ trợ!")

    def forward(self, X):
        """Lan truyền xuôi qua tầng nơ-ron"""
        self.input_cache = X
        z = np.dot(X, self.W) + self.b
        self.output_cache = self.act_fn(z)
        return self.output_cache

    def backward(self, grad_output):
        """
        Lan truyền ngược tính đạo hàm:
        grad_output: dL / da
        Trả về: dL / dX để truyền tiếp về các tầng trước.
        """
        # dL / dz = (dL / da) * (da / dz)
        grad_z = grad_output * self.act_deriv(self.output_cache)

        # dL / dW = X^T * grad_z
        self.grad_W = np.dot(self.input_cache.T, grad_z)

        # dL / db = sum(grad_z, axis=0)
        self.grad_b = np.sum(grad_z, axis=0, keepdims=True)

        # dL / dX = grad_z * W^T
        grad_input = np.dot(grad_z, self.W.T)
        return grad_input

    def update_params(self, lr=0.01, momentum=0.9, weight_decay=1e-4):
        """Cập nhật tham số theo Gradient Descent có Momentum và L2 Regularization"""
        self.v_W = momentum * self.v_W - lr * (self.grad_W + weight_decay * self.W)
        self.v_b = momentum * self.v_b - lr * self.grad_b

        self.W += self.v_W
        self.b += self.v_b


# ==============================================================================
# PHẦN 4: RESTRICTED BOLTZMANN MACHINE (RBM) VỚI CONTRASTIVE DIVERGENCE (CD-1)
# ==============================================================================

class RBM:
    """
    Cỗ máy Boltzmann Giới hạn (Restricted Boltzmann Machine):
    Mô hình đồ thị xác suất 2 tầng không có liên kết nội tầng (Bipartite).
    Năng lượng: E(v, h) = - v^T * W * h - b_v^T * v - b_h^T * h
    """
    def __init__(self, num_visible, num_hidden, is_top_linear=False):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.is_top_linear = is_top_linear  # Dùng cho code layer (Gaussian/Linear)

        # Khởi tạo trọng số ngẫu nhiên nhỏ từ phân phối chuẩn N(0, 0.01^2) như trong bài báo
        self.W = np.random.randn(num_visible, num_hidden) * 0.01
        self.b_v = np.zeros((1, num_visible))  # Bias cho visible units
        self.b_h = np.zeros((1, num_hidden))   # Bias cho hidden units

        # Momentum buffers
        self.v_W = np.zeros_like(self.W)
        self.v_b_v = np.zeros_like(self.b_v)
        self.v_b_h = np.zeros_like(self.b_h)

    def sample_hidden(self, v):
        """
        Pha xuôi: Tính xác suất kích hoạt của tầng ẩn h khi biết visible v:
        P(h_j = 1 | v) = sigmoid(b_h + v * W)
        """
        activation = np.dot(v, self.W) + self.b_h
        if self.is_top_linear:
            # Tầng code layer ở đỉnh có thể là biến liên tục (Gaussian)
            h_prob = activation
            h_sample = h_prob + np.random.randn(*h_prob.shape)
        else:
            h_prob = sigmoid(activation)
            # Lấy mẫu nhị phân ngẫu nhiên (Gibbs Sampling)
            h_sample = (np.random.rand(*h_prob.shape) < h_prob).astype(np.float64)
        return h_prob, h_sample

    def sample_visible(self, h):
        """
        Pha ngược (Tái tạo/Ảo giác): Tính xác suất tầng visible v khi biết hidden h:
        P(v_i = 1 | h) = sigmoid(b_v + h * W^T)
        """
        activation = np.dot(h, self.W.T) + self.b_v
        v_prob = sigmoid(activation)
        v_sample = (np.random.rand(*v_prob.shape) < v_prob).astype(np.float64)
        return v_prob, v_sample

    def contrastive_divergence_step(self, v0, lr=0.1, momentum=0.5, weight_decay=2e-4):
        """
        1 bước Contrastive Divergence (CD-1) theo công thức (2) trong bài báo Hinton 2006:
        Delta W = epsilon * (<v0 * h0>_data - <v1 * h1>_recon)
        """
        batch_size = v0.shape[0]

        # 1. Pha Dương (Positive Phase): Dữ liệu thật lái mạng
        h0_prob, h0_sample = self.sample_hidden(v0)
        # Tương quan dữ liệu: <v0^T * h0>
        positive_grad = np.dot(v0.T, h0_prob)

        # 2. Pha Âm (Negative Phase / Confabulation): Mạng tự tưởng tượng (1 bước Gibbs)
        v1_prob, v1_sample = self.sample_visible(h0_sample)
        h1_prob, h1_sample = self.sample_hidden(v1_prob)
        # Tương quan tái tạo: <v1^T * h1>
        negative_grad = np.dot(v1_prob.T, h1_prob)

        # 3. Tính toán độ biến thiên tham số (Gradients)
        dW = (positive_grad - negative_grad) / batch_size
        db_v = np.mean(v0 - v1_prob, axis=0, keepdims=True)
        db_h = np.mean(h0_prob - h1_prob, axis=0, keepdims=True)

        # 4. Cập nhật có quán tính (Momentum) và suy giảm trọng số (Weight Decay)
        self.v_W = momentum * self.v_W + lr * (dW - weight_decay * self.W)
        self.v_b_v = momentum * self.v_b_v + lr * db_v
        self.v_b_h = momentum * self.v_b_h + lr * db_h

        self.W += self.v_W
        self.b_v += self.v_b_v
        self.b_h += self.v_b_h

        # Sai số tái tạo (Reconstruction Error) trong bước này
        recon_error = np.mean((v0 - v1_prob) ** 2)
        return recon_error


# ==============================================================================
# PHẦN 5: TOÀN BỘ MÔ HÌNH DEEP AUTOENCODER (FROM SCRATCH)
# ==============================================================================

class DeepAutoencoderFromScratch:
    """
    Mô hình Deep Autoencoder hoàn chỉnh:
    Hỗ trợ 2 chế độ:
    1. Pre-training từng tầng bằng RBMs -> Unrolling -> Fine-tuning bằng Backpropagation (Hinton 2006).
    2. Huấn luyện trực tiếp từ đầu bằng Backpropagation thuần NumPy.
    """
    def __init__(self, layer_dims):
        """
        layer_dims: Danh sách kích thước từ Input -> Bottleneck, ví dụ: [64, 32, 16, 2]
        Hệ thống sẽ tự động tạo Decoder đối xứng: [2, 16, 32, 64]
        """
        self.layer_dims = layer_dims
        self.num_stages = len(layer_dims) - 1

        # Tạo danh sách các RBM để pre-train
        self.rbms = []
        for i in range(self.num_stages):
            is_top = (i == self.num_stages - 1)
            rbm = RBM(num_visible=layer_dims[i], 
                      num_hidden=layer_dims[i+1], 
                      is_top_linear=is_top)
            self.rbms.append(rbm)

        # Khởi tạo các tầng Encoder và Decoder
        self.encoder_layers = []
        for i in range(self.num_stages):
            act = "linear" if (i == self.num_stages - 1) else "sigmoid"
            self.encoder_layers.append(DenseLayer(layer_dims[i], layer_dims[i+1], activation=act))

        self.decoder_layers = []
        # Decoder đi ngược lại từ bottleneck về input
        reversed_dims = list(reversed(layer_dims))
        for i in range(self.num_stages):
            # Tầng cuối cùng tái tạo pixel bằng sigmoid [0, 1]
            act = "sigmoid"
            self.decoder_layers.append(DenseLayer(reversed_dims[i], reversed_dims[i+1], activation=act))

    def pretrain_rbms(self, X, epochs=15, lr=0.1, batch_size=32):
        """
        Bước 1: Greedy Layer-Wise Pre-training
        Huấn luyện từng RBM nối tiếp nhau. Đầu ra của RBM trước là đầu vào của RBM sau.
        """
        print("\n" + "="*70)
        print(" BƯỚC 1: GREEDY LAYER-WISE PRE-TRAINING (RBM + CONTRASTIVE DIVERGENCE)")
        print("="*70)

        current_input = X.copy()
        N = X.shape[0]

        for stage, rbm in enumerate(self.rbms):
            print(f"\n---> Đang huấn luyện RBM tầng {stage + 1}/{self.num_stages}: "
                  f"[{rbm.num_visible} -> {rbm.num_hidden}] (Top Linear: {rbm.is_top_linear})")

            for epoch in range(1, epochs + 1):
                # Xáo trộn dữ liệu
                indices = np.random.permutation(N)
                epoch_recon_error = 0.0
                num_batches = int(np.ceil(N / batch_size))

                # Điều chỉnh momentum theo bài báo: 0.5 ở 5 epoch đầu, 0.9 ở các epoch sau
                momentum = 0.5 if epoch <= 5 else 0.9

                for b in range(num_batches):
                    batch_idx = indices[b * batch_size : min((b + 1) * batch_size, N)]
                    v_batch = current_input[batch_idx]
                    err = rbm.contrastive_divergence_step(v_batch, lr=lr, momentum=momentum)
                    epoch_recon_error += err

                epoch_recon_error /= num_batches
                if epoch % 5 == 0 or epoch == 1:
                    print(f"     Epoch [{epoch:2d}/{epochs:2d}] - Sai số tái tạo RBM: {epoch_recon_error:.6f}")

            # Sau khi RBM này học xong, biến đổi toàn bộ dữ liệu qua tầng ẩn để làm đầu vào cho RBM kế tiếp
            h_prob, _ = rbm.sample_hidden(current_input)
            current_input = h_prob

        print("\n===> Hoàn thành giai đoạn Pre-training cho toàn bộ các tầng RBM!")

    def unroll_rbms(self):
        """
        Bước 2: Unrolling (Mở bung mạng)
        Sao chép trọng số đã học từ các RBM vào các tầng Encoder và Decoder tương ứng.
        Decoder ban đầu nhận trọng số là ma trận chuyển vị: W_decoder = W_encoder^T
        """
        print("\n" + "="*70)
        print(" BƯỚC 2: UNROLLING RBMs THÀNH DEEP AUTOENCODER ĐỐI XỨNG")
        print("="*70)

        # Gán trọng số vào Encoder
        for i, rbm in enumerate(self.rbms):
            self.encoder_layers[i].W = rbm.W.copy()
            self.encoder_layers[i].b = rbm.b_h.copy()
            print(f" [Encoder Tầng {i+1}] Nhận W từ RBM {i+1}: {self.encoder_layers[i].W.shape}")

        # Gán trọng số vào Decoder (đối xứng, dùng W^T)
        for i in range(self.num_stages):
            # RBM tương ứng theo thứ tự đảo ngược
            source_rbm = self.rbms[self.num_stages - 1 - i]
            self.decoder_layers[i].W = source_rbm.W.T.copy()
            self.decoder_layers[i].b = source_rbm.b_v.copy()
            print(f" [Decoder Tầng {i+1}] Khởi tạo bằng W^T từ RBM {self.num_stages - i}: {self.decoder_layers[i].W.shape}")

        print("===> Mạng Deep Autoencoder đã sẵn sàng cho bước Fine-tuning toàn cục!")

    def encode(self, X):
        """Nén dữ liệu đầu vào qua các tầng Encoder để lấy Latent Code"""
        out = X
        for layer in self.encoder_layers:
            out = layer.forward(out)
        return out

    def decode(self, Z):
        """Giải nén Latent Code qua các tầng Decoder để tái tạo dữ liệu"""
        out = Z
        for layer in self.decoder_layers:
            out = layer.forward(out)
        return out

    def forward(self, X):
        """Lan truyền xuôi toàn bộ Autoencoder: x -> z -> x_hat"""
        z = self.encode(X)
        x_hat = self.decode(z)
        return x_hat, z

    def finetune_backprop(self, X, epochs=30, lr=0.01, batch_size=32, loss_type="mse"):
        """
        Bước 3: Global Fine-Tuning bằng Backpropagation thuần
        """
        print("\n" + "="*70)
        print(f" BƯỚC 3: GLOBAL FINE-TUNING BẰNG BACKPROPAGATION (Loss: {loss_type.upper()})")
        print("="*70)

        LossClass = BCELoss if loss_type == "bce" else MSELoss
        N = X.shape[0]

        history_loss = []
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(N)
            epoch_loss = 0.0
            num_batches = int(np.ceil(N / batch_size))

            for b in range(num_batches):
                batch_idx = indices[b * batch_size : min((b + 1) * batch_size, N)]
                X_batch = X[batch_idx]

                # 1. Forward Pass
                x_hat, _ = self.forward(X_batch)
                batch_loss = LossClass.forward(x_hat, X_batch)
                epoch_loss += batch_loss

                # 2. Backward Pass (Lan truyền ngược đạo hàm)
                # Đạo hàm hàm mất mát dL / dx_hat
                grad = LossClass.gradient(x_hat, X_batch)

                # Ngược qua Decoder
                for layer in reversed(self.decoder_layers):
                    grad = layer.backward(grad)

                # Ngược tiếp qua Encoder
                for layer in reversed(self.encoder_layers):
                    grad = layer.backward(grad)

                # 3. Cập nhật trọng số
                for layer in self.decoder_layers:
                    layer.update_params(lr=lr)
                for layer in self.encoder_layers:
                    layer.update_params(lr=lr)

            epoch_loss /= num_batches
            history_loss.append(epoch_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"     Epoch [{epoch:2d}/{epochs:2d}] - Loss: {epoch_loss:.6f}")

        elapsed = time.time() - start_time
        print(f"===> Hoàn thành Fine-tuning trong {elapsed:.2f}s! Loss ban đầu: {history_loss[0]:.6f} -> Loss cuối: {history_loss[-1]:.6f}")
        return history_loss


# ==============================================================================
# PHẦN 6: TẠO DỮ LIỆU ĐA TẠP PHI TUYẾN & THỬ NGHIỆM ĐÁNH GIÁ
# ==============================================================================

def generate_synthetic_data(num_samples=600, random_seed=42):
    """
    Tạo bộ dữ liệu nhân tạo phi tuyến (Synthetic Manifold Data):
    Sinh ra từ 2 biến tiềm ẩn thực sự z = [t1, t2] (ví dụ: góc quay và độ co giãn),
    nhưng được biến đổi phi tuyến thành vector 64 chiều (tương tự bức ảnh 8x8).
    Điều này tái hiện hoàn hảo giả thuyết của bài báo Hinton 2006!
    """
    np.random.seed(random_seed)
    # 2 biến tiềm ẩn nội tại nằm trong [-pi, pi]
    t1 = np.random.uniform(-np.pi, np.pi, size=(num_samples, 1))
    t2 = np.random.uniform(-np.pi, np.pi, size=(num_samples, 1))

    # Ánh xạ phi tuyến lên 64 chiều bằng hàm lượng giác phức tạp
    X = np.zeros((num_samples, 64))
    for i in range(64):
        freq1 = (i % 8 + 1) * 0.5
        freq2 = (i // 8 + 1) * 0.5
        X[:, i] = np.sin(freq1 * t1[:, 0]) * np.cos(freq2 * t2[:, 0]) + 0.5 * np.cos(freq1 * t2[:, 0])

    # Chuẩn hóa về đoạn [0.05, 0.95] để thích hợp cho hàm sigmoid
    X_min = X.min()
    X_max = X.max()
    X = (X - X_min) / (X_max - X_min)
    X = 0.05 + 0.9 * X
    return X, np.hstack([t1, t2])


def print_ascii_comparison(original, reconstructed, size=8):
    """Vẽ minh họa so sánh mẫu dữ liệu gốc và mẫu tái tạo bằng ký tự ASCII"""
    chars = " .:-=+*#%@"
    def to_ascii(arr):
        grid = []
        for r in range(size):
            row_str = ""
            for c in range(size):
                val = arr[r * size + c]
                idx = int(np.clip(val * (len(chars) - 1), 0, len(chars) - 1))
                row_str += chars[idx] * 2
            grid.append(row_str)
        return grid

    orig_ascii = to_ascii(original)
    recon_ascii = to_ascii(reconstructed)

    print("\n   [Ảnh Gốc 8x8]               [Ảnh Tái Tạo Bằng Autoencoder]")
    for o, r in zip(orig_ascii, recon_ascii):
        print(f"   {o}       {r}")


# ==============================================================================
# HÀM MAIN: THỰC THI TOÀN BỘ CHƯƠNG TRÌNH
# ==============================================================================

if __name__ == "__main__":
    print("="*75)
    print(" KHỞI ĐỘNG CHƯƠNG TRÌNH: DEEP AUTOENCODER FROM SCRATCH (HINTON 2006)")
    print("="*75)

    # 1. Tạo tập dữ liệu thử nghiệm
    print("\n[1] Đang sinh tập dữ liệu đa tạp phi tuyến tính (Synthetic Manifold)...")
    X_data, true_latents = generate_synthetic_data(num_samples=800, random_seed=42)
    X_train = X_data[:600]
    X_test = X_data[600:]
    print(f"    - Kích thước tập huấn luyện: {X_train.shape} (600 mẫu, 64 chiều)")
    print(f"    - Kích thước tập kiểm thử:   {X_test.shape} (200 mẫu, 64 chiều)")

    # 2. Định nghĩa kiến trúc mạng sâu (Deep Architecture)
    # Nén: 64 chiều -> 32 chiều -> 16 chiều -> 2 chiều (Bottleneck) -> 16 -> 32 -> 64
    architecture = [64, 32, 16, 2]
    print(f"\n[2] Thiết lập kiến trúc mạng: {' -> '.join(map(str, architecture))} (Code Layer: 2 chiều)")
    model = DeepAutoencoderFromScratch(layer_dims=architecture)

    # 3. Bước 1: Greedy Layer-Wise Pre-training từng tầng RBM
    model.pretrain_rbms(X_train, epochs=15, lr=0.08, batch_size=32)

    # 4. Bước 2: Unrolling thành Autoencoder đối xứng
    model.unroll_rbms()

    # 5. Bước 3: Global Fine-Tuning bằng Backpropagation thuần
    history = model.finetune_backprop(X_train, epochs=25, lr=0.05, batch_size=32, loss_type="mse")

    # 6. Đánh giá chất lượng trên tập kiểm thử (Test Set)
    print("\n" + "="*70)
    print(" ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH TRÊN TẬP KIỂM THỬ (TEST SET)")
    print("="*70)
    test_recon, test_latents = model.forward(X_test)
    test_mse = MSELoss.forward(test_recon, X_test)
    print(f" * Sai số tái tạo bình phương trung bình (Test MSE): {test_mse:.6f}")
    print(f" * Dữ liệu đã được nén thành công từ 64 chiều xuống {test_latents.shape[1]} chiều!")

    # 7. Hiển thị so sánh trực quan mẫu ảnh gốc vs ảnh tái tạo
    print("\n[Trực quan hóa một mẫu ngẫu nhiên từ tập kiểm thử]:")
    sample_idx = 7
    print_ascii_comparison(X_test[sample_idx], test_recon[sample_idx], size=8)

    print("\n" + "="*75)
    print(" HOÀN TẤT THỰC THI MÃ NGUỒN FROM SCRATCH THÀNH CÔNG!")
    print("="*75)
