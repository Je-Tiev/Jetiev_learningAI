# Nhật ký học tập: Neural Network Frameworks

> Folder cá nhân ghi lại quá trình học **Lesson 05 - Frameworks** từ AI for Beginners.

## 📂 Cấu trúc folder

```
05-Frameworks/
├── theory/                 
│   ├── 01-frameworks-overview.md   
│   ├── 02-tensors-and-autograd.md  
│   ├── 03-low-level-api.md          
│   ├── 04-high-level-api.md         
│   └── 05-overfitting.md           
├── labs/                          
│   ├── lab01-pytorch-basics.ipynb   
│   ├── lab02-keras-basics.ipynb     
│   ├── lab03-tf-low-level.ipynb     
│   ├── lab04-mnist-compare.ipynb    
│   └── lab05-iris-challenge.ipynb   
└── README.md                           
```

## Lộ trình học (5 Steps)

### Step 1: Nắm lý thuyết nền tảng
- [ ] Đọc `README.md` gốc của bài học
- [ ] Viết `theory/01-frameworks-overview.md` — tổng quan frameworks
- [ ] Viết `theory/02-tensors-and-autograd.md` — tensors & autograd

### Step 2: Thực hành PyTorch (Low-level → High-level)
- [ ] Đọc & chạy `IntroPyTorch.ipynb` gốc
- [ ] Làm `labs/lab01-pytorch-basics.ipynb` — tự code lại từ đầu
- [ ] Viết `theory/03-low-level-api.md` — ghi lại hiểu biết về low-level API

### Step 3: Thực hành TensorFlow + Keras
- [ ] Đọc & chạy `IntroKerasTF.ipynb` gốc (low-level TF)
- [ ] Đọc & chạy `IntroKeras.ipynb` gốc (high-level Keras)
- [ ] Làm `labs/lab02-keras-basics.ipynb` — tự code Keras từ đầu
- [ ] Làm `labs/lab03-tf-low-level.ipynb` — tự code TF low-level
- [ ] Viết `theory/04-high-level-api.md` — so sánh high-level APIs

### Step 4: Bài tập tổng hợp & so sánh
- [ ] Làm `labs/lab04-mnist-compare.ipynb` — cùng bài toán, 2 framework
- [ ] Làm `labs/lab05-iris-challenge.ipynb` — classification challenge
- [ ] Hoàn thành `lab/LabFrameworks.ipynb` gốc (bài tập chính thức)

### Step 5: Overfitting & tổng kết
- [ ] Viết `theory/05-overfitting.md` — lý thuyết overfitting
- [ ] Thí nghiệm overfitting trên MNIST (thêm vào lab04)
- [ ] Cập nhật `notes/learning-log.md` — tổng kết

## Nguyên tắc học
1. **Đọc trước, code sau** — hiểu lý thuyết rồi mới thực hành
2. **Tự viết lại, không copy** — code lại từ đầu để hiểu sâu
3. **Ghi chú mọi thứ** — lỗi gặp, cách fix, insight mới
4. **So sánh đối chiếu** — luôn so sánh PyTorch vs TensorFlow/Keras
