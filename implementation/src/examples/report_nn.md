# Báo cáo: MLP phân lớp dữ liệu spiral và so sánh GD với GDA

## 1. Tóm tắt mục tiêu
Bộ mã nguồn xây dựng một bài toán phân lớp **3 lớp** trên dữ liệu tổng hợp dạng **spiral** trong không gian 2 chiều, sau đó huấn luyện một **MLP (1 hidden layer)** bằng:
- **GD truyền thống** (learning rate cố định).
- **GDA (Gradient Descent Adaptive Algorithm)** theo tiêu chuẩn “sufficient decrease” kiểu Armijo (giảm bước học khi vi phạm điều kiện giảm đủ).

Kết quả được trực quan hóa bằng đồ thị **Accuracy** và **Loss** theo số vòng lặp (epoch).

---

## 2. Dữ liệu và tiền xử lý

### 2.1. Sinh dữ liệu (synthetic 3-class spiral)
- Số lớp: **C = 3**.
- Số điểm mỗi lớp: **N = 1000**.
- Tổng số mẫu: **N*C = 3000**.
- Số chiều đầu vào: **d0 = 2**.

Mỗi lớp được sinh bằng tham số bán kính `r` tăng tuyến tính và góc `t` tăng tuyến tính theo lớp, cộng nhiễu Gaussian để tạo đường xoắn (spiral) trong 2D.

### 2.2. Chia train/test
Dữ liệu được chia theo `train_test_split` với:
- `TEST_SIZE = 500` → test set có **500 mẫu**.
- Train set có **2500 mẫu**.

Lưu ý: trong vòng lặp huấn luyện, **không xáo trộn dữ liệu** (khối `np.random.permutation` bị comment, thay bằng `np.arange(N)`), vì vậy đây là thực nghiệm “without permutation”.

---

## 3. Mô hình Neural Network (MLP)

### 3.1. Kiến trúc
Mạng là một **2-layer MLP** (tính theo số lớp tuyến tính) với:
- Input layer: kích thước **d0 = 2**.
- Hidden layer: **d1 = 100** nút, kích hoạt **ReLU**.
- Output layer: **d2 = 3** nút (logits), xác suất lớp qua **softmax**.

**Kích thước tham số**
- $W_1 \in \mathbb{R}^{d_0 \times d_1} = \mathbb{R}^{2 \times 100}$
- $b_1 \in \mathbb{R}^{d_1} = \mathbb{R}^{100}$
- $W_2 \in \mathbb{R}^{d_1 \times d_2} = \mathbb{R}^{100 \times 3}$
- $b_2 \in \mathbb{R}^{d_2} = \mathbb{R}^{3}$

### 3.2. Khởi tạo
- $W_1, W_2$ khởi tạo Gaussian nhỏ: `0.01 * randn(...)`.
- $b_1, b_2$ khởi tạo $0$.

### 3.3. Lan truyền xuôi (forward pass)
Với một mini-batch $X \in \mathbb{R}^{m \times 2}$:
1. Hidden pre-activation:
$
Z_1 = XW_1 + B_1 \quad (m \times 100)
$
2. ReLU activation:
$
A_1 = \max(0, Z_1) \quad (m \times 100)
$
3. Output:
$
Z_2 = A_1W_2 + B_2 \quad (m \times 3)
$
1. Softmax:
$
\hat{Y} = \text{softmax}(Z_2) \quad (m \times 3)
$

### 3.4. Hàm mất mát (loss)
Sử dụng **Cross-entropy loss**:
$J \triangleq J(\mathbf{W}, \mathbf{b} ; \mathbf{X}, \mathbf{Y})=-\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^C y_{j i} \log \left(\hat{y}_{j i}\right)$

Trong code, loss được tính bằng cách “index” trực tiếp vào xác suất lớp đúng (không cần one-hot đầy đủ).

### 3.5. Dự đoán và đánh giá
- Dự đoán lớp dùng `argmax` trên **logits** $Z_2$ (không bắt buộc softmax vì argmax của softmax và Output $Z_2$ là như nhau).
- Accuracy được tính trên **test set** sau mỗi epoch.

---

## 4. Huấn luyện bằng GD truyền thống (learning rate cố định)

### 4.1. Thiết lập thực nghiệm
- `NUM_OF_EPOCHES = 200`
- `BATCH_SIZE = 50`
- Learning rate: `ETA = 0.5`

### 4.2. Backpropagation (tóm tắt)
Với softmax + cross-entropy, gradient tại output có dạng:
$
\frac{\partial \mathcal{L}}{\partial Z_2} = \frac{\hat{Y} - Y}{N}
$
với N là kích thước của 1 batch (batch_size)

Code dùng “in-place trick”: trừ 1 vào xác suất lớp đúng, rồi chia cho batch size để ra $E_2$.

Gradient chính:
- $\nabla_{\mathbf{W}_{2}}=\mathbf{A}^{(1)} \mathbf{E}^{(2) T}$
- $\nabla_{\mathbf{b}_{2}}=\sum_{n=1}^N \mathbf{e}_n^{(2)}$
- Lan truyền ngược qua ReLU:
  - $\mathbf{E}^{(1)} =\left(\mathbf{W}_{2} \mathbf{E}^{(2)}\right) \odot f^{\prime}\left(\mathbf{Z}^{(1)}\right)$
- $\nabla_{\mathbf{W}_{1}} =\mathbf{A}_{0} \mathbf{E}^{(1) T}=\mathbf{X E}^{(1) T}$
- $\nabla_{\mathbf{b}_{1}}=\sum_{n=1}^N \mathbf{e}_n^{(1)}$

### 4.3. Cập nhật tham số (GD)
Cập nhật theo learning rate cố định $\eta$:
$
\theta \leftarrow \theta - \eta \nabla \mathcal{L}(\theta)
$

Trong code, $\theta$ là tập $(W_1, b_1, W_2, b_2)$.

---

## 5. Huấn luyện bằng GDA (adaptive step size)

## 5.1. Thuật toán GDA (lý thuyết)
Algorithm 1 đặt:
$
x^{k+1} = P_C\left(x^k - \lambda_k \nabla f(x^k)\right)
$
và điều chỉnh $\lambda$ bằng tiêu chí sufficient decrease:
- Nếu thỏa:
$
f(x^{k+1}) \le f(x^k) - \sigma \langle \nabla f(x^k), x^k - x^{k+1}\rangle
$
thì giữ $\lambda_{k+1}=\lambda_k$,
- Ngược lại giảm $\lambda_{k+1}=\kappa \lambda_k$.

Với **unconstrained** (C = $\mathbb{R}^n$) và bước GD thuần:
$
x^{k+1} = x^k - \lambda_k \nabla f(x^k)
\Rightarrow \langle \nabla f(x^k), x^k - x^{k+1} \rangle = \lambda_k \|\nabla f(x^k)\|^2
$

Nên điều kiện giảm đủ trở thành:
$
f(x^{k+1}) \le f(x^k) - \sigma \lambda_k \|\nabla f(x^k)\|^2
$

### 5.2. GDA trong code được hiện thực như thế nào
Trong hàm `neural_network_with_GDA(...)`, code:
- Khởi tạo `my_lambda = lr` và **giữ biến này xuyên suốt** qua các mini-batch và epoch.
- Sau khi cập nhật tham số bằng `my_lambda`, code **tính loss mới** trên mini-batch, rồi kiểm tra điều kiện:
$
\text{if } \mathcal{L}_{new} > \mathcal{L}_{old} - \sigma\lambda\|\nabla \mathcal{L}\|^2
\text{ then } \lambda \leftarrow \kappa\lambda
$

Trong đó $\|\nabla \mathcal{L}\|^2$ được tính bằng tổng bình phương của mọi gradient trên $W_1,W_2,b_1,b_2$.

### 5.3. Khác biệt quan trọng giữa “Algorithm 1” và code hiện tại
Code đang dùng một biến thể “adaptive shrink” đơn giản và có các sai khác sau:
1. **Không có phép chiếu $P_C$**: tương đương chọn $C=\mathbb{R}^n$, hợp lý cho bài toán không ràng buộc.
2. **Không có cơ chế “reject step” và tính lại $x^{k+1}$**: trong Algorithm 1, nếu điều kiện không đạt thì thường giảm $\lambda$ và **tính lại bước** (một dạng line search).  
   Trong code, khi điều kiện thất bại, $\lambda$ bị giảm **sau khi đã cập nhật tham số**, và bước “xấu” vẫn được giữ. Điều này có thể làm loss dao động hoặc làm chậm hội tụ trong một số trường hợp.
3. **Không có tiêu chí dừng $x^{k+1}=x^k$**: code dừng theo số epoch cố định.

Các khác biệt này không làm code “sai”, nhưng làm nó **không trùng khít** với Algorithm 1 về mặt tối ưu hóa.

---

## 6. So sánh GD truyền thống và GDA

### 6.1 Thời gian
Thời gian chạy với cả 3 kappa khác nhau đều chậm hơn so với GD truyền thống.

#### Why GDA Is Slower Than GD:

Sự chênh lệch runtime giữa GDA và GD là hệ quả trực tiếp của cấu trúc thuật toán.

Trong mỗi iteration, ngoài bước cập nhật giống GD:
$
x^{k+1} = x^k - \lambda_k \nabla f(x^k),
$
GDA còn thực hiện thêm **bước kiểm tra điều kiện sufficient decrease**:
$
f(x^{k+1}) \le f(x^k) - \sigma \langle \nabla f(x^k), x^k - x^{k+1} \rangle.
$

Bước này thường kéo theo:
- một phép **tính thêm giá trị hàm mất mát** tại $x^{k+1}$,
- các phép toán vector bổ sung (inner product),
- logic rẽ nhánh để cập nhật step size $\lambda_k$.

Do đó, mỗi iteration của GDA có **chi phí tính toán cao hơn** GD, ngay cả khi số iteration là như nhau.

#### Limited Impact of κ on Runtime

Kết quả cho thấy runtime của GDA **không thay đổi đáng kể theo κ**. Điều này gợi ý rằng:

- phần lớn thời gian của GDA đến từ **chi phí cố định mỗi iteration** (gradient + kiểm tra điều kiện),
- số lần vi phạm điều kiện sufficient decrease (dẫn đến giảm $\lambda_k$) không khác biệt lớn giữa các giá trị κ,
- hoặc thí nghiệm được chạy với số iteration cố định, khiến ảnh hưởng của κ lên hội tụ không phản ánh rõ trong runtime.

### 6.1. Thiết kế so sánh
- GD: $\eta = 0.5$ cố định.
- GDA: $\lambda_0 = 0.5$, $\sigma = 0.1$, $\kappa \in \{0.7, 0.8, 0.9\}$.

### 6.2. Quan sát từ đồ thị kết quả
2 đồ thị ở trên là so sánh accuracy với loss của 2 thuật toán, còn 2 hình ở dưới cũng thế nhưng trong quá trình train thì không xáo trộn (suffle) dữ liệu, nên các đường cong nhìn smooth hơn :>

Từ đồ thị trong file PDF:
- **Accuracy**: cả GD và các cấu hình GDA đều tăng nhanh trong giai đoạn đầu và tiệm cận mức rất cao về cuối (xấp xỉ 98–100% trên test set theo đường cong).  
- **Loss**: GD giảm loss nhanh hơn và thường nằm thấp hơn các đường GDA trong phần lớn quá trình huấn luyện, dù các đường GDA vẫn giảm đều và tiệm cận vùng loss thấp về cuối.

### 6.3. Giải thích cơ chế (mechanism)
- **GD learning rate cố định**: nếu $\eta$ đã được chọn “ổn”, GD có thể đi nhanh và ổn định.
- **GDA**: khi cập nhật gây giảm loss không đủ theo chuẩn Armijo, $\lambda$ bị giảm bởi $\kappa$. Điều này thường làm thuật toán thận trọng hơn, nhưng cũng có thể làm chậm tốc độ giảm loss nếu $\lambda$ bị shrink quá sớm hoặc quá thường xuyên.

Trong thực nghiệm này, việc GDA không "rollback và thử lại" (giữ bước đã đi) có thể làm lợi thế lý thuyết của sufficient decrease yếu hơn, vì $\lambda$ chỉ ảnh hưởng cho các bước sau.
