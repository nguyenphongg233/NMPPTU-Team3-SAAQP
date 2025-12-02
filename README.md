# Self-adaptive algorithms for quasiconvex programming and applications to machine learning

## Mục tiêu
Đọc hiểu, Cài đặt thuật toán, Viết báo cáo (LaTeX) và Thuyết trình (Beamer).

## Phân công theo Team

### Team 1: Báo cáo (report/) - 3 người
**Sản phẩm cần làm:**
- report.tex  
- report.pdf  
- images/ (chỉ chứa hình từ implementation)

**Nhiệm vụ:**
- Trình bày các khái niệm toán học liên quan.
- Giải thích và phác thảo chứng minh ba thuật toán: GDA, DA, SGDA.
- Tổng hợp kết quả từ Team 2, đưa biểu đồ vào báo cáo.
- Format LaTeX đầy đủ (mục lục, reference, hình ảnh).

---

### Team 2: Cài đặt thuật toán (implementation/)
**Sản phẩm cần bàn giao:**
- Code 3 thuật toán: gda.py, da.py, sgda.py
- Code các ví dụ trong paper
- Toàn bộ hình ảnh và log kết quả trong output/

**Nhiệm vụ:**
- Cài đặt thuật toán dạng module.
- Chạy các ví dụ trong paper.
- Chạy logistic regression + neural network.
- Xuất hình + log để chuyển sang Team 1 và Team 3.

---

### Team 3: Slide thuyết trình (slides/)
**Sản phẩm cần bàn giao:**
- slides.pdf
- media/ chứa ảnh thí nghiệm

**Nhiệm vụ:**
- Tóm tắt lý thuyết và thuật toán dưới dạng slide.
- Thêm kết quả từ Team 2.
- Thiết kế slide rõ ràng, dễ trình bày.

---

## Công cụ
- Python 3.8+, PyTorch, NumPy, Matplotlib  
- LaTeX, Beamer  

## Deadline
**23/12**

---

## Cấu trúc thư mục đề xuất + Giải thích

```
NMPPTU-Team/
├── README.md
├── requirements.txt
│
├── report/                     <-- Team 1: Báo cáo
│   ├── report.pdf              <-- File PDF cuối cùng
│   ├── report.tex              <-- File LaTeX chính (viết toàn bộ báo cáo)
│   └── images/                 <-- Ảnh biểu đồ từ implementation/output
│
├── implementation/             <-- Team 2: Cài đặt & chạy thí nghiệm
│   ├── src/                    <-- Mã nguồn Python
│   │   ├── algorithms/         <-- Bộ thư viện 3 thuật toán dạng module
│   │   │   ├── gda.py
│   │   │   ├── da.py
│   │   │   └── sgda.py
│   │   └── examples/           <-- Code chạy các ví dụ cụ thể trong paper
│   └── output/                 <-- Kết quả sinh ra từ code
│       ├── figures/            <-- Biểu đồ (dùng cho report & slides)
│       └── logs/               <-- File nhật ký, bảng so sánh số liệu
│
└── slides/                     <-- Team 3: Slide thuyết trình
    ├── slides.pdf              <-- File slide cuối cùng
    ├── slides.tex              <-- File Beamer chính
    └── media/                  <-- Ảnh/thí nghiệm được đưa vào slide
```
