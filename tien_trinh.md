# Tiến trình: điều kiện dừng, `max_vong_lap`, tìm nghiệm tối ưu

Ngày: 2026-10-07 · File: `bai_lam.ipynb`

## 1. Việc đã làm

### 1.1. Điều kiện dừng (dùng chung cho mọi thuật toán)

Dừng ở vòng đầu tiên mà

$$\|r(w_k)\| \le \text{tol}\cdot\|r(w_0)\|$$

với $r$ là thước đo tối ưu, $r=0 \Leftrightarrow w$ là nghiệm:

| Thuật toán | $r(w)$ |
|---|---|
| GD cố định, GD backtracking, Nesterov, Newton | $\nabla F(w)$ (Ridge) |
| SGD | $\nabla F(w)$ trên **toàn bộ** dữ liệu, kiểm tra cuối mỗi epoch (gradient mini-batch nhiễu, không dùng để dừng được) |
| Subgradient | subgradient có chuẩn nhỏ nhất (thước đo KKT) |
| ISTA / FISTA | gradient mapping $\big(w-\mathrm{prox}_{t\lambda}(w-t\nabla\ell(w))\big)/t$ |

Hết `max_vong_lap` mà chưa đạt ngưỡng thì ghi **"chưa hội tụ (hết ngân sách)"**.

### 1.2. Tham số `max_vong_lap` và `tol` cho mọi thuật toán

| Hàm | Trước | Sau |
|---|---|---|
| `gradient_descent` | `so_vong=1000` | `max_vong_lap=VONG, tol=TOL` |
| `gradient_descent_bt` | `so_vong=250` | `max_vong_lap=VONG, tol=TOL` |
| `gd_nesterov` | `so_vong=1000` | `max_vong_lap=VONG, tol=TOL` |
| `newton` | `so_vong=30` | `max_vong_lap=VONG_NT, tol=TOL` |
| `sgd` | `so_epoch=42` | `max_vong_lap=EPOCH, tol=TOL` (đơn vị: epoch) |
| `subgradient` | `so_vong=250` | `max_vong_lap=VONG, tol=TOL` |
| `proximal_gradient` | `so_vong=250` | `max_vong_lap=VONG, tol=TOL` |

- Mặc định lấy từ ô **CẤU HÌNH CHẠY** (`TOL=1e-3`, `VONG=500`, `VONG_NT=30`, `EPOCH=42`). Ô này đã được **chuyển lên trước** các thuật toán để mặc định `max_vong_lap=VONG, tol=TOL` dùng được.
- Chạy riêng một lần thì sửa thẳng khi gọi, ví dụ `newton(X, y, lam, max_vong_lap=100, tol=1e-10)`.
- `tol` trước đây đọc thẳng biến toàn cục `TOL`, nay là tham số riêng của từng hàm.

### 1.3. Tìm nghiệm tối ưu bằng chính thuật toán tự viết

- **Ridge** (ô Newton): `newton(..., backtracking=True, max_vong_lap=100, tol=1e-10)`, đối chiếu với `F*` của sklearn `newton-cholesky, tol=1e-12`.
- **Lasso** (ô Proximal): `proximal_gradient(..., 1/L, max_vong_lap=5000, tol=1e-6, gia_toc=True)` (FISTA). Lasso không có Newton nên dùng FISTA; `F_SAO_L` = min của kết quả này và mọi lần chạy Lasso khác.

### 1.4. Chạy

`jupyter nbconvert --execute --inplace` bằng `.venv`, chạy toàn bộ notebook mất khoảng 26 phút.

- **Lần 1**: dừng ở ô LightGBM (mục 8) với lỗi `OSError: access violation reading 0x0` trong `lgb.Dataset.set_label`, nên nbconvert không lưu output.
  Tách riêng (cùng kích thước 466243×160, nhãn dạng strided lẫn liền bộ nhớ) thì **không tái hiện được** lỗi. Ô này không liên quan đến thay đổi lần này (lần trước nó chưa từng chạy được vì kernel cũ không có `lightgbm`), nên **chưa sửa**.
- **Lần 2**: thêm `--allow-errors`. Mọi ô đến hết mục 7 chạy xong, output đã lưu vào notebook; riêng ô LightGBM vẫn báo lỗi trên.

## 2. Kết quả (λ = λ₁ = 1e-3, tol = 1e-3)

### 2.1. Nghiệm tối ưu

| Bài toán | Cách tìm | Kết quả | F* |
|---|---|---|---|
| Ridge | Newton backtracking tự viết, tol 1e-10 | **hội tụ sau 10 vòng**, 3.9 giây | **0.043630121211** |
| Ridge | sklearn `newton-cholesky`, tol 1e-12 | (mốc đối chiếu) | 0.043630121211 (lệch 0.0e+00) |
| Lasso | FISTA tự viết, bước 1/L, tol 1e-6 | **chưa hội tụ sau 5000 vòng** (345 giây), 19/160 hệ số khác 0 | **0.046538164358** (tốt nhất tìm được) |
| Lasso | sklearn liblinear L1 | | 0.046688817 |

→ Newton tự viết tìm đúng nghiệm Ridge **tới sai số máy**. Với Lasso, FISTA cho F thấp hơn sklearn 1.5e-4, nhưng đạt tol 1e-6 cần nhiều hơn 5000 vòng.

### 2.2. Quét tham số: setup nào đạt điều kiện dừng (ngân sách 500 vòng / 30 vòng Newton / 42 epoch)

**Ridge**

| Thuật toán | Setup | Trạng thái | F cuối | Giây |
|---|---|---|---|---|
| GD cố định | mọi t ∈ {0.1, 0.5, 1, 1.5, 2, 4}/L | chưa hội tụ | tốt nhất 0.044907766 (4/L) | ~17 |
| GD backtracking | β = 0.1 / 0.3 / 0.5 / 0.7 / 0.9 | hội tụ sau 164 / 208 / 261 / 315 / 394 vòng | 0.04369 → 0.04363 | 13.1 / 18.3 / 26.2 / 41.5 / 116.9 |
| Nesterov | t = 0.1/L | chưa hội tụ | 0.043832406 | 38.5 |
| Nesterov | t = 0.5 / 1 / 1.5 / 2 (/L) | hội tụ sau 454 / 317 / 205 / 172 vòng | ~0.04366–0.04368 | 33.6 / 23.7 / 14.9 / 12.4 |
| Newton cố định | t = 0.1 | chưa hội tụ | 0.062376531 | 10.1 |
| Newton cố định | t = 0.25 / 0.5 / 1 | hội tụ sau 27 / 13 / 6 vòng | ~0.04365–0.04367 | 9.4 / 4.5 / 2.1 |
| Newton backtracking | β = 0.1 / 0.5 / 0.9 | hội tụ sau 6 vòng (thử t=1 luôn được nhận) | 0.043649665 | ~2.3 |
| SGD | mọi α₀, mọi lịch giảm bước | chưa hội tụ (đúng lý thuyết: quả cầu nhiễu) | tốt nhất 0.044234 (α₀ = 0.2, cố định) | ~31 |

**Lasso**

| Thuật toán | Setup | Trạng thái | F cuối | Hệ số ≠ 0 |
|---|---|---|---|---|
| Subgradient | mọi α₀ | chưa hội tụ | tốt nhất 0.050041392 (α₀ = 8) | 154/160 |
| ISTA | mọi t | chưa hội tụ | tốt nhất 0.051836388 (2/L) | 26/160 |
| FISTA | t = 0.1/L | chưa hội tụ | 0.046901107 | 35/160 |
| FISTA | t = 0.5 / 1 / 1.5 / 2 (/L) | hội tụ sau 486 / 335 / 269 / 231 vòng | ~0.04657 | 18–22/160 |

### 2.3. Best của từng thuật toán (mục 6b)

Setup tốt nhất là setup nhanh nhất theo giây **trong số các setup đã hội tụ**; không setup nào hội tụ thì lấy setup có F thấp nhất.

| Thuật toán | Bài | Setup | Vòng | Giây | F − F* | logloss train | logloss val |
|---|---|---|---|---|---|---|---|
| GD cố định | Ridge | t = 4/L | > 500 | > 17.4 | 1.3e-3 | 0.043159 | 0.038855 |
| GD backtracking | Ridge | β = 0.1 | 164 | 13.09 | 5.7e-5 | 0.040929 | 0.036530 |
| GD Nesterov | Ridge | t = 2/L | 172 | 12.41 | 4.7e-5 | 0.040421 | 0.036007 |
| **Newton** | Ridge | t = 1 | **6** | **2.11** | 2.0e-5 | 0.040680 | 0.036263 |
| SGD | Ridge | α₀ = 0.2 | > 42 epoch | > 31.4 | — | 0.040939 | 0.036546 |
| sklearn lbfgs | Ridge | mặc định | 27 | 1.24 | 9.1e-6 | 0.040654 | 0.036241 |
| Subgradient | Lasso | α₀ = 8 | > 500 | > 17.4 | 3.5e-3 | 0.042725 | 0.038013 |
| **FISTA** | Lasso | t = 2/L | 231 | 15.86 | 3.9e-5 | 0.040801 | 0.036070 |
| sklearn liblinear | Lasso | L1 | 20 | 7.60 | 1.5e-4 | 0.041495 | 0.036835 |

(F* Ridge = 0.043630121211; F* Lasso = 0.046538164358.)

### 2.4. 24 sản phẩm, MAP@7 (Newton backtracking)

| | Trước (30 vòng cố định) | Sau (có điều kiện dừng) |
|---|---|---|
| Thời gian huấn luyện 24 mô hình | 245.2 s | **46.6 s** |
| MAP@7, khách có thêm mới | 0.796575 | 0.797596 |
| MAP@7, toàn bộ khách | 0.022488 | 0.022517 |

Có điều kiện dừng, mỗi mô hình dừng sau khoảng 6 vòng thay vì chạy đủ 30 vòng: nhanh hơn **5.3 lần**, MAP@7 gần như không đổi.

## 3. Nhận xét

1. **tol = 1e-3 (tương đối)** đủ để so tốc độ các thuật toán: những setup hội tụ đều dừng cách F* ≲ 6e-5. Muốn ra **đúng nghiệm** thì truyền `tol` chặt hơn và `max_vong_lap` lớn hơn, như ở mục 2.1.
2. **GD bước cố định** và **ISTA** không đạt tol 1e-3 trong 500 vòng: số điều kiện L/μ ≈ 6.25/1e-3 ≈ 6000, tốc độ tuyến tính quá chậm. Backtracking (bước lớn dần) và tăng tốc Nesterov/FISTA giải quyết được việc này.
3. **SGD bước cố định** không bao giờ đạt ngưỡng, vì nó chỉ lảng vảng trong "quả cầu nhiễu" quanh nghiệm, đúng như lý thuyết. Các lịch giảm bước 1/k và 1/√k thì giảm quá nhanh nên còn chậm hơn.
4. **Subgradient** chưa hội tụ ở mọi α₀ (tốc độ O(1/√k)) và gần như không cho nghiệm thưa (154/160 hệ số khác 0, so với 19/160 của FISTA).
5. Với Newton, β của backtracking không ảnh hưởng gì: bước t=1 luôn thoả Armijo nên cả ba β cho kết quả y hệt nhau.

## 4. Còn tồn đọng

- Ô LightGBM (mục 8) lỗi `access violation` trong kernel `.venv` (LightGBM 4.7.0, numpy 2.4.6). Chạy riêng ngoài notebook thì không lỗi; chưa tìm ra nguyên nhân.
- FISTA với tol 1e-6 cần hơn 5000 vòng. Nếu cần F* Lasso chính xác hơn thì tăng `max_vong_lap` (mỗi 1000 vòng mất khoảng 70 giây).
- Output của ô "So với GD thường" ở mục 3b và các hình trong `overleaf/hinh/` đã được ghi đè bằng kết quả lần chạy này.

---

# Phần 2: Đổi cách chọn best — lần chạy thử

## 5. Vì sao đổi

Cách cũ chọn best trong các setup đã đạt điều kiện dừng, lấy setup nhanh nhất theo giây. Cách này có ba vấn đề:

- Mỗi thuật toán dùng một thước đo r riêng.
- Các setup "đã hội tụ" thực ra dừng ở F − F* khác nhau (2e-5 đến 6e-5).
- Setup chưa hội tụ lại được so bằng tiêu chí khác (F cuối).

**Cách mới:** đo thời gian tới cùng một độ chính xác cho mọi thuật toán:

$$t_\varepsilon=\min\{t_k:\ F(w_k)-F^\star\le\varepsilon\}$$

Best = setup có $t_\varepsilon$ nhỏ nhất. $t_\varepsilon$ đọc thẳng từ lịch sử (F, t), nên mỗi setup chỉ cần chạy **một lần** là tính được cho mọi ε.

Đã cân nhắc và loại các cách sau:
- **Log-loss trên validation:** mọi thuật toán cùng một nghiệm w*, nên chênh lệch chỉ do tối ưu chưa xong.
- **Cùng ngân sách thời gian T:** kết quả phụ thuộc mạnh vào T.
- **Diện tích dưới đường cong:** khó giải thích.

## 6. Thay đổi trong notebook

| Chỗ | Thay đổi |
|---|---|
| 7 thuật toán | Thêm tham số `max_giay=np.inf`: trần thời gian, hết trần thì dừng. Mặc định không giới hạn nên các ô cũ chạy như trước |
| `sgd` | Thêm `lich_su_F=None`. Nếu truyền một list vào, cuối mỗi epoch ghi (F trên **toàn bộ** dữ liệu, giây), vì loss mini-batch không đo được F − F* |
| Mục mới **6c** (sau 6b) | Hàm `thoi_gian_toi(F, t, F_sao, eps)`; tính lại F* Lasso chính xác; chạy thử 11 setup đại diện |

Lần chạy thử đầu tiên (bảng ở mục 7) chạy bằng một script `exec` đúng mã của các ô cần thiết, bỏ qua các ô quét tham số và vẽ hình. Từ Phần 3 trở đi, ô 6c đã có output trong notebook (lần chạy toàn bộ); số liệu hai lần lệch nhau, xem mục 12.

## 7. Kết quả chạy thử

**Cấu hình:** mỗi setup tối đa `TRAN = 180` giây, `tol = 1e-12` (gần như chỉ dừng khi hết trần), `max_vong_lap` không giới hạn. Tổng thời gian chạy 38 phút; riêng nạp dữ liệu chỉ mất 34 giây.

**F\* dùng để đo:**
- Ridge: 0.043630121211 (Newton, khớp sklearn tới sai số máy).
- Lasso: **0.046538162389**, từ FISTA chạy 600 giây (8566 vòng). Mức min sau 300 giây và sau 600 giây chỉ lệch nhau 9.9e-10, nên F* Lasso chính xác cỡ 1e-9. Đo được tới ε = 1e-7 là an toàn.

**Thời gian (giây) tới F − F\* ≤ ε**, "–" là không đạt trong 180 giây:

| Setup | Bài | Vòng | s/vòng | F − F* min | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|---|---|---|---|
| Newton t=1 | Ridge | 10 | 0.349 | 0 | 1.7 | 2.1 | **2.4** | 2.4 | 2.8 |
| Newton bt β=0.5 | Ridge | 10 | 0.378 | 0 | 2.0 | 2.4 | 2.7 | 2.7 | 3.1 |
| Nesterov t=2/L | Ridge | 2611 | 0.069 | 2.2e-9 | 4.9 | 9.6 | 16.9 | 34.3 | 61.6 |
| GD backtracking β=0.1 | Ridge | 2259 | 0.080 | 7.7e-16 | 5.7 | 11.4 | 17.3 | 23.8 | 30.5 |
| SGD α0=1, 1/√k | Ridge | 216 epoch | 0.834 | 1.9e-7 | 3.8 | 16.4 | 41.2 | 90.1 | – |
| SGD α0=0.2 cố định | Ridge | 220 epoch | 0.821 | 2.1e-6 | 12.1 | 28.6 | 57.3 | – | – |
| GD cố định t=4/L | Ridge | 5204 | 0.035 | 1.5e-8 | 19.2 | 44.9 | 76.7 | 111.1 | 147.7 |
| GD cố định t=1/L | Ridge | 5228 | 0.034 | 9.8e-5 | 77.0 | 178.9 | – | – | – |
| FISTA t=2/L | Lasso | 2461 | 0.073 | 1.7e-9 | 8.7 | 17.1 | **25.2** | 50.2 | 77.0 |
| ISTA t=2/L | Lasso | 3603 | 0.050 | 5.6e-5 | 72.2 | 163.3 | – | – | – |
| Subgradient α0=8 | Lasso | 3467 | 0.052 | 1.1e-3 | – | – | – | – | – |

## 8. Rút ra từ lần chạy thử

1. **ε = 1e-5 làm mốc chính là hợp lý.** 8/11 setup đạt được trong 3 phút. Ngoài ra nó còn phân biệt rõ các nhóm: Newton vài giây, nhóm có gia tốc hoặc backtracking khoảng 17–25 giây, SGD 40–60 giây, GD cố định trên 75 giây.
2. **Thứ hạng đổi theo ε, đúng như dự đoán:**
   - SGD 1/√k nhanh nhất ở ε = 1e-3 trong nhóm bậc nhất (3.8 s, nhanh hơn Nesterov 4.9 s), nhưng tới 1e-6 thì chậm gấp ~3 lần.
   - Nesterov thắng backtracking ở 1e-4/1e-5, nhưng thua ở 1e-6/1e-7. Nesterov dao động gần nghiệm, còn backtracking với bước lớn dần giữ được tốc độ tuyến tính đều.
3. **SGD bước cố định kẹt ở quả cầu nhiễu:** F − F* nhỏ nhất là 2.1e-6, không xuống thêm được. Lịch 1/√k xuống tới 1.9e-7.
4. **Tốc độ thật nhanh hơn ước lượng bi quan.**
   - Ước lượng trước khi chạy: GD t=4/L cần khoảng 4 phút để tới 1e-5. Thực tế chỉ 77 giây.
   - Suy ngược từ tốc độ đo được: μ hiệu dụng ≈ 2e-3, gấp đôi chặn dưới λ = 1e-3.
   - Ngoại suy cho các setup chưa tới đích: GD t=1/L cần khoảng 280 giây để tới 1e-5 và khoảng 480 giây để tới 1e-7. ISTA t=2/L cần khoảng 255 và 435 giây.
5. **Subgradient không đạt cả ε = 1e-3 sau 180 giây** (đang ở 1.1e-3). Với tốc độ O(1/√k), tới 1e-4 cần khoảng 100 lần số vòng, tức hàng giờ. Trong lần chạy lớn sẽ chỉ chạy hết trần rồi ghi "không đạt".
6. **Chi phí mỗi vòng:**
   - Một epoch SGD (0.82 s) đắt bằng ~24 vòng GD (0.034 s), chủ yếu do bước lấy mini-batch `X[idx]` phải chép dữ liệu.
   - Nesterov tốn 2 gradient mỗi vòng, trong đó 1 gradient chỉ dùng để kiểm tra điều kiện dừng.
   - Newton tốn khoảng 10 lần một vòng GD.
7. **Lãng phí cần sửa cho lần chạy lớn:** với `tol = 1e-12`, các setup như GD backtracking đã chạm F − F* = 1e-16 nhưng vẫn chạy hết 180 giây. Lần chạy lớn cần **dừng sớm khi F − F\* ≤ 1e-8**, nếu không mọi setup đều chạy hết trần.

## 9. Kế hoạch cho lần chạy lớn (ngân sách 10 tiếng)

- **Trần mỗi setup: 600 giây.** Đủ để GD t=1/L và ISTA t ≥ 1/L tới 1e-7, theo ngoại suy ở mục 8.4.
- **Dừng sớm khi F − F\* ≤ 1e-8**, thấp hơn ε nhỏ nhất 10 lần.
- **Lưới tham số:** giữ 46 setup cũ, thêm:
  - bước dày hơn quanh vùng tốt: Nesterov/FISTA/GD t ∈ {2.5, 3, 4, 6}/L;
  - SGD: batch ∈ {1024, 4096, 16384} × lịch {cố định, 1/√k};
  - GD backtracking: thử thêm hệ số Armijo c.
- **Ước lượng thời gian:**
  - khoảng 15 setup không thể đạt sẽ chạy hết trần: ~2.5 giờ;
  - các setup còn lại dừng sớm: ~1.5 giờ;
  - phần mở rộng lưới: ~2–3 giờ;
  - **tổng khoảng 6–7 giờ**, còn dư để dự phòng.
- **Báo cáo:** bảng best theo từng ε ∈ {1e-3, …, 1e-7}, kèm hình F − F* theo thời gian có kẻ đường ngang ε = 1e-5.

---

# Phần 3: Cải thiện biểu đồ so sánh

## 10. Thay đổi trong notebook

**Hàm vẽ (ô `import matplotlib`)**

| Thay đổi | Chi tiết |
|---|---|
| `ve()` thêm 5 tham số | `net` (kiểu nét từng đường), `dau` (marker), `chu` (legend tự đặt), `ten_y` (tên trục tung), `ghi_chu` (dòng chữ ở góc dưới trái) |
| `chu_giai()` | Ngoài ô vuông màu, legend giờ vẽ được cả mẫu nét đứt/liền/chấm |
| Hàm mới `day_mau(k, mau_goc)` | k màu từ nhạt đến đậm của cùng một màu gốc: tham số nhỏ thì nhạt, lớn thì đậm |
| `chon()`, `tot_nhat()`, `diem_cuoi()` | Chuyển lên ô cấu hình, để các mục 3–5 lấy được "setup tốt nhất" làm mốc |
| `gradient_descent_bt()` | Thêm `lich_su_buoc=None` để ghi độ dài bước được nhận ở mỗi vòng |

**Quy ước chung cho mọi hình so sánh:**
- Cùng tham số thì cùng màu.
- Nét đứt là mốc, nét liền là biến thể đang xét, nét chấm là mốc từ phần khác.
- Mốc luôn gồm lựa chọn tiêu chuẩn theo lý thuyết (1/L) và setup tốt nhất của phần trước.

**7 ô so sánh mới.** Mỗi ô có một đoạn markdown giải thích đi kèm; hình lưu trong `overleaf/hinh/ss_*.pdf`.

| Vị trí | Hình | Nội dung |
|---|---|---|
| Sau 3 (backtracking) | `ss_gd_bt_*`, `ss_bt_buoc` | GD t = 1/L (tiêu chuẩn) và GD tốt nhất (4/L), so với 5 giá trị β của backtracking; thêm hình độ dài bước backtracking chọn |
| Sau 3b (Nesterov) | `ss_gd_nesterov_*` | Từng cặp GD và Nesterov cùng bước, t ∈ {0.5, 1, 2}/L |
| Sau 3c (Newton) | `ss_newton_*` | Newton bước cố định và Newton backtracking trên một hình, mốc là Nesterov tốt nhất; trục số vòng dùng thang log |
| Sau 4 (SGD) | `ss_sgd_gd_*` | SGD (F trên toàn bộ dữ liệu sau mỗi epoch) so với GD và Nesterov tốt nhất, trục hoành là số lượt qua dữ liệu |
| Sau 5b (Proximal) | `ss_ista_fista_*` | Từng cặp ISTA và FISTA cùng bước, mốc là subgradient tốt nhất |
| Mục 6 | `ss_ridge_lasso_*` | GD với ISTA, Nesterov với FISTA, cùng bước 2/L; trục tung là khoảng cách tương đối |
| Mục 6 (thí nghiệm mới) | `ss_lambda_*` | GD, Nesterov và Newton với λ ∈ {1e-2, 1e-3, 1e-4} |

Đã chạy lại toàn bộ notebook (66 phút). Mọi ô có output; riêng ô LightGBM vẫn lỗi `access violation` như cũ.

## 11. Đọc từ các hình mới

1. **Backtracking chọn bước gấp ~25–30 lần 1/L.** Bước trung vị tính theo đơn vị 1/L: β = 0.1 → 24.5; 0.3 → 23.2; 0.5 → 25.0; 0.7 → 27.0; 0.9 → 29.9. Mức này cao hơn hẳn ngưỡng 2/L của lý thuyết.
   - Lý do: L = σ_max(X)²/(4n) dùng chặn σ'(z) ≤ 1/4. Nhưng dữ liệu chỉ có 1.2% mẫu dương, nên tại nghiệm σ(1−σ) chỉ cỡ 0.012. Độ cong thật vì vậy nhỏ hơn khoảng 20 lần so với chặn của L.
   - Đây là lý do GD t = 4/L vẫn hội tụ dù vượt 2/L, và là lý do backtracking thắng GD bước cố định.
   - β nhỏ cho bước dao động mạnh hơn: mỗi lần co bước bị co mạnh.
2. **Nesterov so với GD cùng bước:** ở mọi bước, Nesterov xuống tới 1e-4 trong 170–450 vòng, còn GD sau 500 vòng vẫn ở mức 1e-2. Khoảng cách giữa hai nét cùng màu chính là phần tăng tốc.
3. **Newton:** backtracking **trùng hẳn** t = 1 (notebook kiểm tra tự động, in ra `True`). t = 0.5 và 0.25 cũng hội tụ nhưng mất đi pha bậc hai; t = 0.1 hội tụ tuyến tính chậm. Theo thời gian, Newton t = 1 tới 1e-4 sau khoảng 2 giây, Nesterov tốt nhất cần khoảng 12 giây.
4. **SGD trên trục số lượt qua dữ liệu:**
   - SGD bước cố định (α₀ = 0.2) tới **~6e-5 sau 42 lượt**, trong khi GD 4/L ở cùng số lượt mới khoảng 3e-2. Tính theo lượt qua dữ liệu thì SGD vượt trội.
   - Lịch 1/√k với α₀ = 0.2 thì bước teo quá sớm, dừng ở khoảng 2e-3.
   - **Cần chỉnh hình này:** trục hoành kéo tới 400 lượt theo đường GD, nên 42 lượt của SGD bị dồn vào mép trái. Nên dùng `log_x=True` cho hình này.
5. **Ridge so với Lasso:** GD và ISTA gần như **trùng nhau**, Nesterov và FISTA cũng vậy. Phần không trơn λ₁‖w‖₁ **không làm chậm** các phương pháp proximal. Bước prox có công thức đóng nên không tốn thêm gì. Tương phản rõ với subgradient (mục 5).
6. **Thí nghiệm λ khớp với lý thuyết:**

   | λ | κ ≤ L/λ | GD (500 vòng) | Nesterov | Newton |
   |---|---|---|---|---|
   | 1e-2 | 626 | chưa hội tụ, F − F* = 4.3e-3 | hội tụ sau 193 vòng | 6 vòng |
   | 1e-3 | 6 252 | chưa hội tụ, 1.2e-2 | 317 vòng | 6 vòng |
   | 1e-4 | 62 509 | chưa hội tụ, 1.5e-2 | 236 vòng | 7 vòng |

   - Newton gần như không phụ thuộc κ (6–7 vòng). GD chậm đi rõ khi κ tăng.
   - Số vòng Nesterov ở λ = 1e-4 (236) **ít hơn** ở λ = 1e-3 (317), trái với √κ. Lý do: điều kiện dừng đo theo gradient, F − F* ≤ ‖g‖²/(2μ), nên μ nhỏ thì cùng một tol lại dừng ở F − F* lớn hơn (1.0e-4 so với 3.4e-5). Muốn so số vòng giữa các λ thì phải dùng t_ε chung, đúng như cách chọn best mới. Đường cong khoảng cách tương đối trên hình mới là cách đọc đúng.

## 12. Phát hiện khi chạy lại ô 6c: thời gian đo dao động giữa các lần chạy

So sánh lần chạy thử riêng (Phần 2) với ô 6c chạy trong notebook lần này:

| Setup | s/vòng lần trước | s/vòng lần này | t(1e-5) lần trước → lần này |
|---|---|---|---|
| ISTA t=2/L | 0.0500 | 0.0339 | không đạt → 168.9 s |
| Subgradient α0=8 | 0.0519 | 0.0346 | t(1e-3): không đạt → 129.1 s |
| Các setup Ridge | — | — | lệch < 3% |

Chi phí mỗi vòng của ISTA và subgradient **lệch tới khoảng 50%** giữa hai lần chạy cùng mã, có thể do máy đang tải việc khác. Vì t_ε đo bằng giây, **lần chạy lớn nên đo lặp hoặc báo cáo kèm số vòng / số lượt qua dữ liệu tới ε** để thứ hạng không phụ thuộc vào nhiễu thời gian.

---

# Phần 4: Sửa theo note_nhung_gi_can_sua.md

Notebook đã sửa nhưng **chưa chạy lại** (người dùng tự chạy). Các ô bị sửa đã được xoá output cũ. Để bắt lỗi, tôi chạy thử mọi ô (trừ LightGBM) bằng script trên mẫu 20 000 dòng, ngân sách nhỏ: không lỗi. Số liệu của lần chạy thử đó không có giá trị.

## 13. Thay đổi

**Đo thời gian cho đúng chi phí**
- `F_ridge()` (ô hàm mục tiêu): chỉ tính F, không tính gradient. Vòng `while` thử bước của GD backtracking và Newton backtracking dùng hàm này thay cho `ridge()`, nên mỗi lần thử bước không còn tốn thêm một phép `X.T @ (...)`.
- Nesterov và FISTA **dừng đồng hồ** khi tính gradient tại w, vì gradient đó chỉ dùng để ghi F và kiểm tra dừng. Mỗi vòng giờ tốn 1 gradient như GD/ISTA. ISTA vẫn tính giờ phần này, vì nó dùng chính gradient đó để đi.
- Cả 7 thuật toán có thêm `F_dung=-np.inf`: dừng khi F ≤ F_dung. Ô 6c truyền F* + 1e-8 để setup nào chạm nghiệm thì dừng ngay, không chạy hết trần.

**Màu (ô vẽ)**
- Bỏ `MAU_TT`, `MAU_6` và `day_mau()`. Thay bằng **một danh sách `MAU` gồm 8 màu, thứ tự cố định**; đường vẽ thứ i lấy màu thứ i.
- sklearn lấy màu kế tiếp sau các đường. Trên hình theo vòng, sklearn là đường ngang đứt; trên hình theo thời gian, sklearn là **một điểm** (giây, F − F*) và trục hoành được nới để điểm này không bị cắt.
- Các hình cặp đôi (GD/Nesterov, ISTA/FISTA): mỗi tham số một màu theo thứ tự, thuật toán phân biệt bằng kiểu nét. Các hình này truyền `mau=MAU[:k]`, `ve()` lặp vòng màu.

**Bỏ:**
- hình 2b (tách đường backtracking);
- hình độ dài bước backtracking, thay bằng một câu trong markdown: bước trung vị khoảng 25/L, vì σ(1−σ) ≈ 0.01 ≪ 1/4;
- hai hình Newton riêng (bước cố định, backtracking);
- hình "loss mini-batch so với F toàn bộ";
- hình Ridge vs Lasso;
- mục 6b (bảng và biểu đồ cột).

**Sửa:**

| Chỗ | Sửa |
|---|---|
| SGD đổi lịch | 3 lịch × 2 α₀ (`eta_tot` và 1), so bằng F toàn bộ (`SGD_F`). Màu là lịch, kiểu nét là α₀ |
| SGD vs GD | Mỗi lịch lấy α₀ tốt hơn; Nesterov tính 1 lượt mỗi vòng; trục hoành log; in tỉ lệ 1 epoch / 1 vòng GD |
| Newton gộp | Thêm giải thích: t = 1 thoả Armijo ngay từ vòng đầu nên β không có dịp ảnh hưởng |
| Subgradient | Sửa câu "sinh nghiệm thưa": thực tế 154/160 hệ số khác 0 |
| Tổng kết Ridge | SGD dùng F toàn bộ; in thêm cột F − F*; markdown ghi rõ setting của 5 best |
| Tổng kết Lasso | `solver="saga"`, `l1_ratio=1`, `random_state=0`. liblinear phạt cả intercept nên giải bài khác |
| Thí nghiệm λ | Đếm số vòng tới khoảng cách tương đối ≤ 1e-6 (không dùng TOL), mỗi lần chạy tối đa `TRAN_LAM = 120` giây. In tỉ lệ số vòng khi λ giảm 10 lần, so với lý thuyết (GD ×10, Nesterov ×3.2, Newton ×1). Ghi rõ L/λ chỉ là cận trên của κ |
| 6c | Thành **kết quả chính**. Công tắc `CHAY_DAY_DU`: False chạy 11 setup với trần 180 giây (khoảng 40 phút); True chạy mọi setup với trần 600 giây (khoảng 5 giờ). Thêm bảng best theo từng ε, mỗi ô ghi giây / vòng [setup]. Kết quả lưu trong `KQ_6C` |
| Mục 7 | `nhanh_nhat` lấy từ `KQ_6C` (Ridge, nhanh nhất tới 1e-6), thay cho `ds_best` của 6b đã xoá |
| MAP@7, LightGBM | Thêm tiêu đề "Phần II. Mở rộng về mô hình" |

## 14. Chưa làm / cần biết

- **Nesterov có restart** (note ghi "cân nhắc"): chưa thêm.
- **Danh sách setting 5 best** trong markdown mục 6 lấy từ lần chạy trước. Sau khi chạy 6c đầy đủ cần cập nhật lại; theo note, SGD sẽ đổi thành α₀ = 1, lịch 1/√k.
- **Thời gian chạy tăng thêm** so với lần trước:
  - SGD đổi lịch: 6 thay vì 3 lần chạy, khoảng +1.5 phút;
  - thí nghiệm λ: tối đa khoảng 12 phút;
  - saga có thể chậm hơn liblinear.
  
  Với `CHAY_DAY_DU = False`, cả notebook mất khoảng 80 phút.
- **Hình cũ không còn được tạo** nhưng vẫn nằm trong `overleaf/hinh/`: `gd_bt_vonglap_tach.pdf`, `ss_bt_buoc.pdf`, `newton_codinh_*.pdf`, `newton_bt_*.pdf`, `sgd_muot_vs_batch.pdf`, `ss_ridge_lasso_*.pdf`, `best_so_sanh.pdf`. Chưa xoá, nếu slide/Overleaf còn tham chiếu thì cần bỏ.

---

# Phần 5: Chạy toàn bộ notebook (`CHAY_DAY_DU = False`) và lưu kết quả 6c

## 15. Lần chạy toàn bộ sau khi sửa theo note (2026-10-07, 77 phút)

`jupyter nbconvert --execute --inplace --allow-errors`. Mọi ô có output; ô LightGBM vẫn lỗi `access violation` như cũ.

**Tác dụng của việc sửa cách đo thời gian** (so với lần chạy thử ở Phần 2, cùng setup):

| Setup | t(1e-5) trước | t(1e-5) sau | Lý do |
|---|---|---|---|
| Nesterov t=2/L | 16.8 s | **8.1 s** | không tính gradient tại w (dùng để kiểm tra dừng) |
| FISTA t=2/L | 17.6–25.2 s | **8.8 s** | như trên |
| GD backtracking β=0.1 | 17.1 s | **12.5 s** | vòng thử bước chỉ tính F (`F_ridge`) |
| GD cố định t=4/L, Newton | 75.9 / 2.3 s | 73.2 / 2.4 s | không đổi, đúng như mong đợi |

**Bảng best 6c** (11 setup đại diện; mỗi ô ghi giây / vòng; SGD: vòng = epoch):

| Thuật toán | ε = 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|
| Newton (t=1) | 1.7 / 5 | 2.1 / 6 | **2.4 / 7** | 2.4 / 7 | 2.7 / 8 |
| Nesterov t=2/L | 2.3 / 71 | 4.6 / 138 | 8.1 / 244 | 16.5 / 498 | 30.3 / 894 |
| GD backtracking β=0.1 | 4.2 / 72 | 8.3 / 144 | 12.5 / 217 | 17.3 / 300 | 22.2 / 383 |
| SGD α0=1, 1/√k | 3.8 / 5 | 16.5 / 22 | 41.1 / 55 | 89.7 / 120 | – |
| GD cố định t=4/L | 18.4 / 560 | 42.8 / 1297 | 73.2 / 2217 | 106.6 / 3214 | 143.4 / 4272 |
| FISTA t=2/L (Lasso) | 2.8 / 82 | 5.5 / 161 | **8.8 / 256** | 21.3 / 615 | 34.4 / 997 |
| ISTA t=2/L (Lasso) | 46.5 / 1370 | 110.1 / 3216 | 170.1 / 4992 | – | – |
| Subgradient α0=8 (Lasso) | 125.8 / 3730 | – | – | – | – |

- **Nesterov thắng backtracking ở ε ≥ 1e-5, thua ở ε ≤ 1e-6.** Kết quả này giống lần chạy thử, nay rõ hơn vì đã đo đúng chi phí.
- F* Lasso: FISTA 600 giây (17 420 vòng) cho **0.046538162365**. Min sau 300 giây và sau 600 giây chỉ lệch 1.6e-11.
- Mục 7 tự chọn **Newton (t=1)**: tới 1e-6 sau 2.37 giây.

**Các ô khác:**
- **SGD đổi lịch:** F − F* sau 42 epoch:

  | | cố định | 1/√k | 1/k |
  |---|---|---|---|
  | α₀ = 0.2 | 6.1e-5 | 1.9e-3 | 9.0e-3 |
  | α₀ = 1 | 8.7e-5 | **1.5e-5** | 4.6e-4 |

  Đúng như note dự đoán: lịch giảm bước cần α₀ lớn hơn, và α₀ = 1 với 1/√k tốt nhất.
- **1 epoch SGD = 0.744 giây = khoảng 23 vòng GD** (0.0328 giây).
- **Lasso, sklearn saga:** F = 0.046538998 (cách F* 8.4e-7, sát hơn hẳn liblinear trước đây: 1.5e-4), 19/160 hệ số khác 0, mất 55.5 giây. Chậm hơn FISTA (tới 1e-6 sau 21 giây).
- **Thí nghiệm λ.** Số vòng tới khoảng cách tương đối 1e-6, kèm κ thật = λ_max(H)/λ_min(H) tính từ Hessian tại w* (cùng công thức Hessian của Newton):

  | λ | L/λ (cận trên) | λ_max(H) | λ_min(H) | **κ thật** | √κ | GD 1/L | Nesterov | **Nesterov / √κ** | Newton |
  |---|---|---|---|---|---|---|---|---|---|
  | 1e-2 | 626 | 0.609 | 0.00633 | **96** | 9.8 | > 3682 (hết 120 s) | 384 | **39.2** | 6 |
  | 1e-3 | 6 252 | 0.392 | 0.00100 | **392** | 19.8 | > 3571 | 896 | **45.3** | 7 |
  | 1e-4 | 62 509 | 0.299 | 0.00010 | **2 986** | 54.6 | > 3473 | 1247 | **22.8** | 8 |

  - **κ thật nhỏ hơn cận L/λ từ 7 đến 21 lần.** Nguyên nhân: λ_max(H) chỉ 0.3–0.6, so với L = 6.25. Lý do giống mục 11: dữ liệu chỉ có 1.2% mẫu dương, nên trung bình σ(1−σ) chỉ khoảng 0.01, thay vì chặn 1/4.
  - **Cột Nesterov / √κ không hằng số** (39 → 45 → 23), nên **chưa kết luận được** số vòng Nesterov tỉ lệ với √κ.
    - Giữa λ = 1e-2 và 1e-3, tỉ số lệch khoảng 15%: còn tạm khớp.
    - Ở λ = 1e-4, κ tăng ×7.6 nhưng số vòng chỉ tăng ×1.4. Số vòng tăng chậm hơn hẳn √κ.
  - Cách giải thích hợp lý (chưa kiểm chứng):
    - Với λ ≤ 1e-3, λ_min(H) **đúng bằng λ**. Vậy theo hướng riêng nhỏ nhất, dữ liệu gần như không có độ cong, có thể do các cột lag gần cộng tuyến; chỉ phần phạt tạo ra độ cong ở hướng đó.
    - Nếu w₀ = 0 và nghiệm w* chỉ khác nhau rất ít theo hướng đó, thì hướng đó đóng góp rất ít vào F − F*. Khi đó đạt mức tương đối 1e-6 mà **chưa cần hội tụ theo hướng chậm nhất**.
    - √κ là chặn **trường hợp xấu nhất**, không phải tốc độ thật.
    - Ngoài ra, Nesterov ở đây là bản cho hàm lồi tổng quát (θ_k không dùng μ), và bước 1/L dùng L = 6.25 chứ không dùng λ_max(H).
  - Muốn kiểm chứng: chiếu w* lên vector riêng của λ_min(H), hoặc đo thêm ở mức chính xác sâu hơn (1e-9) để xem tỉ số có ổn định lại không.
  - Newton tăng ×1.2 rồi ×1.1, gần như không phụ thuộc κ, đúng lý thuyết.
  - GD bước 1/L không đạt 1e-6 trong 120 giây ở cả ba λ, nên **chưa đo được tỉ lệ cho GD**. Muốn có thì cần tăng `TRAN_LAM` hoặc dùng bước 4/L.
  - Số κ ở bảng trên được tính riêng bằng script dùng đúng code notebook; số vòng lấy từ lần chạy notebook ngày 2026-10-07. Ô thí nghiệm λ trong notebook đã được sửa để tự tính κ thật và cột Nesterov / √κ, sẽ có output khi chạy lại notebook.

## 16. Lưu và đọc lại kết quả 6c

Ô 6c lưu kết quả của **mỗi chế độ vào một file riêng**, để chạy `True` không ghi đè kết quả `False`:

| `CHAY_DAY_DU` | File | Nội dung |
|---|---|---|
| False | `kq_6c_dai_dien.pkl` | `KQ_6C` (lịch sử F, t của mọi setup), F* Lasso, F* Ridge, `TRAN`, thời điểm chạy |
| True | `kq_6c_day_du.pkl` | như trên |

- Nếu file đã có, ô **đọc lại thay vì chạy lại** (0 giây thay vì 40 phút hoặc 5 giờ). Các bảng in ra như bình thường.
- `CHAY_LAI_6C = True` để bắt buộc chạy lại, ví dụ sau khi sửa thuật toán. Khi chạy lại, file của chế độ đó bị ghi đè.
- Nếu F* Ridge hiện tại khác lúc lưu (dữ liệu hoặc λ đã đổi), ô in **cảnh báo**.
- Đã kiểm tra cả hai nhánh bằng cách chạy thử trên mẫu nhỏ hai lần: lần 1 lưu, lần 2 đọc lại trong 0 giây.
- **Chưa có file nào trong repo.** Lần chạy `False` ở mục 15 diễn ra trước khi có tính năng lưu, nên lần chạy notebook tới vẫn mất khoảng 40 phút cho 6c rồi mới tạo `kq_6c_dai_dien.pkl`. Các file `.pkl` chưa được thêm vào `.gitignore`. Kích thước dự kiến: bản đại diện vài trăm KB, bản đầy đủ vài MB.

## 17. Lưới SGD cho bản đầy đủ: dò cả α₀ cho mỗi tổ hợp (batch, lịch)

**Vấn đề:**
- Kế hoạch cũ (mục 9) ghi SGD chạy batch × lịch nhưng không nói đến α₀.
- Bản đầy đủ trong ô 6c lại chỉ dò α₀ ở batch 4096.
- Bước tốt phụ thuộc cả cỡ batch lẫn lịch giảm bước. Bản đại diện đã cho thấy lịch 1/√k cần α₀ = 1, không phải 0.2.
- Dùng chung một α₀ cho mọi tổ hợp thì so sánh giữa các tổ hợp bị lệch.

**Sửa (ô 6c):**
- Lưới mới: batch ∈ {1024, 4096, 16384} × lịch ∈ {cố định, 1/√k} × **α₀ ∈ {0.2, 1, 5}**, tổng 18 lần chạy. Cấu hình nằm ở `BATCH_SGD`, `LICH_SGD`, `ETA_SGD` đầu ô.
- `sgd_F()` nhận thêm tham số `batch`. Tên setup có dạng `B=4096 1/sqrt α0=1`.
- In thêm bảng **α₀ tốt nhất cho từng tổ hợp (batch, lịch)** ở mọi ε, mỗi ô ghi giây / epoch [α₀]. Nhờ vậy các tổ hợp được so ở bước tốt nhất của chính chúng.
- Bản đại diện (`CHAY_DAY_DU = False`) giữ nguyên 2 setup SGD: (4096, cố định, 0.2) và (4096, 1/√k, 1). Bảng mới chỉ in khi có từ 2 tổ hợp trở lên.
- **Thời gian bản đầy đủ tăng từ khoảng 5 lên khoảng 6–6.5 giờ.** Lý do: SGD bước cố định không bao giờ chạm F* + 1e-8 nên chạy hết trần 600 giây; 18 lần chạy so với 10 lần trước là khoảng +80 phút.
- Đã chạy thử nhánh đầy đủ trên mẫu 20 000 dòng, trần 3 giây: không lỗi, ra đủ 18 tổ hợp và bảng mới.

---

# Phần 6: Mở rộng lưới 6c và in bảng theo note_nhung_gi_can_sua_2nd.md

Notebook đã sửa, **chưa chạy lại trên dữ liệu thật**. Tôi kiểm tra bằng cách chạy mọi ô (trừ LightGBM) trên mẫu 20 000 dòng với trần nhỏ, ở cả `CHAY_DAY_DU = False` và `True`, cùng nhánh đọc lại từ file: không lỗi, mọi bảng in đúng.

## 18. Mở rộng lưới bản đầy đủ (ô 6c)

| Phần mở rộng | Lưới | Số lần chạy thêm |
|---|---|---|
| Bước lớn hơn cho GD, Nesterov, FISTA (`BUOC_THEM`) | t ∈ {2.5, 3, 4, 6}/L, hợp với lưới cũ | GD +3, Nesterov +4, FISTA +4 |
| Hệ số Armijo c cho backtracking (`C_ARMIJO`) | c ∈ {1e-4, 1e-2, 0.1, 0.3} × β ∈ {0.1, 0.5}, cộng 5 giá trị β ở c = 1e-4 | +6 |

- Lý do thêm bước lớn: backtracking tự chọn bước khoảng 25/L (mục 11), nên bước cố định > 2/L vẫn có thể hội tụ và nhanh hơn. Trên mẫu nhỏ, Nesterov và FISTA tốt nhất đúng là rơi vào các bước mới thêm (6/L, 3/L).
- Tổng cộng 71 setup, so với 46 trước đó. Ước tính bản đầy đủ khoảng **7 giờ**. Phần lớn các setup mới dừng sớm khi chạm F* + 1e-8; riêng GD 2.5/L và 3/L có thể chạy hết trần 600 giây.
- Bản đại diện (`False`) giữ đúng 11 setup cũ. Riêng tên setup backtracking đổi thành `β=0.1 c=0.0001`.

## 19. Mọi mục trong note 2nd đều in ra bảng

Thêm hàm dùng chung **`in_bang(cot, hang, tieu_de)`** vào ô cấu hình. Hàm này in bảng căn cột và tự đổi ký hiệu LaTeX trong tên setup ($\beta$, $\alpha_0$, tên lịch) thành chữ thường. Các hình vẫn giữ nguyên, bảng được in kèm.

| Mục trong note | Ô | Cột |
|---|---|---|
| **Chi phí mỗi vòng** (mới) | cuối 6c | Thuật toán, lượt qua dữ liệu mỗi vòng (lý thuyết), giây/vòng (trung vị trên các setup), × vòng GD |
| **Newton** | 3c, ô quét | Setup, trạng thái, giây, F − F*, **bước có bị co?** (`newton()` có thêm `lich_su_buoc`, ghi bước thực dùng ở mỗi vòng) |
| **Thí nghiệm λ** | 6, thí nghiệm λ | Bảng 1: λ, L/λ (cận κ), κ thật, √κ, số vòng GD / Nesterov / Newton, Nesterov/√κ. Bảng 2: tỉ lệ tăng khi λ giảm 10 lần, đặt cạnh mức tăng của κ và √κ |
| **So với sklearn** | 6, tổng kết Ridge | Phương pháp, setup, trạng thái, **số vòng** (sklearn: `n_iter_`), F cuối, F − F*, giây |
| **Tổng kết Lasso** | 6, tổng kết Lasso | Phương pháp, setup, trạng thái, số vòng, F, F − F*, giây, **hệ số ≠ 0** |
| **Kết quả quét tham số** | mọi ô quét (GD, backtracking, Nesterov, SGD, SGD lịch, subgradient, ISTA/FISTA) | Tham số, trạng thái, F cuối, F − F* (Ridge), giây; Lasso thêm hệ số ≠ 0 |
| **Log-loss train/val** | ô mới sau tổng kết Lasso | Phương pháp, bài, setup, log-loss train, log-loss val, cho setup tốt nhất của 9 phương pháp, kể cả 2 sklearn. Ô chạy lại đúng setup tốt nhất để lấy w, mất khoảng 1–2 phút |

Ba bảng có sẵn trong 6c (theo từng setup, best theo ε, α₀ tốt nhất của SGD) cũng chuyển sang `in_bang`. Trước đó chúng bị lệch cột khi tên setup SGD dài.

---

# Phần 7: Chạy bản đầy đủ (`CHAY_DAY_DU = True`)

## 20. Tiến trình

| Thời điểm | Sự kiện |
|---|---|
| 2026-10-08 00:58 | Đặt `CHAY_DAY_DU = True` trong ô 6c, bắt đầu `jupyter nbconvert --execute --inplace --allow-errors` cho toàn bộ notebook. Dự kiến khoảng 8 giờ: ô 6c khoảng 7 giờ với 71 setup, trần 600 giây mỗi setup; các ô khác khoảng 40 phút. Kết quả 6c sẽ được lưu vào `kq_6c_day_du.pkl` ngay khi ô 6c chạy xong, nên vẫn còn dù các ô sau có lỗi. |
| 2026-10-08 01:55 | Các ô trước 6c chạy xong. Hình cuối cùng trước 6c (`ss_lambda_*.pdf`, thí nghiệm λ) được ghi lúc 01:55, nên ô 6c bắt đầu vào khoảng giờ này. |
| 2026-10-08 07:36 | Kiểm tra: nbconvert vẫn đang chạy ô 6c (đã khoảng 5 giờ 40 phút), chưa có `kq_6c_day_du.pkl`. Theo ước tính 7 giờ, ô 6c sẽ xong khoảng 09:00. |
| 2026-10-08 10:20 | **Ô 6c chạy xong** sau khoảng 8 giờ 25 phút (01:55 → 10:20), lâu hơn ước tính 7 giờ. Đã lưu `kq_6c_day_du.pkl` (9.4 MB). |
| 2026-10-08 10:21 | **Toàn bộ notebook chạy xong** (562 phút ≈ 9 giờ 23 phút). Mọi ô có output; ô LightGBM vẫn lỗi `access violation` như cũ. Cảnh báo `_validate` trong log là `MissingIDFieldWarning` (ô thiếu trường id do script chèn ô), vô hại. |

## 21. Kết quả bản đầy đủ và một bất thường về thời gian

### 21.1 Bất thường: trong phần lớn thời gian chạy, máy chậm hơn bình thường, Newton chậm tới khoảng 33 lần

| Đo | Lúc bình thường (các lần chạy trước, và đo lại 10:22 sau khi chạy xong) | Trong ô 6c lần này |
|---|---|---|
| Newton, giây/vòng | 0.31–0.34 | **10.7–11.5** |
| GD, giây/vòng | 0.033–0.034 | 0.053 |
| GD backtracking β = 0.1, giây/vòng | 0.058 | 0.090 |
| SGD, giây/epoch | 0.74–0.75 | 0.73–0.93 |
| Subgradient / ISTA / FISTA (chạy **cuối** ô 6c), giây/vòng | 0.034 | 0.032–0.035 (bình thường) |

- Các ô **trước** 6c cũng có dấu hiệu chậm: quét subgradient và ISTA mất khoảng 26 giây so với 16 giây trước đây; sklearn saga mất 100 giây so với 55 giây. Ô quét Newton (khoảng 01:10) thì vẫn bình thường (0.32 giây/vòng).
- Như vậy máy bị tải ngoài trong khoảng từ khoảng 01:20 đến khi ô 6c bắt đầu chạy phần Lasso. Phép nhân ma trận lớn đa luồng của Newton (dựng Hessian) bị ảnh hưởng nặng nhất, vì các luồng BLAS phải tranh CPU với tiến trình khác.
- Chưa rõ nguyên nhân. Một dấu hiệu: lúc 10:22, Task Manager đã dùng khoảng 3 900 giây CPU.
- **Hệ quả:**
  - **Số vòng tới ε vẫn đúng.**
  - **Thời gian tới ε của các setup Ridge (GD, backtracking, Nesterov, Newton, SGD) bị đội lên**, nặng nhất là Newton. So thời gian giữa Ridge với Lasso, và giữa Newton với phần còn lại, trong lần chạy này **không đáng tin**.
  - Mục 7 vì vậy chọn "nhanh nhất tới 1e-6" là **Nesterov t=6/L (12.3 giây)** thay vì Newton. Đây là sai lệch do đo thời gian; ở các lần đo bình thường, Newton tới 1e-6 sau khoảng 2.4 giây. MAP@7 không đổi (0.797596 / 0.022517), vì ô huấn luyện 24 mô hình vẫn dùng Newton backtracking.

### 21.2 Kết quả không bị ảnh hưởng: số vòng tới ε và phần Lasso

**Best theo số vòng tới ε = 1e-5** (giây chỉ để tham khảo, phần Ridge bị đội):

| Thuật toán | Setup tốt nhất | Vòng tới 1e-5 | So với setup tốt nhất cũ |
|---|---|---|---|
| Newton | t = 1 (backtracking trùng hẳn) | 7 | như cũ |
| Nesterov | **t = 6/L** (bước mới thêm) | 139 | t = 2/L: 244 vòng |
| FISTA (Lasso) | **t = 6/L** (bước mới thêm) | 147 (5.1 s, đo bình thường) | t = 2/L: 256 vòng, 8.5 s |
| GD backtracking | β = 0.1, c = 1e-4 | 217 | như cũ |
| GD cố định | **t = 6/L** (bước mới thêm) | 1 476 | t = 4/L: 2 217 vòng |
| SGD | B = 4096, 1/√k, α0 = 1 | 55 epoch | như bản đại diện |
| ISTA (Lasso) | t = 1.5/L | 6 657 (214 s) | lần đầu ISTA đạt 1e-5 |
| Subgradient (Lasso) | α0 = 30 | 11 734 (379 s) | lần đầu subgradient đạt 1e-5 |

**Rút ra:**
1. **Bước lớn hơn 2/L tốt hơn ở mọi thuật toán bậc nhất:**
   - Không setup nào trong lưới mở rộng phân kỳ, kể cả 6/L.
   - Bước tốt nhất nằm ở **mép trên của lưới (6/L)** với GD, Nesterov và FISTA, nên bước tối ưu thật có thể còn lớn hơn.
   - Khớp với mục 11: backtracking tự chọn bước khoảng 25/L, vì L tính theo chặn σ' ≤ 1/4 quá lỏng.
2. **Hệ số Armijo c gần như không ảnh hưởng:** với β = 0.1, c từ 1e-4 đến 0.1 cho 478–485 vòng, c = 0.3 cho 500 vòng. **β mới là tham số quan trọng:** β = 0.1 cần 478 vòng, β = 0.9 cần 739 vòng.
3. **SGD:** α0 tốt nhất **đổi theo tổ hợp (batch, lịch) và theo ε**, xác nhận đúng lý do phải dò α0 cho từng tổ hợp (mục 17). Ở ε = 1e-5:

   | Batch | Lịch | α0 tốt nhất |
   |---|---|---|
   | 1024 | 1/√k | 0.2 |
   | 4096 | 1/√k | 1 |
   | 16384 | 1/√k | 5 |
   | 16384 | cố định | 1 |

   Lịch 1/√k xuống sâu nhất, tới 1e-7 với B = 1024 và B = 4096. Bước cố định kẹt ở khoảng 1e-6–1e-5, đúng với hiện tượng quả cầu nhiễu.
4. **Newton t = 0.1** không đạt 1e-4 trong 600 giây. Một phần là do chậm bất thường; với tốc độ bình thường, t = 0.1 cần khoảng 109 vòng (số liệu chạy thử).
5. F* Lasso từ FISTA 600 giây: 0.046538162369. Min sau 300 giây và sau 600 giây lệch 2.6e-10, nhất quán với các lần trước.

### 21.3 Các ô khác (giống các lần trước)

- Quét tham số, tổng kết Ridge/Lasso, thí nghiệm λ: số liệu F, số vòng và κ **trùng** lần chạy trước. Chỉ cột giây của vài ô chậm hơn, do bất thường ở 21.1.
- **Bảng log-loss mới (setup tốt nhất):**

  | Phương pháp | Train | Val |
  |---|---|---|
  | GD cố định | 0.043159 | 0.038855 |
  | Backtracking | 0.040929 | 0.036530 |
  | Nesterov | 0.040421 | 0.036007 |
  | Newton | 0.040680 | 0.036263 |
  | SGD | 0.040939 | 0.036546 |
  | sklearn lbfgs | 0.040654 | 0.036241 |
  | Subgradient | 0.042725 | 0.038013 |
  | FISTA | 0.040801 | 0.036070 |
  | sklearn saga | 0.040952 | 0.036256 |

  Các phương pháp đã hội tụ chỉ chênh nhau ở chữ số thứ 4.
- **Chi phí mỗi vòng (bảng mới):** phần Ridge bị đội như ở 21.1, nên chưa dùng được bản này.

## 22. Chạy lại 4 nhóm Ridge bị ảnh hưởng (GD cố định, GD backtracking, Nesterov, Newton)

- **Sửa ô 6c:** `CHAY_LAI_6C` nhận thêm dạng **list tên thuật toán**:
  - `False`: có file thì đọc lại;
  - `True`: chạy lại tất cả;
  - list, ví dụ `["Newton"]`: đọc file, chỉ chạy lại các setup của những thuật toán đó, giữ nguyên phần còn lại, gộp rồi lưu đè file.
  
  Đã chạy thử trên mẫu nhỏ: chạy lại đúng 36 setup của 4 nhóm rồi lưu.
- Bản sao file kết quả trước khi chạy lại được giữ ở thư mục tạm (`kq_6c_day_du.ban_dau.pkl`).
- SGD không chạy lại: chỉ chậm 0–20% và được so theo epoch.

| Thời điểm | Sự kiện |
|---|---|
| 2026-10-08 10:58 | Tải CPU lúc bắt đầu khoảng 10%, Task Manager đã đóng. Bắt đầu chạy toàn bộ notebook với `CHAY_LAI_6C = ["GD cố định", "GD backtracking", "Nesterov", "Newton"]`. Dự kiến khoảng 40 phút cho các ô khác, cộng khoảng 1–1.5 giờ cho 36 setup chạy lại. |
| 2026-10-08 14:11 | **Chạy xong** (194 phút, lâu hơn dự kiến). Ô 6c chạy lại đúng 36 setup rồi lưu đè `kq_6c_day_du.pkl` (9.47 MB). Mọi ô có output; LightGBM vẫn lỗi như cũ. Sau đó đặt lại `CHAY_LAI_6C = False` trong mã nguồn (không chạy lại), để lần chạy sau chỉ đọc file. |

### 22.1 Tốc độ sau khi chạy lại

| Nhóm | Giây/vòng trong 6c lần 1 | Trong 6c lần chạy lại | Đo riêng (bình thường) | Đánh giá |
|---|---|---|---|---|
| Newton | 10.7–11.5 | **0.31–0.43** | 0.31–0.34 | đã đúng |
| Nesterov t = 2–6/L | 0.053 | **0.033** | 0.033 | đã đúng |
| Nesterov t = 0.1–1.5/L | 0.053 | 0.046–0.051 | 0.033 | vẫn chậm khoảng 1.5 lần |
| GD cố định (mọi t) | 0.053 | 0.051 | 0.034 | vẫn chậm khoảng 1.5 lần |
| GD backtracking | 0.090 | 0.088 | 0.058 | vẫn chậm khoảng 1.5 lần |

- Log tải CPU (đo 5 phút một lần): từ 11:00 đến 11:56, tổng tải 40–60%, có lúc phiên điều khiển từ xa (`remoting_host` của Chrome Remote Desktop, `anydesk`, `remote_assistance_host`) chiếm nhiều CPU. Từ 12:00 trở đi tải ổn định ở 22–34%, chủ yếu là kernel Python của notebook.
- Không khớp được chắc chắn từng setup với từng khung giờ, nên **chưa xác định được nguyên nhân** làm GD và backtracking chậm. Điểm lạ là ở ô quét GD (mục 3) **cùng lần chạy**, GD vẫn đạt 0.032 giây/vòng.
- **Ảnh hưởng tới kết luận:** thứ hạng thời gian **không đổi** dù chia thời gian của GD và backtracking cho 1.5. Ví dụ ở ε = 1e-5:
  - backtracking: 19.1 giây, quy về khoảng 12.7 giây;
  - Nesterov: 4.6 giây;
  - Newton: 2.2 giây.

  Số vòng tới ε không bị ảnh hưởng.
- Bảng "chi phí mỗi vòng" lấy GD (0.0514) làm mốc, nên cột "× vòng GD" của các thuật toán khác **bị đánh giá thấp** khoảng 1.5 lần. Quy về mốc đo riêng 0.034: Newton khoảng 9.6 vòng GD, SGD khoảng 24, backtracking khoảng 1.7 (dùng 0.058 đo riêng), Nesterov/ISTA/FISTA khoảng 1.

### 22.2 Bảng best theo ε (sau khi chạy lại; giây / vòng [setup])

| Thuật toán | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|
| Newton | 1.6 / 5 [t=1] | 1.9 / 6 | **2.2 / 7** | **2.2 / 7** | **2.5 / 8** |
| Nesterov | 1.3 / 39 [t=6/L] | 2.5 / 74 | 4.6 / 139 | 7.7 / 230 | 12.5 / 376 |
| FISTA (Lasso) | 1.6 / 46 [t=6/L] | 3.0 / 85 | 5.1 / 147 | 11.9 / 345 | 17.6 / 508 |
| GD backtracking* | 6.3 / 72 [β=0.1] | 12.7 / 144 | 19.1 / 217 | 26.4 / 300 | 33.7 / 383 |
| SGD | 1.6 / 2 [B=1024 1/√k α0=1] | 10.5 / 13 [như trên] | 51.3 / 55 [B=4096 1/√k α0=1] | 111.5 / 120 [như trên] | 228.9 / 248 [như trên] |
| GD cố định* | 19.0 / 371 [t=6/L] | 44.3 / 863 | 75.8 / 1476 | 109.9 / 2141 | 146.1 / 2846 |
| ISTA (Lasso) | 52.4 / 1370 [t=2/L] | 137.8 / 4289 [t=1.5/L] | 214.3 / 6657 [t=1.5/L] | 309.4 / 7324 [t=2/L] | 417.4 / 10420 [t=2/L] |
| Subgradient (Lasso) | 120.8 / 3730 [α0=8] | 219.6 / 6796 [α0=30] | 379.3 / 11734 [α0=30] | – | – |

\* thời gian đội khoảng 1.5 lần (mục 22.1).

- **Mục 7** giờ chọn **Newton (t=1), tới 1e-6 sau 2.19 giây**, đúng như các lần đo bình thường. MAP@7 không đổi (0.797596 / 0.022517).
- Ở ε = 1e-3, Nesterov 6/L (1.3 giây) nhanh hơn cả Newton (1.6 giây). Từ ε = 1e-4 trở đi, Newton dẫn đầu.

### 22.3 Danh sách setting 5 best nếu chọn theo t_ε ở ε = 1e-6 (theo note)

| Thuật toán | Theo cách chọn hiện tại ở mục 6 (`chon()`, TOL) | Theo t_ε, ε = 1e-6 (6c đầy đủ) |
|---|---|---|
| GD cố định | t = 4/L | **t = 6/L** |
| GD backtracking | β = 0.1, c = 1e-4 | β = 0.1, c = 1e-2 (c = 1e-4 cho cùng số vòng 300, chỉ lệch 0.0x giây) |
| Nesterov | t = 2/L | **t = 6/L** |
| Newton | t = 1 | t = 1 |
| SGD | α0 = 0.2, cố định, batch 4096 | **α0 = 1, 1/√k, batch 4096** (đúng như note dự đoán) |

**Chưa sửa** markdown mục 6 và hình tổng kết. Hình tổng kết vẫn chọn best bằng `chon()` trên các ô quét (ngân sách 500 vòng, lưới cũ), nên nếu chỉ đổi danh sách chữ thì sẽ lệch với hình. Muốn khớp thì hình tổng kết phải lấy best và đường (F, t) từ `KQ_6C`; việc này cần chuyển ô 6c lên trước mục 6 hoặc đưa hình tổng kết xuống sau 6c.

## 23. Chạy riêng ngoài notebook các nhóm còn lệch thời gian, rồi gộp vào file kết quả

- **Script mới `chay_6c_rieng.py`** (trong repo):
  - lấy đúng mã từ `bai_lam.ipynb`: dữ liệu, các hàm thuật toán, hằng số lưới (chỉ các dòng gán, không chạy ô quét) và ô 6c;
  - chạy ô 6c với `CHAY_LAI_6C = [danh sách thuật toán truyền vào]`, nên đọc `kq_6c_day_du.pkl`, chạy lại riêng các nhóm đó, gộp rồi lưu đè;
  - sau đó notebook để `CHAY_LAI_6C = False` sẽ đọc file đã gộp.
  
  Cách dùng: `python chay_6c_rieng.py "GD cố định" "GD backtracking" "Nesterov"`. Phải chạy tuần tự, không song song.
- **Lý do:** lần đo riêng bằng một tiến trình mới (10:22) cho tốc độ bình thường, còn trong notebook GD và backtracking vẫn chậm khoảng 1.5 lần.
- Bản sao file kết quả trước bước này: `kq_6c_day_du.sau_chay_lai_1.pkl` (thư mục tạm).

| Thời điểm | Sự kiện |
|---|---|
| 2026-10-08 14:31 | Lần chạy đầu của script lỗi ngay ở dòng in đầu tiên (`UnicodeEncodeError`: stdout trên Windows mặc định cp1252). File kết quả chưa bị động tới. Sửa: script đặt `sys.stdout.reconfigure(encoding="utf-8")`. |
| 2026-10-08 14:32 | Chạy lại `chay_6c_rieng.py "GD cố định" "GD backtracking" "Nesterov"` (28 setup), dự kiến khoảng 1–1.2 giờ. |
| 2026-10-08 16:09 | **Script xong** (96 phút). Đã chạy lại 29 setup: GD 9, backtracking 11 (5 β ở c = 1e-4, cộng 2 β × 3 giá trị c khác), Nesterov 9. Lưu đè `kq_6c_day_du.pkl` (10.1 MB). Lưới đầy đủ thực tế có **73 setup** (trước tôi ước 71, vì đếm thiếu 2 setup backtracking). |

### 23.1 Tốc độ: đã bình thường

| Nhóm | Trong notebook (lần chạy lại 14:11) | Script riêng | Đo riêng trước đó |
|---|---|---|---|
| GD cố định | 0.051 | **0.033** | 0.034 |
| GD backtracking β = 0.1 | 0.088 | **0.056** | 0.058 |
| Nesterov (mọi t) | 0.033–0.051 | **0.033** | 0.033 |

- Log CPU lúc chạy script: tổng tải khoảng 52–60% suốt thời gian chạy, vẫn có phiên điều khiển từ xa. Vậy phiên điều khiển từ xa **không phải** nguyên nhân chính.
- Khi chạy trong notebook ở các đoạn bị chậm, tổng tải chỉ khoảng 22–34%. Như vậy trong những đoạn đó, **chính kernel notebook dùng ít nhân CPU hơn** (phép nhân ma trận chạy ít luồng hơn). Nguyên nhân cụ thể chưa rõ.
- Dù vậy, số vòng tới ε trùng hoàn toàn giữa các lần chạy, chỉ thời gian khác.

### 23.2 Nguồn gốc các nhóm trong `kq_6c_day_du.pkl` (bản cuối)

| Nhóm | Chạy ở đâu, lúc nào | Tốc độ |
|---|---|---|
| GD cố định, GD backtracking, Nesterov | `chay_6c_rieng.py`, 2026-10-08 14:33–16:09 | bình thường |
| Newton | notebook, lần chạy lại 2026-10-08 10:58–14:11 | bình thường (0.31–0.43 giây/vòng) |
| SGD, Subgradient, ISTA, FISTA | notebook, lần chạy đầy đủ 2026-10-08 01:55–10:20 | SGD chậm 0–20% (được so theo epoch); Lasso bình thường |

### 23.3 Bảng best theo ε, bản cuối (giây / vòng [setup]; SGD: vòng = epoch)

| Thuật toán | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|
| **Newton** [t=1] | 1.6 / 5 | 1.9 / 6 | **2.2 / 7** | **2.2 / 7** | **2.5 / 8** |
| **Nesterov** [t=6/L] | **1.3 / 39** | 2.5 / 74 | 4.6 / 139 | 7.7 / 230 | 12.5 / 376 |
| FISTA, Lasso [t=6/L] | 1.6 / 46 | 3.0 / 85 | 5.1 / 147 | 11.9 / 345 | 17.6 / 508 |
| GD backtracking [β=0.1, c=0.01] | 4.1 / 72 | 8.1 / 144 | 12.2 / 217 | 16.8 / 300 | 21.4 / 383 |
| SGD | 1.6 / 2 [B=1024 1/√k α0=1] | 10.5 / 13 [như trên] | 51.3 / 55 [B=4096 1/√k α0=1] | 111.5 / 120 [như trên] | 228.9 / 248 [như trên] |
| GD cố định [t=6/L] | 12.3 / 371 | 28.6 / 863 | 49.0 / 1476 | 71.2 / 2141 | 94.6 / 2846 |
| ISTA, Lasso | 52.4 / 1370 [t=2/L] | 137.8 / 4289 [t=1.5/L] | 214.3 / 6657 [t=1.5/L] | 309.4 / 7324 [t=2/L] | 417.4 / 10420 [t=2/L] |
| Subgradient, Lasso | 120.8 / 3730 [α0=8] | 219.6 / 6796 [α0=30] | 379.3 / 11734 [α0=30] | – | – |

**Chi phí mỗi vòng ở tốc độ bình thường** (quy ra số vòng GD): GD 1, Nesterov 1, ISTA/FISTA/subgradient khoảng 1, backtracking **2.1**, Newton khoảng **9.7** (0.32 giây), SGD khoảng **24 / epoch** (0.8 giây).

**Kết luận về thời gian tới ε (Ridge):**
- **ε ≤ 1e-4:** Newton nhanh nhất.
- **ε = 1e-3:** Nesterov 6/L nhanh nhất (1.3 giây so với 1.6 giây của Newton).
- **Ở mọi mức ε:** Nesterov nhanh hơn backtracking khoảng 1.7–3 lần, và nhanh hơn GD cố định khoảng 7–10 lần.
- **SGD** chỉ cạnh tranh ở ε = 1e-3.
| 2026-10-08 16:09 | Chạy lại cả notebook với `CHAY_LAI_6C = False`: ô 6c chỉ đọc file đã gộp, để output trong notebook là bảng cuối. Dự kiến khoảng 45 phút. |
| 2026-10-08 16:45 | **Notebook chạy xong** (khoảng 35 phút). Ô 6c đọc `kq_6c_day_du.pkl` (bản ghép lúc 16:09, 73 setup) và in đúng bảng ở mục 23.3. Mục 7 chọn Newton (t=1), tới 1e-6 sau 2.19 giây; MAP@7 không đổi (0.797596 / 0.022517). Chỉ ô LightGBM lỗi như cũ. |

## 24. Vẽ lại hình tổng kết và hình cặp đôi từ dữ liệu 6c, thêm ghi chú cho hình quét

Theo note "vẽ hình tổng kết và hình cặp đôi từ dữ liệu 6c". **Không chạy lại 6c**: vẫn `CHAY_DAY_DU = True`, `CHAY_LAI_6C = False`, ô 6c chỉ đọc `kq_6c_day_du.pkl` (bản ghép lúc 16:09, 73 setup). ε chính = **1e-6** (`EPS_CHINH`).

| Thời điểm | Sự kiện |
|---|---|
| 2026-10-08, trước 20:46 | Chạy thử mọi ô (trừ LightGBM) trên mẫu 20 000 dòng. Ô 6c đọc **file pkl tạm** trong thư mục tạm (bản sao, F Ridge dịch theo F* của mẫu để t_ε giữ nguyên, TRAN = 5); `kq_6c_day_du.pkl` không bị ghi. Không lỗi; sửa một lỗi nhãn ("cố, định"). |
| 2026-10-08 20:46 | Chạy toàn bộ notebook (`nbconvert --execute --inplace --allow-errors`). |
| 2026-10-08 21:35 | **Xong** (49 phút). Chỉ ô LightGBM lỗi như cũ. Mục 7: Newton (t=1), 2.19 giây; MAP@7 không đổi (0.797596 / 0.022517). |

### 24.1 Ô đã di chuyển / sửa / thêm

Thứ tự mới: … mục 5b → **6c** → **6d** (mới: 3 hình cặp đôi) → **6e** (tổng kết, trước là "mục 6") → thí nghiệm λ → Phần II.

| Ô | Việc |
|---|---|
| Ô vẽ (`ve`) | Thêm `x_max` (cắt trục hoành bằng xlim, kể cả khi log_x; không cắt dữ liệu), `ngang=(y, nhãn)` (đường ngang chấm xám, có trong legend). `ghi_chu` nay đặt **dưới** khung hình. Thêm `so_e` (1e-06 → "1e-6") và `ghi_quet(ngân sách)`. |
| Ô quét GD, backtracking, GD vs backtracking, Nesterov, Newton gộp, SGD α0, SGD lịch × α0, subgradient, ISTA/FISTA | Chỉ thêm `ghi_chu` "Dừng khi ‖r‖ ≤ 1e-3·‖r₀‖ hoặc hết <500 vòng / 30 vòng Newton / 42 epoch>; điểm dừng giữa các đường không cùng mức F − F*." (Newton gộp: "30 vòng Newton / 500 vòng Nesterov", giữ thêm dòng "backtracking luôn nhận bước t = 1…"). Dữ liệu không đổi. |
| 6c (code) | Giữ nguyên phần chạy/đọc và các bảng; **thêm ở cuối** các hàm dùng chung: `EPS_CHINH`, `GHI_6C`, `best_6c` (t_ε nhỏ nhất; không setup nào đạt thì lùi sang ε lớn kế tiếp), `lay_6c` (lấy đúng setup, thiếu thì in tên và dừng), `ten_setup`, `nhan_6c`, `x_toi`. |
| 6c (markdown) | Câu đầu sửa thành "Các ô quét ở mục 3–5 dừng theo…"; thêm câu chỉ sang 6d/6e. |
| 6d (mới) | Chuyển 3 cặp ô (markdown + code) **GD vs Nesterov**, **SGD vs GD**, **ISTA vs FISTA** từ mục 3b/4/5b xuống sau 6c (cách ít xáo trộn nhất: giữ nguyên tên file hình và kiểu hình, chỉ đổi nguồn dữ liệu sang `KQ_6C`). Markdown SGD vs GD và ISTA/FISTA sửa cho khớp cách chọn mới. |
| 6e (trước là mục 6) | Markdown viết lại: best theo t_ε ở 1e-6, danh sách setting 5 best mới, nguồn gốc số liệu thời gian, nhiễu đo ở 1e-3. Ô Ridge, Lasso, log-loss viết lại hoàn toàn (không dùng `chon()`). sklearn lbfgs và saga **đo lại trong ô**. |
| Thí nghiệm λ | Chuyển xuống sau 6e (trước nằm giữa mục 6 và 6c), không sửa. |

### 24.2 Best mới (đọc từ `KQ_6C`, ε = 1e-6)

**Ridge**

| Thuật toán | Setup | Vòng tới ε | giây tới ε | F − F* min | Tổng vòng / giây |
|---|---|---|---|---|---|
| GD cố định | t = 6/L | 2141 | 71.15 | 1.0e-08 | 3609 / 120.0 |
| GD backtracking | β = 0.1, c = 0.01 | 300 | 16.82 | 9.7e-09 | 478 / 26.8 |
| Nesterov | t = 6/L | 230 | 7.65 | 9.8e-09 | 750 / 25.0 |
| Newton | t = 1 | 7 | **2.19** | 3.9e-10 | 8 / 2.5 |
| SGD | B = 4096, 1/√k, α0 = 1 | 120 epoch | 111.53 | 3.9e-08 | 649 / 600.7 |
| sklearn lbfgs (đo lại) | mặc định | 27 vòng (cả lần fit) | 1.23 | 9.1e-06 | – |

**Lasso**

| Thuật toán | Setup | ε dùng | Vòng tới ε | giây tới ε | F − F* min |
|---|---|---|---|---|---|
| Subgradient | α0 = 30 | **1e-5** | 11734 | 379.27 | 6.3e-06 |
| ISTA | t = 2/L | 1e-6 | 7324 | 309.42 | 2.0e-08 |
| FISTA | t = 6/L | 1e-6 | 345 | **11.91** | 9.7e-09 |
| sklearn saga (đo lại) | L1 | – | 100 (max_iter, có ConvergenceWarning) | 58.74 | 8.4e-07 |

Hình cặp đôi:
- **GD vs Nesterov**, **ISTA vs FISTA**: t ∈ {0.5, 1, 2}/L, đủ trong `KQ_6C`. Mốc subgradient = α0 = 30 (chọn ở ε = 1e-5).
- **SGD vs GD**: SGD cố định → B = 16384, α0 = 0.2 (298.0 giây / 373 epoch tới 1e-6); SGD 1/√k → B = 4096, α0 = 1 (111.5 giây / 120 epoch); GD 6/L, Nesterov 6/L. 1 epoch SGD = 0.815 giây ≈ 25 vòng GD (0.0333 giây).
- Trục cắt: hình theo vòng ở 500 vòng (SGD vs GD: 500 lượt, log_x); hình theo thời gian ở thời gian tương ứng (vòng 500 của đường chậm nhất; SGD vs GD: max(SGD tới epoch 42, GD/Nesterov tới vòng 500) ≈ 38 giây).
- Hình tổng kết: log_x cả hai trục, cắt ở 1.1 × lúc đường chậm nhất trong các best đạt 1e-7 (Ridge: SGD 229 giây / GD 2846 vòng; Lasso: ISTA 417 giây / 10420 vòng; subgradient không tới 1e-7 nên không tính).

### 24.3 Log-loss (không cộng phạt) của setup best mới

Chạy lại đúng setup best (cùng tham số, cùng seed) tới khi F − F* ≤ ε đã dùng để chọn (hoặc hết 600 giây) để lấy w.

| Phương pháp | Bài | Setup | Dừng ở ε | F − F* | log-loss train | log-loss val | Hệ số ≠ 0 | giây chạy lại |
|---|---|---|---|---|---|---|---|---|
| GD cố định | Ridge | t = 6/L | 1e-6 | 1.0e-06 | 0.040623 | 0.036205 | 154/160 | 72.5 |
| GD backtracking | Ridge | β = 0.1, c = 0.01 | 1e-6 | 9.3e-07 | 0.040621 | 0.036204 | 154/160 | 17.0 |
| Nesterov | Ridge | t = 6/L | 1e-6 | 9.6e-07 | 0.040613 | 0.036194 | 154/160 | 15.5 |
| Newton | Ridge | t = 1 | 1e-6 | 1.9e-07 | 0.040591 | 0.036173 | 154/160 | 2.2 |
| SGD | Ridge | B = 4096, 1/√k, α0 = 1 | 1e-6 | 9.6e-07 | 0.040607 | 0.036196 | 154/160 | 95.0 |
| sklearn lbfgs | Ridge | mặc định | – | 9.1e-06 | 0.040654 | 0.036241 | 154/160 | 1.2 |
| Subgradient | Lasso | α0 = 30 | 1e-5 | 1.0e-05 | 0.040886 | 0.036191 | 122/160 | 395.8 |
| ISTA | Lasso | t = 2/L | 1e-6 | 1.0e-06 | 0.040955 | 0.036260 | 19/160 | 245.0 |
| FISTA | Lasso | t = 6/L | 1e-6 | 9.6e-07 | 0.040945 | 0.036246 | 20/160 | 23.2 |
| sklearn saga | Lasso | L1 | – | 8.4e-07 | 0.040952 | 0.036256 | 19/160 | 58.7 |

Các phương pháp Ridge chênh nhau ở chữ số thứ 4–5; Lasso: subgradient không cho hệ số đúng bằng 0 (122/160 ≠ 0), còn ISTA/FISTA/saga thưa như nhau (19–20/160).

### 24.4 Chỗ khác với kỳ vọng / cần lưu ý

- **Subgradient không có setup nào tới 1e-6** trong 600 giây → `best_6c` lùi sang ε = 1e-5 và chọn α0 = 30 (đúng bảng 23.3). Ghi rõ trong bảng, ghi chú dưới hình Lasso và dòng in ở ô ISTA/FISTA.
- **SGD vs GD chỉ còn 2 đường SGD** (cố định, 1/√k): lưới 6c không có lịch 1/k. Best lịch cố định là **B = 16384** (không phải 4096).
- **sklearn đo lại**: lbfgs 1.23 giây (27 vòng), saga 58.7 giây và chạm `max_iter = 100` (ConvergenceWarning) nhưng vẫn đạt F − F* = 8.4e-7.
- **Cột "giây chạy lại" ở bảng log-loss khác t_ε**: đó là thời gian đồng hồ thật của cả lần gọi. Ví dụ Nesterov 15.5 giây so với t_ε = 7.65 giây, vì gradient tại w (chỉ để ghi F / kiểm tra dừng) không tính vào t của Nesterov/FISTA; ISTA 245 giây so với 309 giây ở 6c là nhiễu đo thời gian.
- Trục vòng lặp vẫn theo quy ước cũ của `ve`: điểm thứ k của lịch sử F vẽ tại x = k + 1 (để dùng được log_x).
- Best ở 1e-5 và 1e-6 trùng nhau cho mọi thuật toán Ridge, như dự kiến.

### 24.5 Hình cũ trong `overleaf/hinh/` không còn được tạo (chưa xoá)

`best_so_sanh.pdf`, `gd_bt_vonglap_tach.pdf`, `newton_bt_thoigian.pdf`, `newton_bt_vonglap.pdf`, `newton_codinh_thoigian.pdf`, `newton_codinh_vonglap.pdf`, `sgd_muot_vs_batch.pdf`, `ss_bt_buoc.pdf`, `ss_ridge_lasso_thoigian.pdf`, `ss_ridge_lasso_vonglap.pdf`. (Các file này đã không còn được tạo từ trước lần sửa này.)

## 25. Mọi hình khớp với kết quả 6c (không còn "tốt nhất" lệch)

**Vấn đề:** sau mục 24 vẫn còn hai chỗ lệch với 6c.
- `ss_gd_bt_*` ghi "GD cố định t = 4/L (tốt nhất)" và `ss_newton_*` ghi "Nesterov t = 2/L (tốt nhất)", trong khi best theo t_ε ở 6c là **6/L** cho cả hai.
- Nguyên nhân: các hình quét ở mục 3–5 dùng lưới cũ (GD tới 4/L; Nesterov, FISTA tới 2/L), còn 6c có thêm `BUOC_THEM = [2.5, 3, 4, 6]`.

**Đã sửa:**

| Chỗ | Sửa |
|---|---|
| Ô quét GD (mục 3) | `BUOC_THEM` chuyển về đây; `BUOC_GD = sorted(set(BUOC + BUOC_THEM))` = {0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6}/L. Ô 6c dùng lại biến này (lưới 6c **không đổi**). `chay_6c_rieng.py` lấy thêm dòng `BUOC_THEM`. |
| Ô quét Nesterov, ISTA/FISTA | `BUOC_NES = sorted(set(BUOC + BUOC_THEM))` cho Nesterov và FISTA; ISTA giữ `BUOC` (như 6c). |
| `ve()` | Quá 8 đường thì tự thêm kiểu nét (đường thứ 9 dùng lại màu 1 với nét đứt), theo quy tắc màu. |
| `ss_gd_bt_*` | Chuyển xuống 6d, vẽ từ `KQ_6C`: GD 1/L (tiêu chuẩn), GD 6/L (best t_ε), backtracking 5 β (c = 1e-4) + β = 0.1, c = 0.01 (best t_ε). |
| `ss_newton_*` | Chuyển xuống 6d, vẽ từ `KQ_6C`: Newton t ∈ {0.1, 0.25, 0.5, 1}, bt β ∈ {0.1, 0.5, 0.9}, mốc Nesterov 6/L (best t_ε). Kiểm tra "backtracking trùng t = 1" tính lại trên dữ liệu 6c: vẫn **True**. |
| Markdown | Hai đoạn so sánh sửa "tốt nhất đã quét" thành "tốt nhất theo t_ε ở 6c"; 6d: "Năm hình". |

Thứ tự 6d: GD vs backtracking → GD vs Nesterov → Newton → SGD vs GD → ISTA vs FISTA.

| Thời điểm | Sự kiện |
|---|---|
| 2026-10-09 00:21 | Chạy thử mẫu 20 000 dòng không lỗi; bắt đầu chạy toàn bộ notebook. |
| 2026-10-09 01:13 | **Xong** (52 phút). Chỉ ô LightGBM lỗi như cũ. |

**Kiểm tra sau khi chạy:**
- Ô quét GD (500 vòng): 6/L thấp nhất (F − F* = 5.0e-4 so với 1.3e-3 của 4/L).
- Ô quét Nesterov: 6/L hội tụ nhanh nhất (91 vòng, 3.04 giây).
- Hình FISTA quét: 6/L nhanh nhất.
- Như vậy hình quét, hình so sánh và 6c cùng một kết luận.
- Lưu ý: ở `ss_gd_bt`, đường backtracking β = 0.1, c = 1e-4 gần như trùng đường best (c = 0.01): cùng 300 vòng tới 1e-6, chỉ lệch 0.07 giây.
- Còn mở: 6/L là bước lớn nhất trong lưới, nên chưa biết bước lớn hơn (8/L, 12/L) có nhanh hơn không.

## 26. Thêm 6/L vào hình GD vs Nesterov (`ss_gd_nesterov_*`)

- Ô GD vs Nesterov (6d): `K_SS = [0.5, 1, 2, 6]`. 6/L là best theo t_ε của cả GD và Nesterov. Hình có 4 màu × 2 kiểu nét (8 đường), dữ liệu lấy từ `KQ_6C`. Markdown thêm một câu ghi chú điều này.
- Không chạy lại cả notebook. Script `chay_mot_o.py` (thư mục tạm) chạy riêng ô này trong một kernel mới:
  - kernel chỉ nạp import, ô cấu hình, hàm vẽ, hằng số lưới và ô 6c (đọc pkl, F* Ridge lấy từ pkl), không tải dữ liệu;
  - ô GD vs backtracking chạy theo vì định nghĩa `X_CAT`, `giay_tai`, nên `ss_gd_bt_*` được ghi lại với dữ liệu y hệt;
  - output được chép vào đúng ô trong `bai_lam.ipynb`.
- Đọc từ hình, sau 500 vòng:

  | Bước | GD cố định | Nesterov |
  |---|---|---|
  | 6/L | khoảng 5e-4 | khoảng 2e-7 |
  | 2/L | khoảng 5e-3 | khoảng 1e-6 |

  Cùng bước, khoảng cách giữa GD và Nesterov vẫn là 3–4 bậc độ lớn.

## 27. Thêm bước lớn cho ISTA và FISTA: 6/L (ISTA), 19, 29, 32, 48/L (cả hai)

**Lý do:** 6/L là bước lớn nhất trong lưới và lại là best, nên cần thử bước lớn hơn. Backtracking tự chọn bước khoảng 25/L.

**Đã sửa:**
- Ô quét proximal: `BUOC_PX_LON = [19, 29, 32, 48]`, `BUOC_ISTA = BUOC + [6] + BUOC_PX_LON`, `BUOC_FISTA = BUOC_NES + BUOC_PX_LON`. Ô 6c dùng lại đúng các biến này, nên ô quét và 6c cùng lưới.
- `CHAY_LAI_6C` và `chay_6c_rieng.py` nhận thêm phần tử dạng `"Thuật toán|setup"` (ví dụ `"ISTA|t=6/L"`) để chạy hoặc thêm đúng một setup. Khi ghép, setup mới được xếp theo thứ tự lưới.
- Hình `ss_ista_fista_*` tự lấy bước: {0.5, 1, 2}/L cộng bước best theo t_ε của ISTA và của FISTA (đọc từ 6c), và in ra các bước đã vẽ.
- `ve()`: trục log mở rộng tới 1.5 × thời gian của sklearn, để điểm sklearn không bị cắt ở mép phải. Thay đổi này có hiệu lực từ lần chạy sau; hình `lasso_thoigian` hiện tại vẫn có điểm sklearn sát mép.

| Thời điểm | Sự kiện |
|---|---|
| 2026-10-09 01:47–01:51 | `chay_6c_rieng.py "ISTA\|t=6/L"`: 4 phút. ISTA 6/L hội tụ (tới 1e-6 sau 82.1 giây). |
| 2026-10-09 ~01:55 | Dừng lần chạy notebook đang dở, vì sẽ phải chạy lại sau khi thêm bước lớn. |
| 2026-10-09 02:07–02:13 | `chay_6c_rieng.py` 8 setup ISTA/FISTA × {19, 29, 32, 48}/L: 6 phút, **cả 8 hội tụ** (tới F* + 1e-8). `kq_6c_day_du.pkl` nay có **82 setup**. Bản sao trước mỗi bước ghép nằm trong thư mục tạm (`kq_6c_day_du.truoc_ista6.pkl`, `kq_6c_day_du.truoc_buoc_lon.pkl`). |
| 2026-10-09 02:13–03:02 | Chạy lại toàn bộ notebook (49 phút). Chỉ ô LightGBM lỗi như cũ. Mục 7 không đổi: Newton t=1, MAP@7 0.797596 / 0.022517. |

### 27.1 Các setup mới (giây tới ε; vòng = tổng số vòng tới F* + 1e-8)

| Setup | Vòng | Tổng giây | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|---|---|
| ISTA 6/L | 6571 | 219.5 | 14.9 | 35.6 | 55.6 | 82.1 | 117.1 |
| **ISTA 19/L** | 1923 | 64.4 | **4.6** | **11.0** | **15.7** | **20.9** | **28.7** |
| ISTA 29/L | 1298 | 44.1 | 10.6 | 17.9 | 23.7 | 29.7 | 35.8 |
| ISTA 32/L | 1231 | 42.0 | 12.7 | 19.7 | 25.3 | 30.6 | 35.9 |
| ISTA 48/L | 1351 | 45.9 | 26.4 | 31.7 | 35.4 | 39.0 | 42.4 |
| FISTA 19/L | 777 | 26.2 | 3.7 | 3.9 | 8.5 | 11.7 | 13.3 |
| FISTA 29/L | 588 | 19.6 | 2.9 | 3.1 | 5.6 | 9.5 | 10.7 |
| **FISTA 32/L** | 574 | 19.2 | 3.3 | 3.4 | 5.8 | **9.4** | **10.7** |
| FISTA 48/L | 514 | 17.2 | 3.3 | 3.4 | 5.3 | 10.2 | 12.2 |
| (so sánh) FISTA 6/L | 935 | 32.7 | **1.6** | **3.0** | **5.1** | 11.9 | 17.6 |

- **ISTA:** best ở mọi ε là **19/L**, nằm giữa lưới (29, 32, 48/L chậm dần), nên không còn ở biên. So với best cũ 2/L ở 1e-6: 309.4 → 20.9 giây, tức nhanh hơn khoảng 15 lần.
- **FISTA:**
  - ε ≥ 1e-5: 6/L vẫn nhanh nhất.
  - ε = 1e-6 và 1e-7: **32/L** nhanh nhất, nhưng chỉ hơn 29/L 0.1 giây (trong nhiễu đo) và hơn 6/L 2.5 giây.
- **Dao động:** bước lớn làm F tăng vọt mấy vòng đầu (F − F* lên tới khoảng 10, cao hơn F(0)). FISTA bước lớn dao động mạnh quanh F* (xem mục 27.3).

### 27.2 Bảng kết quả cập nhật (thay cho bảng 23.3 và 24.2 ở phần Lasso; phần Ridge không đổi)

**Best theo ε** (giây / vòng [setup]):

| Thuật toán | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|
| Newton [t=1] | 1.6 / 5 | 1.9 / 6 | 2.2 / 7 | **2.2 / 7** | **2.5 / 8** |
| Nesterov [t=6/L] | **1.3 / 39** | 2.5 / 74 | 4.6 / 139 | 7.7 / 230 | 12.5 / 376 |
| GD backtracking [β=0.1, c=0.01] | 4.1 / 72 | 8.1 / 144 | 12.2 / 217 | 16.8 / 300 | 21.4 / 383 |
| GD cố định [t=6/L] | 12.3 / 371 | 28.6 / 863 | 49.0 / 1476 | 71.2 / 2141 | 94.6 / 2846 |
| SGD | 1.6 / 2 [B=1024 1/√k α0=1] | 10.5 / 13 [như trên] | 51.3 / 55 [B=4096 1/√k α0=1] | 111.5 / 120 [như trên] | 228.9 / 248 [như trên] |
| **FISTA (Lasso)** | 1.6 / 46 [6/L] | 3.0 / 85 [6/L] | 5.1 / 147 [6/L] | **9.4 / 282 [32/L]** | **10.7 / 320 [32/L]** |
| **ISTA (Lasso)** | **4.6 / 138 [19/L]** | **11.0 / 330 [19/L]** | **15.7 / 471 [19/L]** | **20.9 / 626 [19/L]** | **28.7 / 859 [19/L]** |
| Subgradient (Lasso) | 120.8 / 3730 [α0=8] | 219.6 / 6796 [α0=30] | 379.3 / 11734 [α0=30] | – | – |

**Tổng kết Lasso ở ε = 1e-6** (thay bảng Lasso ở mục 24.2):

| Thuật toán | Setup | ε dùng | Vòng tới ε | giây tới ε | F − F* min |
|---|---|---|---|---|---|
| Subgradient | α0 = 30 | 1e-5 | 11734 | 379.27 | 6.3e-06 |
| ISTA | **t = 19/L** (cũ 2/L) | 1e-6 | 626 | **20.94** (cũ 309.42) | 1.0e-08 |
| FISTA | **t = 32/L** (cũ 6/L) | 1e-6 | 282 | **9.40** (cũ 11.91) | 8.6e-09 |
| sklearn saga (đo lại) | L1 | – | 100 (max_iter) | 66.12 | 8.4e-07 |

**Log-loss** (thay 3 dòng Lasso của bảng 24.3; các dòng Ridge không đổi):

| Phương pháp | Setup | Dừng ở ε | F − F* | log-loss train | log-loss val | Hệ số ≠ 0 | giây chạy lại |
|---|---|---|---|---|---|---|---|
| Subgradient | α0 = 30 | 1e-5 | 1.0e-05 | 0.040886 | 0.036191 | 122/160 | 378.8 |
| ISTA | t = 19/L | 1e-6 | 9.9e-07 | 0.040899 | 0.036204 | 19/160 | 20.2 |
| FISTA | t = 32/L | 1e-6 | 9.5e-07 | 0.040909 | 0.036209 | 20/160 | 18.2 |
| sklearn saga | L1 | – | 8.4e-07 | 0.040952 | 0.036256 | 19/160 | 66.1 |

- **Thời gian chạy notebook:** ô log-loss nhanh hơn trước khoảng 4 phút, vì ISTA best chỉ cần khoảng 20 giây thay vì 245 giây.
- **Nhiễu đo F* Lasso trong ô quét:** dòng "F* Lasso" in ở ô quét proximal là 0.046538164358. F* dùng cho 6c vẫn là 0.046538162369 (FISTA 600 giây). Không ảnh hưởng gì.

### 27.3 Lưu ý về định nghĩa t_ε khi đường dao động (chưa đổi, chờ người dùng quyết)

t_ε hiện là **lần đầu** F − F* ≤ ε. FISTA bước lớn chạm 1e-6 ở một đáy dao động rồi lại vọt lên trên. So với định nghĩa "**từ đó trở đi luôn** ≤ ε":

| Thuật toán | ε | Lần đầu chạm (hiện tại) | Luôn ≤ ε từ đó |
|---|---|---|---|
| FISTA | 1e-5 | 6/L, 5.1 giây | 6/L, 6.5 giây |
| FISTA | 1e-6 | **32/L, 9.4 giây** | **6/L, 11.9 giây** |
| Nesterov | 1e-6 | 6/L, 7.7 giây | 6/L, 9.5 giây |
| SGD | 1e-5 | B=4096 1/√k α0=1, 51.3 giây | B=1024 1/√k α0=0.2, 93.6 giây |
| SGD | 1e-6 | B=4096 1/√k α0=1, 111.5 giây | B=1024 1/√k α0=0.2, 526.0 giây |
| ISTA, GD, backtracking, Newton | 1e-5, 1e-6 | không đổi | không đổi |

### 27.4 Còn mở

- GD cố định 6/L và Nesterov 6/L vẫn nằm ở biên lưới.
- `overleaf/bao_cao.tex` và `README.md` vẫn ghi số liệu của các lần chạy rất cũ (ngân sách 250 vòng; best 4/L, 2/L; FISTA 99/231 vòng…). Cần cập nhật theo bảng 27.2.

## 28. Kiểm tra: λ_max dùng cho bước ISTA/FISTA phải là của H_S (Lasso), không phải H (Ridge)

**Vấn đề:** lập luận chọn bước lớn cho ISTA/FISTA (mục 27) dựa trên độ cong của bài **Ridge**:
- backtracking GD Ridge chọn khoảng 25/L;
- README ghi λ_max = 0.392 của H Ridge, suy ra 2/λ_max ≈ 32/L.

Với Lasso, prox-gradient chỉ bị giới hạn bởi độ cong của **phần trơn** ℓ (không có λI) trên **tập hỗ trợ** S = {j : w*_j ≠ 0} tại nghiệm Lasso: H_S = X_Sᵀ diag(σ(1−σ)) X_S / n.

**Kiểm tra:** script `kiem_tra_HS.py` (thư mục tạm) lấy đúng mã notebook. w*_Lasso tính bằng FISTA 6/L tới F − F* = 9e-11 (1685 vòng, 115 giây).

| Ma trận | Cỡ | λ_max | λ_min | κ | 2/λ_max (× 1/L) | t tối ưu 2/(λ_max+λ_min) (× 1/L) |
|---|---|---|---|---|---|---|
| Ridge: H(w*_R) = ∇²ℓ + λI′ | 161 | 0.3916 | 1.0e-3 | 392 | **31.9** | 31.8 |
| Lasso: ∇²ℓ(w*_L), mọi toạ độ | 161 | 0.3321 | ≈ 0 (suy biến) | ∞ | 37.6 | – |
| **Lasso: H_S(w*_L)**, \|S\| = 20 (gồm intercept) | 20 | **0.0964** | 2.9e-4 | 337 | **129.7** | 129.4 |

**Kết luận:**
- **Đúng là dùng nhầm.** Ngưỡng ổn định cục bộ của ISTA/FISTA trên Lasso là **khoảng 130/L**, gấp khoảng 4 lần ngưỡng khoảng 32/L suy từ H Ridge.
  - Mọi bước đã thử (≤ 48/L) đều nằm xa dưới ngưỡng, nên 32/L **không** phải "biên ổn định" của Lasso.
  - Không thể dùng ∇²ℓ trên mọi toạ độ thay cho H_S: ma trận đó suy biến (λ_min ≈ 0) khi bỏ λI.
- **Số liệu đo (t_ε, bảng 27.2) không bị ảnh hưởng.** Đó là kết quả chạy thật. Chỉ có lập luận và lựa chọn lưới bị ảnh hưởng.
- **Lý thuyết cục bộ** (hệ số co ρ(t) = max|1 − t·λ_i(H_S)|) dự đoán: bước càng lớn (tới khoảng 130/L) thì giai đoạn cuối càng nhanh.
  - Khớp quan sát: số vòng ISTA tới 1e-8 giảm từ 1923 (19/L) xuống 1231 (32/L).
  - Nhưng ở ε = 1e-6, 19/L lại thắng, vì bước lớn tốn **giai đoạn đầu** lâu hơn: tới 1e-3 mất 4.6 giây ở 19/L, 12.7 giây ở 32/L, 26.4 giây ở 48/L.
  - Nguyên nhân: tại w = 0 thì σ(1−σ) = 1/4, độ cong bằng đúng L. Tập hỗ trợ cũng chưa được xác định, nên bước lớn làm F vọt lên (F − F* tới khoảng 10) trước khi giảm.
  - Hệ số co từ λ_min(H_S) chỉ là chặn trường hợp xấu nhất. Dự đoán khoảng 2600 vòng mỗi bậc ở 19/L, trong khi thực tế khoảng 230 vòng mỗi bậc (từ 1e-6 xuống 1e-7). Vì vậy không dùng nó để dự đoán số vòng.
- **Đã sửa chú thích** ở ô quét proximal (mã không đổi, không cần chạy lại).
- **Còn mở:** nếu muốn kiểm tra sát ngưỡng thật thì thử thêm ISTA/FISTA ở 64, 96, 128/L.

## 29. Thử ISTA/FISTA sát ngưỡng cục bộ của H_S: t ∈ {64, 96, 128}/L

**Mục tiêu:** mục 28 cho ngưỡng ổn định cục bộ 2/λ_max(H_S) ≈ 130/L. Thử các bước sát ngưỡng đó.

- `BUOC_PX_LON = [19, 29, 32, 48, 64, 96, 128]` (ô quét proximal, dùng chung với 6c).
- Chạy riêng bằng `chay_6c_rieng.py` 6 setup, 2026-10-09 09:30–10:33 (63 phút), ghép vào pkl: nay có **88 setup**.
- Bản sao trước khi ghép: `kq_6c_day_du.truoc_64_128.pkl` (thư mục tạm).
- **Chưa chạy lại notebook** (theo yêu cầu): output trong notebook vẫn là của lần chạy 02:13–03:02.

**Kết quả** (giây tới ε; "–" = không đạt trong 600 giây):

| Setup | Vòng | Tổng giây | s/vòng | F − F* min | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|---|---|---|---|
| (best) ISTA 19/L | 1923 | 64.4 | 0.034 | 1.0e-8 | **4.6** | **11.0** | **15.7** | **20.9** | **28.7** |
| ISTA 64/L | 1116 | 55.6 | 0.050 | 9.8e-9 | 32.8 | 38.8 | 43.1 | 47.2 | 51.4 |
| ISTA 96/L | 1476 | 76.0 | 0.052 | 9.7e-9 | 61.1 | 65.1 | 68.0 | 70.7 | 73.3 |
| ISTA 128/L | 1387 | 70.9 | 0.051 | 9.4e-9 | 59.8 | 62.7 | 64.9 | 66.9 | 68.9 |
| (best) FISTA 32/L | 574 | 19.2 | 0.033 | 8.6e-9 | 3.3 | 3.4 | 5.8 | **9.4** | **10.7** |
| FISTA 64/L | 15173 | 600 (hết trần) | 0.040 | **5.1e-3** | – | – | – | – | – |
| FISTA 96/L | 11696 | 600 (hết trần) | 0.051 | **4.9e-2** | – | – | – | – | – |
| FISTA 128/L | 9883 | 600 (hết trần) | 0.061 | **7.3e-2** | – | – | – | – | – |

**Đọc kết quả:**
- **Best không đổi:** ISTA 19/L, FISTA 32/L ở ε = 1e-6 (FISTA 6/L ở ε ≥ 1e-5). Các bảng ở mục 27.2 vẫn đúng.
- **ISTA vẫn hội tụ tới 128/L**, khớp với ngưỡng cục bộ khoảng 130/L của H_S.
  - Số vòng tới 1e-8 ít nhất ở 64/L (1116 vòng). Lên 96–128/L thì không giảm thêm.
  - Giai đoạn đầu dài hẳn (tới 1e-3 mất 33–61 giây so với 4.6 giây ở 19/L), nên chậm hơn 19/L ở mọi ε.
  - Sau khi tới 1e-3, bước lớn đi rất nhanh: 128/L từ 1e-3 xuống 1e-7 chỉ mất 9 giây, so với 24 giây ở 19/L. Đúng như lý thuyết cục bộ (mục 28): bước lớn chỉ có lợi ở giai đoạn cuối.
- **FISTA hỏng từ 64/L:** không tới được 1e-3 trong 600 giây (F − F* thấp nhất 5e-3 → 7e-2, càng tệ khi bước càng lớn).
  - Momentum làm FISTA mất ổn định ở bước nhỏ hơn hẳn ngưỡng của ISTA. Ranh giới nằm giữa 48/L (hội tụ, 17.2 giây) và 64/L.
  - Vì vậy ngưỡng 2/λ_max(H_S) chỉ áp dụng được cho ISTA (không momentum). Với FISTA, bước an toàn thực tế nhỏ hơn khoảng 2 lần.
- **s/vòng tăng lên 0.040–0.061** ở các bước lớn (bình thường khoảng 0.034). Nguyên nhân chưa xác định: có thể do nhiễu máy trong 63 phút chạy, hoặc chi phí tính toán thay đổi khi w lớn. Số vòng không bị ảnh hưởng.
- **Lần chạy notebook tới:**
  - ô quét proximal sẽ có thêm 6 đường (ISTA 13 đường, FISTA 16 đường), tốn thêm khoảng 1–2 phút;
  - các ô 6c, 6d, 6e tự cập nhật; best không đổi, nên hình `ss_ista_fista_*` và `lasso_*` không đổi setup.

## 30. Mọi hình vẽ theo đúng best của 6c; chạy lại toàn bộ notebook (88 setup)

**Sửa mã vẽ** (script `sua14.py`, thư mục tạm):

| Chỗ | Trước | Sau |
|---|---|---|
| `ss_gd_nesterov_*` | bước cố định `[0.5, 1, 2, 6]` | {0.5, 1, 2}/L + bước best theo t_ε của GD và của Nesterov, đọc từ 6c. Ô in ra các bước đã vẽ; legend ghi "(best GD & Nesterov)". |
| Ô quét backtracking (`gd_bt_*`) | chỉ 5 giá trị β ở c = 1e-4, không có setup best (β = 0.1, c = 0.01) | quét đúng lưới 6c (11 setup): `C_ARMIJO`, `LUOI_BT` chuyển về ô quét, ô 6c dùng lại. Đã kiểm tra `LUOI_BT` trùng đúng 11 setup backtracking trong pkl. `chay_6c_rieng.py` lấy thêm `C_ARMIJO`, `LUOI_BT`. |
| `ss_newton_*` | không đánh dấu best | legend "Newton t = 1 (best t_ε)" |
| Đã đúng từ trước | | `ss_gd_bt`, `ss_sgd_gd`, `ss_ista_fista` (bước best ISTA/FISTA tự đọc), `tongket_*`, `lasso_*`, bảng log-loss |

**Sự cố:** `sua14.py` xoá output của các ô nó sửa, trong đó có ô 6c. Phát hiện khi người dùng hỏi 6c có khớp mục 27.2 không. Bản sao trước đó (`bai_lam.truoc_sua14.ipynb`) cho thấy output 6c lúc ấy (82 setup) **khớp đúng** bảng 27.2. Người dùng chọn chạy lại đầy đủ.

| Thời điểm | Sự kiện |
|---|---|
| 2026-10-09 ~19:40 | Chạy thử mẫu 20 000 dòng: không lỗi. 6c đọc 88 setup; các hình in đúng bước best (GD/Nesterov 6/L; ISTA 19/L, FISTA 32/L). |
| 2026-10-09 19:50–20:42 | Chạy lại toàn bộ notebook (52 phút). Chỉ ô LightGBM lỗi như cũ. Các ô code không có output đều là ô chỉ định nghĩa hàm. |

**Kết quả:**
- 6c đọc `kq_6c_day_du.pkl` (lưu lúc 10:33, **88 setup**). Bảng "BEST theo từng ε" **trùng từng số với bảng 27.2**. 6 setup 64–128/L (mục 29) không đổi best nào.
- Ô quét backtracking (500 vòng, TOL): β = 0.1 với c = 1e-4 và c = 0.01 cho **đường giống hệt nhau** (cùng 164 vòng, cùng F cuối 0.043687423), nên trên hình `gd_bt_*` hai đường chồng lên nhau.
- Hình `lasso_thoigian`: điểm sklearn (55 giây) nay nằm trọn trong khung (trục log mở tới 1.5 × thời gian sklearn).
- Mục 7 không đổi: Newton t = 1, MAP@7 0.797596 / 0.022517.
- **Vẫn chờ người dùng quyết** (mục 27.3): giữ t_ε = "lần đầu chạm ε", hay đổi sang "từ đó luôn ≤ ε". Nếu đổi thì FISTA ở 1e-6 thành 6/L và SGD đổi setup.

## 31. Hình quét ISTA/FISTA: chỉ vẽ 7 bước, bảng vẫn in đủ

**Vấn đề:** hình `fista_*` (16 đường) và `proximal_*` (ISTA, 13 đường) quá rối.

**Đã sửa (ô quét proximal):** ô vẫn chạy và in bảng **đủ mọi bước**; hình chỉ vẽ 7 bước (`VE_ISTA`, `VE_FISTA`), mỗi bước một ý. Ghi chú dưới hình thêm dòng "Hình vẽ 7/13 (7/16) bước; đủ các bước ở bảng trên".

| Bước | ISTA | FISTA | Vai trò |
|---|---|---|---|
| 0.1/L | ✓ | ✓ | bước quá nhỏ, rất chậm |
| 1/L | ✓ | ✓ | bước chuẩn của lý thuyết |
| 2/L | ✓ | ✓ | ngưỡng an toàn của lý thuyết |
| 6/L | ✓ | ✓ | FISTA: best ở ε ≥ 1e-5; ISTA: so cặp |
| 19/L | ✓ (best) | | best ISTA theo t_ε |
| 32/L | | ✓ (best) | best FISTA ở ε = 1e-6 |
| 48/L | | ✓ | FISTA: lớn nhất còn hội tụ |
| 64/L | ✓ | ✓ | ISTA: bước lớn, giai đoạn đầu rất chậm; FISTA: bắt đầu hỏng (96, 128/L cùng ý) |
| 128/L | ✓ | | ISTA: lớn nhất, sát ngưỡng cục bộ khoảng 130/L |

Bỏ khỏi hình: 0.5, 1.5, 2.5, 3, 4/L (nằm giữa 1/L và 6/L, không thêm ý); 19, 29/L của FISTA (gần trùng 32/L); 29, 32, 48, 96/L của ISTA; 96, 128/L của FISTA. Danh sách bước ghi cứng trong ô kèm chú thích nguồn (6c), vì ô quét chạy trước 6c.

**Chạy:** chỉ chạy lại ô này, không chạy lại notebook (theo yêu cầu).
- Script `chay_o_quet.py` (thư mục tạm) mở một kernel mới, nạp dữ liệu, các hàm, hằng số lưới và ô quét subgradient (ô proximal cần `duong_sg` để tính F* Lasso), chạy ô proximal, rồi chép output vào đúng ô trong `bai_lam.ipynb`.
- Lần đầu lỗi cú pháp: `\n` trong chuỗi ghi chú bị ghi thành xuống dòng thật khi sửa ô. Đã sửa và kiểm tra biên dịch.
- Lần hai: 2026-10-09 21:02–21:18 (16 phút).

**Kết quả:** bảng quét trùng với lần chạy trước (thuật toán tất định). F* Lasso trong ô = 0.046538164358, như cũ. Bốn hình `proximal_vonglap`, `proximal_thoigian`, `fista_vonglap`, `fista_thoigian` đã được ghi lại. Các ô khác không đổi.

## 32. GD cố định và Nesterov chạy tới cận ổn định; best GD đổi sang 30/L; cập nhật mọi hình

**Cận ước tính** (H Ridge tại w*, λ = 1e-3, λ_max = 0.392; lý thuyết: GD hội tụ khi t < 2/λ_max, Nesterov/FISTA cần t ≤ 1/λ_max):

| λ | λ_max(H) | Cận GD 2/λ_max | Cận Nesterov 1/λ_max |
|---|---|---|---|
| 1e-2 | 0.609 | 20.6/L | 10.3/L |
| **1e-3 (bài chính)** | 0.392 | **31.9/L** | **16.0/L** |
| 1e-4 | 0.299 | 41.8/L | 20.9/L |

Kiểm chứng chéo trên Lasso: 1/λ_max(H_S) ≈ 65/L, và FISTA hỏng đúng ở 64/L (mục 29).

**Lưới thêm** (ô quét và 6c dùng chung):
- `BUOC_GD_LON = [12, 24, 30, 34]` (ô quét GD);
- `BUOC_NES_LON = [8, 12, 14, 16, 18, 24]` (ô quét Nesterov);
- FISTA tách khỏi lưới Nesterov: `BUOC_FISTA = sorted(set(BUOC + BUOC_THEM)) + BUOC_PX_LON`, không đổi tập setup;
- `chay_6c_rieng.py` lấy thêm các hằng số này.

| Thời điểm | Sự kiện |
|---|---|
| 2026-10-09 21:28–22:05 | `chay_6c_rieng.py` 10 setup (37 phút). pkl nay có **98 setup**. Bản sao trước khi ghép: `kq_6c_day_du.truoc_nes_gd_lon.pkl` (thư mục tạm). |
| 2026-10-09 ~22:07 | Sửa mã vẽ (`sua15.py`). Chạy thử mẫu 20 000 dòng: không lỗi. |
| 2026-10-09 22:08–23:02 | Chạy lại toàn bộ notebook (54 phút). Chỉ ô LightGBM lỗi như cũ. Mục 7 không đổi (Newton t = 1, MAP@7 0.797596 / 0.022517). |

### 32.1 Kết quả các setup mới (giây tới ε; "–" = không đạt trong 600 giây)

| Setup | Vòng | Tổng giây | F − F* min | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|---|---|---|
| GD 6/L (best cũ) | 3609 | 120.0 | 1.0e-8 | 12.3 | 28.6 | 49.0 | 71.2 | 94.6 |
| GD 12/L | 1775 | 57.9 | 1.0e-8 | 5.1 | 13.0 | 23.1 | 33.9 | 45.4 |
| GD 24/L | 1189 | 38.7 | 1.0e-8 | 7.4 | 15.5 | 22.0 | 27.7 | 33.2 |
| **GD 30/L** | 870 | 28.3 | 9.9e-9 | **3.2** | **9.7** | **15.2** | **19.6** | **23.9** |
| GD 34/L | 18512 | 600 (hết trần) | **1.7e-4** | 4.4 | – | – | – | – |
| Nesterov 6/L | 750 | 25.0 | 9.8e-9 | 1.3 | 2.5 | 4.6 | **7.7** | **12.5** |
| Nesterov 8/L | 687 | 22.3 | 1.0e-8 | **1.1** | 2.4 | **4.3** | 8.8 | 15.2 |
| Nesterov 12/L | 946 | 30.6 | 9.5e-9 | 2.4 | 3.0 | 6.3 | 9.8 | 16.8 |
| Nesterov 14/L | 934 | 30.5 | 9.8e-9 | 2.7 | 3.1 | 6.1 | 9.5 | 17.6 |
| Nesterov 16/L | 923 | 30.1 | 9.5e-9 | 2.7 | 4.2 | 7.3 | 10.4 | 18.0 |
| Nesterov 18/L | 778 | 25.3 | 9.9e-9 | 2.4 | 2.7 | 5.4 | 8.3 | 15.4 |
| Nesterov 24/L | 18564 | 600 (hết trần) | **1.3e-5** | 2.0 | **2.3** | – | – | – |

- **GD cố định:** best đổi **6/L → 30/L** ở mọi ε (1e-6: 71.2 → 19.6 giây, nhanh hơn 3.6 lần). 34/L kẹt ở F − F* khoảng 1.7e-4, dao động không hội tụ. **Ranh giới nằm giữa 30 và 34/L, khớp cận 2/λ_max ≈ 32/L.** GD 30/L nay chỉ chậm hơn backtracking 2.8 giây ở 1e-6 (19.6 so với 16.8).
- **Nesterov:** best ở **ε = 1e-6 không đổi (6/L)**. 8/L nhanh hơn ở 1e-3 và 1e-5, chỉ chênh 0.2–0.3 giây, trong nhiễu đo. Ở 1e-4, best ghi 24/L (2.3 giây) nhưng setup này **không hội tụ** (kẹt ở 1.3e-5): nó chỉ chạm 1e-4 trước khi dao động (xem lưu ý t_ε ở mục 27.3). 18/L vẫn hội tụ, 24/L thì không. **Ranh giới nằm giữa 18 và 24/L, hơi trên cận 1/λ_max ≈ 16/L.**
- **Bước gần cận** làm F vọt lên vài vòng đầu (ở w = 0 độ cong bằng L). Đây là lý do best của Nesterov vẫn là 6–8/L chứ không sát cận.

### 32.2 Bảng kết quả cập nhật (thay bảng 27.2; chỉ đổi hai dòng GD cố định và Nesterov)

**Best theo ε** (giây / vòng [setup]):

| Thuật toán | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|
| Newton [t=1] | 1.6 / 5 | 1.9 / 6 | 2.2 / 7 | **2.2 / 7** | **2.5 / 8** |
| **Nesterov** | **1.1 / 35 [8/L]** | 2.3 / 70 [24/L]* | 4.3 / 134 [8/L] | 7.7 / 230 [6/L] | 12.5 / 376 [6/L] |
| GD backtracking [β=0.1, c=0.01] | 4.1 / 72 | 8.1 / 144 | 12.2 / 217 | 16.8 / 300 | 21.4 / 383 |
| **GD cố định [t=30/L]** | 3.2 / 98 | 9.7 / 296 | 15.2 / 465 | 19.6 / 600 | 23.9 / 733 |
| SGD | 1.6 / 2 [B=1024 1/√k α0=1] | 10.5 / 13 [như trên] | 51.3 / 55 [B=4096 1/√k α0=1] | 111.5 / 120 [như trên] | 228.9 / 248 [như trên] |
| FISTA (Lasso) | 1.6 / 46 [6/L] | 3.0 / 85 [6/L] | 5.1 / 147 [6/L] | 9.4 / 282 [32/L] | 10.7 / 320 [32/L] |
| ISTA (Lasso) [t=19/L] | 4.6 / 138 | 11.0 / 330 | 15.7 / 471 | 20.9 / 626 | 28.7 / 859 |
| Subgradient (Lasso) | 120.8 / 3730 [α0=8] | 219.6 / 6796 [α0=30] | 379.3 / 11734 [α0=30] | – | – |

\* Nesterov 24/L không hội tụ (kẹt ở 1.3e-5), chỉ chạm 1e-4 lúc đầu.

**Tổng kết Ridge ở ε = 1e-6:** chỉ dòng GD cố định đổi, thành **t = 30/L: 600 vòng, 19.56 giây**, F − F* min 9.9e-9 (870 vòng / 28.3 giây tới F* + 1e-8). Các dòng khác như mục 24.2. sklearn lbfgs đo lại: 1.25 giây.

**Log-loss:** chỉ dòng GD cố định đổi, thành t = 30/L, F − F* 9.9e-7, train 0.040564, val 0.036146, 154/160 hệ số khác 0, chạy lại 19.4 giây. Các dòng khác như mục 24.3 và 27.2.

### 32.3 Hình đã cập nhật

| Hình | Thay đổi |
|---|---|
| `gd_codinh_*` (quét) | chỉ vẽ 7/13 bước: 0.1, 1, 2, 6, 12, **30 (best)**, 34/L (vượt cận, kẹt ở khoảng 3e-4); bảng in đủ |
| `nesterov_*` (quét) | chỉ vẽ 7/15 bước: 0.1, 1, 2, **6 (best)**, 8, 18 (lớn nhất còn hội tụ), 24/L (dao động mãi); bảng in đủ |
| `ss_gd_nesterov_*` | cặp cùng bước 0.5, 1, 2, 6/L ("6/L (best Nesterov)") + **GD 30/L vẽ riêng** ("best GD; không có cặp"), vì Nesterov đã không hội tụ từ 24/L nên không có 30/L |
| `ss_gd_bt_*`, `ss_sgd_gd_*`, `tongket_*`, bảng log-loss | tự lấy GD 30/L |
| Markdown | 6c (danh sách bước chạy bổ sung), 6e (setting: GD 30/L, Nesterov 6/L kèm ghi chú 8/L, nguồn gốc số liệu), đoạn GD vs backtracking (hội tụ tới 30/L, 34/L thì không), đoạn GD vs Nesterov |

## 33. `ket_luan.md`: thêm cột "luôn ≤ ε từ đó" vào bảng chi tiết

- Mục 4 của `ket_luan.md` có thêm cột **Luôn ≤ ε từ**: thời điểm mà từ đó trở đi F − F* luôn ≤ 10⁻⁶ (subgradient: 10⁻⁵). Best được đánh dấu theo hai cách: **★** = lần đầu chạm ε (định nghĩa notebook đang dùng), **☆** = luôn ≤ ε.
- Bảng sinh thẳng từ `kq_6c_day_du.pkl` (98 setup) bằng script `bang_chi_tiet.py` (thư mục tạm). Notebook không đổi.
- **Hai cách chọn khác nhau ở 3 thuật toán** (ε = 10⁻⁶):

  | Thuật toán | ★ lần đầu chạm | ☆ luôn ≤ ε từ đó |
  |---|---|---|
  | Nesterov | 6/L, 7.7 giây | **8/L, 8.8 giây** (6/L: 9.5 giây) |
  | SGD | B = 4096, 1/√k, α₀ = 1, 111.5 giây | B = 1024, 1/√k, α₀ = 0.2, 526.0 giây |
  | FISTA | 32/L, 9.4 giây | 6/L, 11.9 giây (32/L: 16.4 giây) |

- GD cố định, backtracking, Newton, ISTA và subgradient trùng nhau theo cả hai cách.
- **Sửa lại mục 27.3:** khi đó bảng ghi Nesterov "luôn ≤ ε" vẫn là 6/L, vì lúc đó chưa có setup 8/L. Với lưới hiện tại thì là 8/L.
