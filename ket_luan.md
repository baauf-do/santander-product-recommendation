# Kết luận — Tối ưu hoá log-loss (Santander, sản phẩm `ind_recibo_ult1`)

Số liệu lấy từ lần chạy notebook cuối, ngày 09/10/2026 lúc 22:08–23:02. Ô 6c đọc `kq_6c_day_du.pkl` (98 setup). Chi tiết quá trình ở `tien_trinh.md`, đặc biệt các mục 23–32.

**Bài toán**
- Dữ liệu: 466 243 mẫu train × 161 cột (160 đặc trưng + intercept). Tỉ lệ dương 1,22%. Validation 75 839 mẫu.
- Hàm mục tiêu: Ridge F = log-loss + (λ/2)‖w‖², Lasso F = log-loss + λ₁‖w‖₁, với λ = λ₁ = 10⁻³. L = 6,25.
- F*: Ridge = 0,043630121211 (Newton = sklearn newton-cholesky). Lasso = 0,046538162369 (FISTA chạy 600 giây).

**Cách chọn best**
- t_ε = thời gian **lần đầu** F − F* ≤ ε. ε chính = 10⁻⁶; best ở 10⁻⁵ và 10⁻⁶ chỉ khác nhau ở Nesterov và FISTA.
- Mỗi setup chạy tới F* + 10⁻⁸ hoặc hết trần 600 giây.
- Ký hiệu **★** = nhanh nhất trong cùng bài toán.

---

## 0. Các hằng số trong bài

### 0.1 Dữ liệu

| Hằng số | Giá trị | Ghi chú |
|---|---|---|
| Sản phẩm dùng để so thuật toán | `ind_recibo_ult1` | tỉ lệ mua cao nhất trong 24 sản phẩm |
| Các tháng dùng | 2015-01 → 2016-01 (13 tháng) | dùng cả 17 tháng thì hết RAM |
| Số tháng lịch sử cho mỗi sản phẩm | 5 | |
| Train / validation | 2015-06 → 2015-12 / 2016-01 | tháng đích phải có đủ 5 tháng trước |
| Số mẫu train | 466 243 | |
| Số mẫu validation | 75 839 | |
| Số chiều d | 161 = 160 đặc trưng + 1 intercept | đặc trưng chuẩn hoá theo trung bình và độ lệch chuẩn của train |
| Tỉ lệ dương | 1,22% (train), 1,07% (validation) | |
| Điểm khởi đầu w₀ | 0 | dùng cho mọi thuật toán; F(w₀) = log 2 |

### 0.2 Hàm mục tiêu và hằng số giải tích

| Hằng số | Giá trị | Ghi chú |
|---|---|---|
| λ (Ridge) | 10⁻³ | không phạt intercept |
| λ₁ (Lasso) | 10⁻³ | không phạt intercept |
| L | 6,2508 = λ_max(XᵀX/n)/4 | chặn trên độ cong của log-loss (σ′ ≤ 1/4); bước "t = k/L" nghĩa là t = k/6,2508 |
| 1/L | 0,1600 | bước chuẩn của lý thuyết |
| F* Ridge | 0,043630121211 | Newton (tol 10⁻¹⁰), trùng sklearn newton-cholesky (tol 10⁻¹²) |
| F* Lasso | 0,046538162369 | min F của FISTA bước 1/L chạy 600 giây (sai số khoảng 10⁻⁹) |
| C của sklearn | 1/(nλ) ≈ 2,145·10⁻³ | để sklearn tối thiểu hoá đúng F (sklearn dùng tổng loss, bài này dùng trung bình) |

### 0.3 Độ cong tại nghiệm (dùng để ước cận bước)

| Ma trận | λ_max | λ_min | κ | 2/λ_max | 1/λ_max |
|---|---|---|---|---|---|
| Hessian Ridge tại w*, λ = 10⁻³ | 0,392 | 1,0·10⁻³ | 392 | 31,9/L | 16,0/L |
| Hessian log-loss của Lasso trên tập hỗ trợ (20 toạ độ gồm intercept) | 0,0964 | 2,9·10⁻⁴ | 337 | 129,7/L | 64,9/L |
| Hessian Ridge tại w*, λ = 10⁻² | 0,609 | 6,3·10⁻³ | 96 | 20,6/L | 10,3/L |
| Hessian Ridge tại w*, λ = 10⁻⁴ | 0,299 | 1,0·10⁻⁴ | 2 986 | 41,8/L | 20,9/L |

### 0.4 Điều kiện dừng và ngân sách của các ô quét tham số

| Hằng số | Giá trị | Ghi chú |
|---|---|---|
| Ngưỡng dừng tương đối | 10⁻³ | dừng khi ‖r(w_k)‖ ≤ 10⁻³·‖r(w₀)‖; r là gradient (Ridge), subgradient chuẩn nhỏ nhất, hoặc gradient mapping (ISTA/FISTA) |
| Số vòng tối đa | 500 | GD, backtracking, Nesterov, subgradient, ISTA, FISTA |
| Số vòng tối đa của Newton | 30 | |
| Số epoch tối đa của SGD | 42 | |
| Số lần ghi loss mini-batch mỗi epoch | 20 | chỉ để vẽ hình SGD |

### 0.5 Phép đo chính (thời gian tới F − F* ≤ ε)

| Hằng số | Giá trị | Ghi chú |
|---|---|---|
| Trần thời gian mỗi setup | 600 giây | |
| Dừng sớm khi | F ≤ F* + 10⁻⁸ | đã đủ chính xác |
| Ngưỡng gradient | 10⁻¹² | rất chặt, để chỉ dừng theo hai điều kiện trên |
| Số vòng tối đa | không giới hạn | chỉ giới hạn thời gian |
| Các mức ε báo cáo | 10⁻³, 10⁻⁴, 10⁻⁵, 10⁻⁶, 10⁻⁷ | |
| ε chính để chọn best | 10⁻⁶ | |
| Số setup | 98 | xem mục 4 |
| Trục hoành hình cặp đôi | cắt ở 500 vòng | chỉ cắt trục, không cắt dữ liệu |

### 0.6 Lưới tham số

Bước của GD, Nesterov, ISTA và FISTA ghi theo bội của 1/L.

| Thuật toán | Lưới | Số setup |
|---|---|---|
| GD cố định | 0,1; 0,5; 1; 1,5; 2; 2,5; 3; 4; 6; 12; 24; 30; 34 | 13 |
| GD backtracking | β ∈ {0,1; 0,3; 0,5; 0,7; 0,9} với c = 10⁻⁴, cộng β ∈ {0,1; 0,5} × c ∈ {10⁻²; 0,1; 0,3}. Bước khởi đầu 1; mỗi vòng thử gấp đôi bước trước rồi nhân β tới khi thoả Armijo | 11 |
| Nesterov | 0,1; 0,5; 1; 1,5; 2; 2,5; 3; 4; 6; 8; 12; 14; 16; 18; 24. Bản cho hàm lồi tổng quát (hệ số momentum không dùng μ), không restart | 15 |
| Newton | bước cố định t ∈ {0,1; 0,25; 0,5; 1}; backtracking β ∈ {0,1; 0,5; 0,9}, c = 10⁻⁴ | 7 |
| SGD | batch {1024; 4096; 16384} × lịch {cố định; α₀/√k} × α₀ {0,2; 1; 5}; seed 0; F đo trên toàn bộ dữ liệu cuối mỗi epoch | 18 |
| SGD (riêng ô quét) | α₀ ∈ {0,01; 0,05; 0,2; 1; 5}, batch 4096; lịch {cố định; α₀/√k; α₀/k} | – |
| Subgradient | α₀ ∈ {0,5; 2; 8; 30; 100}, lịch α₀/√k | 5 |
| ISTA | 0,1; 0,5; 1; 1,5; 2; 6; 19; 29; 32; 48; 64; 96; 128 | 13 |
| FISTA | 0,1; 0,5; 1; 1,5; 2; 2,5; 3; 4; 6; 19; 29; 32; 48; 64; 96; 128 | 16 |

**Bước được vẽ trên hình quét** (bảng trong notebook in đủ mọi bước):

| Hình | Bước vẽ |
|---|---|
| GD cố định (`gd_codinh_*`) | 0,1; 1; 2; 6; 12; 30; 34 |
| Nesterov (`nesterov_*`) | 0,1; 1; 2; 6; 8; 18; 24 |
| ISTA (`proximal_*`) | 0,1; 1; 2; 6; 19; 64; 128 |
| FISTA (`fista_*`) | 0,1; 1; 2; 6; 32; 48; 64 |

### 0.7 Thí nghiệm λ và Phần II

| Hằng số | Giá trị | Ghi chú |
|---|---|---|
| Các λ thử | 10⁻², 10⁻³, 10⁻⁴ | |
| Mức cần đạt | (F − F*)/(F(w₀) − F*) ≤ 10⁻⁶ | |
| Trần mỗi lần chạy | 120 giây | |
| Bước GD, Nesterov | 1/L, với L = 6,2508 + λ | |
| MAP@7 | 24 sản phẩm, k = 7 | Newton backtracking, tối đa 30 vòng; sản phẩm có ít hơn 10 mẫu dương thì lấy tỉ lệ nền |

---

## 1. Kết luận cuối cùng của các thuật toán

Các bảng dưới chỉ ghi setup best. Kết quả **mọi setup** của từng thuật toán (98 setup, kèm thời gian và số vòng tới từng ε) nằm ở **mục 4**.

### 1.1 Ridge — best của từng thuật toán ở ε = 10⁻⁶

| Thuật toán | Setup best | Vòng tới 10⁻⁶ | Giây tới 10⁻⁶ | Tới F* + 10⁻⁸ (vòng / giây) | Giây / vòng |
|---|---|---|---|---|---|
| **Newton ★** | **t = 1** (backtracking cho đường y hệt) | **7** | **2,2** | 8 / 2,5 | 0,33 (≈ 9,8 vòng GD) |
| Nesterov | t = 6/L | 230 | 7,7 | 750 / 25,0 | 0,033 |
| GD backtracking | β = 0,1, c = 10⁻² | 300 | 16,8 | 478 / 26,8 | 0,068 (≈ 2,1 vòng GD) |
| GD cố định | t = 30/L | 600 | 19,6 | 870 / 28,3 | 0,033 |
| SGD | batch 4096, α_k = α₀/√k, α₀ = 1 | 120 epoch | 111,5 | không tới (min 3,9·10⁻⁸ sau 600 s) | 0,82 / epoch (≈ 24,5 vòng GD) |
| *sklearn lbfgs (mốc)* | mặc định, C = 1/(nλ) | 27 vòng | 1,25 (cả lần fit) | dừng ở F − F* = 9,1·10⁻⁶ | – |

- Newton nhanh nhất ở mọi ε ≤ 10⁻⁴. Ở 10⁻³ thì Nesterov 8/L nhanh hơn (1,1 giây so với 1,6 giây), nhưng mức chênh nằm trong nhiễu đo.
- sklearn lbfgs dừng ở 9·10⁻⁶ theo tol riêng của nó, nên chưa tới 10⁻⁶.

### 1.2 Lasso — best của từng thuật toán ở ε = 10⁻⁶

| Thuật toán | Setup best | Vòng tới ε | Giây tới ε | Tới F* + 10⁻⁸ (vòng / giây) | Hệ số ≠ 0 |
|---|---|---|---|---|---|
| **FISTA ★** | **t = 32/L** (ở ε ≥ 10⁻⁵: 6/L) | **282** | **9,4** | 574 / 19,2 | 20/160 |
| ISTA | t = 19/L | 626 | 20,9 | 1923 / 64,4 | 19/160 |
| Subgradient | α₀ = 30, α_k = α₀/√k | 11 734 (**ε = 10⁻⁵**) | 379,3 (ε = 10⁻⁵) | không tới 10⁻⁶ (min 6,3·10⁻⁶) | 122/160 |
| *sklearn saga (mốc)* | L1, C = 1/(nλ₁) | 100 (chạm `max_iter`) | 54,4 (cả lần fit) | dừng ở 8,4·10⁻⁷ | 19/160 |

### 1.3 Best theo từng ε (giây / vòng [setup]; SGD: vòng = epoch)

| Thuật toán | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 10⁻⁶ | 10⁻⁷ |
|---|---|---|---|---|---|
| Newton [t=1] | 1,6 / 5 | **1,9 / 6 ★** | **2,2 / 7 ★** | **2,2 / 7 ★** | **2,5 / 8 ★** |
| Nesterov | **1,1 / 35 [8/L] ★** | 2,3 / 70 [24/L]¹ | 4,3 / 134 [8/L] | 7,7 / 230 [6/L] | 12,5 / 376 [6/L] |
| GD backtracking [β=0,1, c=0,01] | 4,1 / 72 | 8,1 / 144 | 12,2 / 217 | 16,8 / 300 | 21,4 / 383 |
| GD cố định [t=30/L] | 3,2 / 98 | 9,7 / 296 | 15,2 / 465 | 19,6 / 600 | 23,9 / 733 |
| SGD [1/√k, α₀=1] | 1,6 / 2 [B=1024] | 10,5 / 13 [B=1024] | 51,3 / 55 [B=4096] | 111,5 / 120 [B=4096] | 228,9 / 248 [B=4096] |
| **FISTA (Lasso)** | **1,6 / 46 [6/L] ★** | **3,0 / 85 [6/L] ★** | **5,1 / 147 [6/L] ★** | **9,4 / 282 [32/L] ★** | **10,7 / 320 [32/L] ★** |
| ISTA (Lasso) [t=19/L] | 4,6 / 138 | 11,0 / 330 | 15,7 / 471 | 20,9 / 626 | 28,7 / 859 |
| Subgradient (Lasso) | 120,8 / 3730 [α₀=8] | 219,6 / 6796 [α₀=30] | 379,3 / 11 734 [α₀=30] | – | – |

¹ Nesterov 24/L **không hội tụ** (kẹt ở F − F* ≈ 1,3·10⁻⁵). Nó chỉ chạm 10⁻⁴ một lần lúc đầu, nên ô này là hệ quả của định nghĩa "lần đầu chạm ε" (xem mục 1.8).

### 1.4 Bước tốt nhất và cận ổn định

Cận ước tính từ λ_max của Hessian tại nghiệm:
- GD/ISTA (không momentum) hội tụ khi t < 2/λ_max.
- Nesterov/FISTA (có momentum) cần t ≤ 1/λ_max.

| Thuật toán | Hessian dùng để ước cận | Cận ước tính | Best | Lớn nhất còn hội tụ | Bước đầu tiên hỏng |
|---|---|---|---|---|---|
| GD cố định | H Ridge tại w*, λ_max = 0,392 | 2/λ_max ≈ **32/L** | 30/L | 30/L | **34/L** (kẹt ở 1,7·10⁻⁴) |
| Nesterov | như trên | 1/λ_max ≈ **16/L** | 6/L (8/L ở 10⁻³, 10⁻⁵) | 18/L | **24/L** (kẹt ở 1,3·10⁻⁵) |
| ISTA | H_S = Hessian log-loss trên tập hỗ trợ S (20 toạ độ) tại w*_Lasso, λ_max = 0,096 | 2/λ_max ≈ **130/L** | 19/L | 128/L (đã thử tới đây) | – |
| FISTA | như trên | 1/λ_max ≈ **65/L** | 6/L (ε ≥ 10⁻⁵), 32/L (10⁻⁶) | 48/L | **64/L** (không tới nổi 10⁻³) |
| GD backtracking | – | – | tự chọn bước, trung vị ≈ 25/L | – | – |
| Newton | – | – | t = 1 (Armijo nhận ngay từ vòng đầu) | – | – |

- **Cận lý thuyết 2/L quá bảo thủ.** L tính từ chặn σ′ ≤ 1/4, nhưng dữ liệu chỉ có 1,2% mẫu dương nên tại nghiệm σ(1−σ) ≈ 0,01. Độ cong thật vì vậy nhỏ hơn L khoảng 16 lần (Ridge) đến 65 lần (Lasso, trên S).
- **Cận ước tính khớp với chỗ hỏng thực tế:**
  - GD hỏng giữa 30 và 34/L, cận khoảng 32/L.
  - FISTA hỏng giữa 48 và 64/L, cận khoảng 65/L.
  - Nesterov hỏng giữa 18 và 24/L, hơi trên cận khoảng 16/L.
- **Best không nằm sát cận**, trừ GD. Tại w = 0 độ cong bằng đúng L, nên bước lớn làm F vọt lên (F − F* tới khoảng 10) ở vài vòng đầu rồi mới giảm. Bước sát cận chỉ có lợi ở giai đoạn cuối.

### 1.5 Log-loss không cộng phạt (chạy lại đúng setup best tới ε dùng để chọn)

| Phương pháp | Bài | Setup | F − F* | Log-loss train | Log-loss val | Hệ số ≠ 0 |
|---|---|---|---|---|---|---|
| GD cố định | Ridge | t = 30/L | 9,9·10⁻⁷ | 0,040564 | **0,036146** | 154/160 |
| GD backtracking | Ridge | β = 0,1, c = 10⁻² | 9,3·10⁻⁷ | 0,040621 | 0,036204 | 154/160 |
| Nesterov | Ridge | t = 6/L | 9,6·10⁻⁷ | 0,040613 | 0,036194 | 154/160 |
| Newton | Ridge | t = 1 | 1,9·10⁻⁷ | 0,040591 | 0,036173 | 154/160 |
| SGD | Ridge | B = 4096, 1/√k, α₀ = 1 | 9,6·10⁻⁷ | 0,040607 | 0,036196 | 154/160 |
| sklearn lbfgs | Ridge | mặc định | 9,1·10⁻⁶ | 0,040654 | 0,036241 | 154/160 |
| Subgradient | Lasso | α₀ = 30 (ε = 10⁻⁵) | 1,0·10⁻⁵ | 0,040886 | 0,036191 | 122/160 |
| ISTA | Lasso | t = 19/L | 9,9·10⁻⁷ | 0,040899 | 0,036204 | 19/160 |
| FISTA | Lasso | t = 32/L | 9,5·10⁻⁷ | 0,040909 | 0,036209 | 20/160 |
| sklearn saga | Lasso | L1 | 8,4·10⁻⁷ | 0,040952 | 0,036256 | 19/160 |

- Khi đã cùng tới F − F* ≈ 10⁻⁶, mọi phương pháp **chỉ khác nhau ở chữ số thứ 4–5**. Khác biệt giữa các thuật toán nằm ở **thời gian**, không nằm ở chất lượng mô hình.
- Lasso cho mô hình thưa (19–20 trên 160 hệ số khác 0) mà log-loss val gần như bằng Ridge.
- **Subgradient không cho hệ số bằng 0 thật** (122 trên 160 hệ số khác 0). Muốn có độ thưa thì cần bước prox.

### 1.6 Thí nghiệm λ: số vòng theo số điều kiện κ

Số vòng tới mức tương đối 10⁻⁶. GD và Nesterov dùng bước 1/L; trần 120 giây.

| λ | L/λ (cận κ) | κ thật = λ_max/λ_min | GD 1/L | Nesterov | Nesterov / √κ | Newton |
|---|---|---|---|---|---|---|
| 10⁻² | 626 | 96 | > 3795 | 384 | 39,2 | 6 |
| 10⁻³ | 6 252 | 392 | > 3732 | 896 | 45,3 | 7 |
| 10⁻⁴ | 62 509 | 2 986 | > 3697 | 1247 | 22,8 | 8 |

- **κ thật nhỏ hơn cận L/λ từ 7 đến 21 lần**, cùng lý do như mục 1.4.
- **Newton gần như không phụ thuộc κ** (6 → 8 vòng), đúng lý thuyết.
- **Nesterov / √κ không hằng số** (39 → 45 → 23), nên chưa kết luận được số vòng tỉ lệ với √κ. √κ chỉ là chặn trường hợp xấu nhất.
- **GD bước 1/L** không tới được mức này trong 120 giây ở cả ba λ, vì 1/L chỉ bằng 1/20 đến 1/40 cận thật. Cận cho từng λ: 20,6/L, 31,9/L và 41,8/L.

### 1.7 Phần II — toàn bộ 24 sản phẩm

- Dùng thuật toán nhanh nhất của 6c (**Newton t = 1**) để train 24 mô hình, mất 42 giây.
- **MAP@7 = 0,797596** trên khách có thêm sản phẩm mới (2,82% số khách), và **0,022517** trên toàn bộ khách (cách Kaggle chấm).
- Ô so sánh với LightGBM (mục 8) **chưa chạy được**: lỗi `OSError: access violation` khi nạp thư viện.

### 1.8 Những điểm còn mở

- **Định nghĩa t_ε chưa chốt.** Hiện dùng "lần đầu chạm ε", vốn lạc quan với các đường dao động. Nếu đổi sang "từ đó trở đi luôn ≤ ε":
  - FISTA ở 10⁻⁶ thành **6/L (11,9 giây)** thay cho 32/L;
  - SGD thành B = 1024, α₀ = 0,2 (526 giây);
  - Nesterov ở 10⁻⁶ thành **8/L (8,8 giây)** thay cho 6/L, vì 6/L chạm 10⁻⁶ lúc 7,7 giây nhưng còn vọt lên, tới 9,5 giây mới ở hẳn dưới;
  - Nesterov 24/L không còn được chọn ở 10⁻⁴;
  - GD cố định, backtracking, Newton, ISTA, subgradient giữ nguyên.

  Mục 4 có cột "Luôn ≤ ε từ" và đánh dấu best theo cả hai cách (★ lần đầu chạm, ☆ luôn ≤ ε).
- **Thời gian đo lệch giữa các lần chạy.** Cùng một setup có thể lệch tới khoảng 50% (notebook có lúc chậm 1,5 lần so với script riêng). Các chênh lệch dưới khoảng 0,5 giây ở 10⁻³ không đủ để xếp hạng; khi đó dùng cột số vòng.
- **ε chính trong `CLAUDE.md` vẫn ghi 10⁻⁵**, còn notebook dùng 10⁻⁶ (thống nhất với mục 7). Ở 10⁻⁵ chỉ đổi best của Nesterov (8/L) và FISTA (6/L).
- **`overleaf/bao_cao.tex` và `README.md` còn số liệu rất cũ** (ngân sách 250 vòng; best 4/L, 2/L; FISTA 99/231 vòng). Cần cập nhật theo file này.

---

## 2. Kinh nghiệm rút ra trong quá trình thực hiện

### Về cách so sánh thuật toán

1. **Điều kiện dừng riêng của từng thuật toán không so sánh được với nhau.** Cùng TOL = 10⁻³ trên gradient, gradient mapping hay subgradient ứng với các mức F − F* khác nhau. Chọn "setup đã hội tụ nhanh nhất" vì thế dễ chọn nhầm. Chuyển sang **t_ε với cùng một F\*** mới so được công bằng. Muốn vậy F\* phải rất chính xác (Newton tol 10⁻¹⁰; FISTA chạy 600 giây cho Lasso).
2. **Một lần chạy cho mọi ε.** Lưu đủ lịch sử (F, t) rồi đọc t_ε cho nhiều ε, thay vì chạy lại cho từng ε.
3. **"Lần đầu chạm ε" lạc quan với đường dao động** (FISTA và Nesterov bước lớn, SGD). Nên báo cáo kèm "luôn ≤ ε từ đó", hoặc ít nhất ghi chú lại.
4. **Tính theo số vòng và tính theo thời gian cho thứ hạng khác nhau.**
   - SGD thắng theo lượt qua dữ liệu ở giai đoạn đầu nhưng thua xa theo thời gian: một epoch tốn bằng khoảng 24 vòng GD, chủ yếu do chép mini-batch `X[idx]`.
   - Newton thắng nhờ d = 160 nhỏ: Hessian 160 × 160 rẻ, mỗi vòng chỉ bằng khoảng 10 vòng GD. Với d lớn kết luận sẽ khác.

### Về chọn tham số

5. **Best ở biên lưới thì phải mở rộng lưới.** Chuyện này lặp lại bốn lần (GD 4/L → 6/L → 30/L; Nesterov, ISTA, FISTA dừng ở 6/L). Mỗi lần mở rộng đều đổi kết luận: ISTA nhanh hơn 15 lần, GD nhanh hơn 3,6 lần.
6. **Ngưỡng 2/L của lý thuyết chỉ là trường hợp xấu nhất.** Ước cận từ λ_max của Hessian tại nghiệm dự đoán đúng chỗ thuật toán bắt đầu hỏng. Nhưng phải dùng **đúng Hessian và đúng loại cận**:
   - Lasso dùng **H_S** (phần trơn, trên tập hỗ trợ), không dùng H của Ridge. Từng dùng nhầm và ra 32/L thay vì 130/L.
   - Hessian trên mọi toạ độ của Lasso suy biến khi không có λI.
   - Có momentum thì cận là **1/λ_max**, chặt hơn GD 2 lần.
7. **Bước lớn nhất còn ổn định chưa chắc là bước nhanh nhất.** Giai đoạn đầu (w = 0, độ cong bằng L) phạt bước lớn. Best thường nằm giữa: ISTA 19/L so với cận 130/L, Nesterov 6–8/L so với cận 16/L.
8. **Với mỗi tổ hợp (batch, lịch giảm bước) của SGD đều phải dò lại α₀.** Lịch 1/√k cần α₀ = 1 chứ không phải 0,2 như bước cố định. Dùng chung một α₀ cho mọi tổ hợp sẽ làm lệch so sánh.

### Về đo đạc và quy trình chạy

9. **Đo thời gian rất nhiễu.** Có những đoạn kernel notebook dùng ít luồng BLAS hơn, làm GD và backtracking chậm 1,5 lần; chạy bằng script riêng (`chay_6c_rieng.py`) thì ổn định.
   - Luôn báo cáo kèm **số vòng**.
   - Chạy tuần tự, không chạy song song.
   - Khi nghi ngờ thì đo lại bằng tiến trình riêng.
10. **Lưu kết quả đắt và chạy lại từng phần.** Ô 6c đầy đủ mất khoảng 7 giờ. Lưu vào pkl, cho phép chạy lại riêng một thuật toán hoặc một setup (`"ISTA|t=6/L"`) rồi ghép vào, và luôn giữ bản sao pkl trước khi ghép.
11. **Chạy thử trên mẫu 20 000 dòng trước khi chạy thật** (khoảng 10 phút so với khoảng 55 phút). Cách này bắt được mọi lỗi mã trước khi tốn một lần chạy dài.
12. **Sửa notebook bằng script dễ hỏng.** Hay gặp lỗi thoát ký tự: `\n` trong chuỗi thành xuống dòng thật, `\\beta` mất một dấu gạch. Script sửa ô còn có thể xoá output. Sau mỗi lần sửa nên kiểm tra biên dịch và kiểm tra output.

### Về hình và bảng

13. **Hình và bảng phải lấy cùng một nguồn số liệu.**
    - Khi hình tổng kết và hình so sánh còn lấy từ các ô quét (dừng theo TOL) mà kết luận lại lấy từ 6c, chúng ghi "tốt nhất" khác nhau.
    - Cách sửa: mọi hình so sánh đọc thẳng `KQ_6C` và tự lấy best (`best_6c`); ô quét và 6c dùng chung hằng số lưới.
    - Riêng các ô quét chạy trước 6c, nên danh sách bước để vẽ phải ghi tay và cập nhật khi best đổi.
14. **Ít đường, mỗi đường kể một ý.** Hình 13–16 đường không đọc được. Giữ khoảng 7 bước: quá nhỏ, chuẩn lý thuyết, ngưỡng 2/L, best, lớn nhất còn hội tụ, bắt đầu hỏng. Số liệu đầy đủ để ở bảng ngay trên hình.
15. **Con số chính xác thì dùng bảng, hình dạng đường cong mới dùng hình.** Ví dụ log-loss chỉ khác nhau ở chữ số thứ 4 thì in bảng; Newton chỉ có vài điểm thì bảng rõ hơn hình.
16. **Màu cố định theo thứ tự**, không chọn màu theo tên. Ghi chú dưới hình nêu điều kiện dừng và nguồn dữ liệu. Trục F − F* dùng thang log. Muốn cắt trục thì dùng xlim, không cắt dữ liệu.

---

## 3. Các hình nên đưa vào báo cáo

Mọi file nằm trong `overleaf/hinh/`. Hình nào có cả bản theo vòng (`_vonglap`) và theo thời gian (`_thoigian`) thì chọn bản đúng thông điệp. Các hình dưới đây đã được tạo lại ở lần chạy cuối.

### 3.1 Bắt buộc (thông điệp chính)

| # | Hình | Thông điệp |
|---|---|---|
| 1 | `tongket_thoigian.pdf` | **Kết quả chính Ridge.** 5 thuật toán ở setup best, có đường ngang ε = 10⁻⁶ và điểm sklearn. Newton tới 10⁻⁶ sau 2,2 giây, Nesterov 7,7 giây, backtracking và GD khoảng 17–20 giây, SGD 111 giây. |
| 2 | `lasso_thoigian.pdf` | **Kết quả chính Lasso.** FISTA nhanh hơn ISTA khoảng 2 lần, subgradient không tới được 10⁻⁶; điểm sklearn saga. |
| 3 | `ss_gd_nesterov_vonglap.pdf` | **Tăng tốc Nesterov.** Cùng bước, Nesterov vượt GD 3–4 bậc độ lớn sau 500 vòng; GD 30/L (best GD) vẫn thua Nesterov 6/L. |
| 4 | `ss_gd_bt_thoigian.pdf` | **Chọn bước.** Backtracking (β nhỏ) ngang GD bước tối ưu 30/L mà không cần biết trước cận; GD 1/L chậm hẳn. |
| 5 | `ss_newton_vonglap.pdf` | **Hội tụ bậc hai.** Newton t = 1 tới 10⁻⁹ trong 8 vòng; backtracking trùng hẳn t = 1; bước t < 1 mất tính bậc hai. |
| 6 | `ss_sgd_gd_luot.pdf` + `ss_sgd_gd_thoigian.pdf` | **SGD:** thắng theo lượt qua dữ liệu ở đầu, thua theo thời gian vì 1 epoch ≈ 24 vòng GD. Hai hình nên đặt cạnh nhau. |
| 7 | `ss_ista_fista_vonglap.pdf` | **Proximal và gia tốc** trên Lasso: FISTA hơn ISTA cùng bước; mốc subgradient; FISTA bước lớn dao động mạnh. |
| 8 | `gd_codinh_vonglap.pdf` | **Cận ổn định.** GD hội tụ tới 30/L, còn 34/L (vượt cận khoảng 32/L) kẹt và dao động. Đặt cùng bảng cận ở mục 1.4. |

### 3.2 Nên có (minh hoạ thêm, có thể đưa vào phụ lục)

| Hình | Thông điệp |
|---|---|
| `nesterov_vonglap.pdf` | Nesterov hội tụ tới 18/L; 24/L (trên cận khoảng 16/L) dao động mãi. |
| `fista_vonglap.pdf` | FISTA: 6/L nhanh nhất giai đoạn đầu; 32/L và 48/L dao động; 64/L hỏng (cận khoảng 65/L). |
| `proximal_vonglap.pdf` | ISTA: 19/L tách hẳn; bước 64–128/L có đỉnh vọt lên ở đầu rồi đi chậm. |
| `ss_lambda_vonglap.pdf` | Ảnh hưởng của λ (số điều kiện): Newton gần như không đổi, GD và Nesterov chậm dần khi λ nhỏ. Dùng kèm bảng 1.6. |
| `sgd_eta_epoch.pdf` | SGD bước cố định: α₀ nhỏ (0,01) rất chậm, α₀ lớn (5) dừng ở mức cao hơn do nhiễu; α₀ = 0,2–1 tốt nhất; loss mini-batch nhiễu. |
| `sgd_lich_epoch.pdf` | Lịch giảm bước × α₀: 1/√k với α₀ = 1 tốt nhất. |
| `gd_bt_vonglap.pdf` | Backtracking: β nhỏ tốt hơn; c gần như không ảnh hưởng. |
| `subgrad_vonglap.pdf` | Subgradient chậm, phải dùng bước teo dần. |
| `tongket_vonglap.pdf`, `lasso_vonglap.pdf` | Bản theo số vòng của hai hình chính (Newton 8 vòng so với GD hàng trăm vòng). |

### 3.3 Không dùng

- **Bản `_thoigian` của các hình quét** (`gd_codinh`, `nesterov`, `fista`, `proximal`, `gd_bt`, `sgd_*`, `subgrad`): trùng thông điệp với bản theo vòng. Mọi thuật toán bậc nhất có chi phí mỗi vòng gần bằng nhau (bảng chi phí ở 6c).
- **File cũ không còn được notebook tạo ra**, có thể mang số liệu sai: `best_so_sanh.pdf`, `gd_bt_vonglap_tach.pdf`, `newton_bt_*.pdf`, `newton_codinh_*.pdf`, `sgd_muot_vs_batch.pdf`, `ss_bt_buoc.pdf`, `ss_ridge_lasso_*.pdf`. Nên xoá khỏi `overleaf/hinh/` để khỏi dùng nhầm.

### 3.4 Bảng nên đưa vào báo cáo (thay vì hình)

- **Bảng 1.3:** best theo từng ε, kèm số vòng.
- **Bảng 1.4:** bước tốt nhất và cận ổn định.
- **Bảng chi phí mỗi vòng:** giây/vòng và quy ra số vòng GD, cuối ô 6c. Đây là cầu nối giải thích vì sao thứ hạng theo vòng và theo thời gian khác nhau.
- **Bảng 1.5:** log-loss train/val và độ thưa.
- **Bảng 1.6:** thí nghiệm λ và κ.

---

## 4. Bảng kết quả chi tiết từng thuật toán

Mọi setup của ô 6c (`kq_6c_day_du.pkl`, 98 setup). Mỗi setup chạy tới F* + 10⁻⁸ hoặc hết trần 600 giây.

- Cột ε: **giây (vòng)** lần đầu F − F* ≤ ε; "–" là không đạt trong 600 giây. Với SGD, vòng là epoch.
- Cột **Kết thúc**: "F* + 10⁻⁸" là dừng sớm vì đã đủ chính xác; "hết trần" là chạy đủ 600 giây. Cột **F − F\* min** là mức thấp nhất đạt được.
- Cột **Luôn ≤ ε từ**: **giây (vòng)** mà **từ đó trở đi** F − F* luôn ≤ ε (ε = 10⁻⁶; subgradient: 10⁻⁵). Với đường giảm đều thì trùng cột 10⁻⁶; với đường dao động (FISTA và Nesterov bước lớn, SGD) thì muộn hơn.
- **In đậm** là ô nhanh nhất trong cột của thuật toán đó. Setup best ở ε chính = 10⁻⁶ (nếu không setup nào tới 10⁻⁶ thì lấy ở ε nhỏ nhất đạt được), theo hai cách:
  - **★** theo **lần đầu** chạm ε (định nghĩa t_ε đang dùng trong notebook);
  - **☆** theo **luôn ≤ ε từ đó**.
  - Hai cách chọn khác nhau ở **Nesterov** (★ t = 6/L; ☆ t = 8/L); **SGD** (★ B = 4096, 1/√k, α₀ = 1; ☆ B = 1024, 1/√k, α₀ = 0,2); **FISTA** (★ t = 32/L; ☆ t = 6/L); các thuật toán còn lại trùng nhau.
- Thời gian đo có thể lệch tới khoảng 50% giữa các lần chạy, nên khi chênh lệch nhỏ thì so số vòng.

### 4.1 GD cố định (Ridge, 13 setup)

Lưới 0,1–34/L. Best 30/L ở mọi ε. 34/L vượt cận ổn định 2/λ_max ≈ 32/L: dao động, kẹt ở khoảng 10⁻⁴. Các bước ≤ 1/L chạy hết trần chưa tới F* + 10⁻⁸ vì quá chậm (vẫn đang giảm).

**Best ở ε = 10⁻⁶:**

- ★ lần đầu chạm ε: **t = 30/L**, 19,6 giây (600 vòng)
- ☆ luôn ≤ ε từ đó: **t = 30/L**, 19,6 giây (600 vòng) — cùng setup với ★

| Setup | Kết thúc | Vòng | Giây | Giây/vòng | F − F\* min | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 10⁻⁶ | 10⁻⁷ | Luôn ≤ 10⁻⁶ từ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t = 0,1/L | hết trần | 17922 | 600,0 | 0,0335 | 1,6·10⁻³ | – | – | – | – | – | – |
| t = 0,5/L | hết trần | 18004 | 600,0 | 0,0333 | 9,3·10⁻⁶ | 148,8 (4490) | 346,1 (10394) | 591,8 (17757) | – | – | – |
| t = 1/L | hết trần | 18164 | 600,0 | 0,0330 | 5,7·10⁻⁸ | 74,2 (2245) | 172,4 (5196) | 294,8 (8877) | 428,7 (12867) | 565,6 (17100) | 428,7 (12867) |
| t = 1,5/L | F* + 10⁻⁸ | 14453 | 471,3 | 0,0326 | 1,0·10⁻⁸ | 48,0 (1496) | 111,7 (3463) | 191,1 (5917) | 277,0 (8577) | 368,7 (11399) | 277,0 (8577) |
| t = 2/L | F* + 10⁻⁸ | 10839 | 363,4 | 0,0335 | 1,0·10⁻⁸ | 37,5 (1122) | 87,0 (2597) | 148,9 (4438) | 216,8 (6432) | 287,2 (8548) | 216,8 (6432) |
| t = 2,5/L | F* + 10⁻⁸ | 8670 | 288,5 | 0,0333 | 1,0·10⁻⁸ | 29,6 (897) | 69,0 (2077) | 118,0 (3550) | 171,2 (5145) | 227,5 (6838) | 171,2 (5145) |
| t = 3/L | F* + 10⁻⁸ | 7225 | 240,3 | 0,0333 | 1,0·10⁻⁸ | 24,7 (747) | 57,5 (1731) | 98,3 (2957) | 142,5 (4287) | 189,4 (5698) | 142,5 (4287) |
| t = 4/L | F* + 10⁻⁸ | 5417 | 180,1 | 0,0333 | 1,0·10⁻⁸ | 18,5 (560) | 43,1 (1297) | 73,7 (2217) | 106,8 (3214) | 142,0 (4272) | 106,8 (3214) |
| t = 6/L | F* + 10⁻⁸ | 3609 | 120,0 | 0,0332 | 1,0·10⁻⁸ | 12,3 (371) | 28,6 (863) | 49,0 (1476) | 71,2 (2141) | 94,6 (2846) | 71,2 (2141) |
| t = 12/L | F* + 10⁻⁸ | 1775 | 57,9 | 0,0326 | 1,0·10⁻⁸ | 5,1 (157) | 13,0 (399) | 23,1 (707) | 33,9 (1039) | 45,4 (1392) | 33,9 (1039) |
| t = 24/L | F* + 10⁻⁸ | 1189 | 38,7 | 0,0326 | 1,0·10⁻⁸ | 7,4 (226) | 15,5 (476) | 22,0 (674) | 27,7 (849) | 33,2 (1019) | 27,7 (849) |
| **t = 30/L ★ ☆** | F* + 10⁻⁸ | 870 | 28,3 | 0,0326 | 9,9·10⁻⁹ | **3,2 (98)** | **9,7 (296)** | **15,2 (465)** | **19,6 (600)** | **23,9 (733)** | **19,6 (600)** |
| t = 34/L | hết trần | 18512 | 600,0 | 0,0324 | 1,7·10⁻⁴ | 4,4 (134) | – | – | – | – | – |

### 4.2 GD backtracking (Ridge, 11 setup)

11 setup: 5 giá trị β ở c = 10⁻⁴, cộng β ∈ {0,1; 0,5} × c ∈ {10⁻²; 0,1; 0,3}. Mỗi vòng thử bước gấp đôi bước trước rồi co theo β. β nhỏ tốt hơn rõ; c gần như không ảnh hưởng (β = 0,1 với c = 10⁻⁴ và 10⁻² cùng 300 vòng tới 10⁻⁶). Giây/vòng gồm cả các lần thử bước.

**Best ở ε = 10⁻⁶:**

- ★ lần đầu chạm ε: **β = 0,1, c = 0,01**, 16,8 giây (300 vòng)
- ☆ luôn ≤ ε từ đó: **β = 0,1, c = 0,01**, 16,8 giây (300 vòng) — cùng setup với ★

| Setup | Kết thúc | Vòng | Giây | Giây/vòng | F − F\* min | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 10⁻⁶ | 10⁻⁷ | Luôn ≤ 10⁻⁶ từ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| β = 0,1, c = 0,0001 | F* + 10⁻⁸ | 478 | 26,9 | 0,0563 | 9,7·10⁻⁹ | 4,1 (72) | 8,1 (144) | 12,2 (217) | 16,9 (300) | 21,5 (383) | 16,9 (300) |
| β = 0,3, c = 0,0001 | F* + 10⁻⁸ | 558 | 34,1 | 0,0611 | 9,9·10⁻⁹ | 6,8 (111) | 11,2 (184) | 16,7 (274) | 22,2 (364) | 27,8 (455) | 22,2 (364) |
| β = 0,5, c = 0,0001 | F* + 10⁻⁸ | 682 | 46,8 | 0,0687 | 9,8·10⁻⁹ | 8,4 (121) | 14,9 (215) | 22,2 (321) | 29,9 (434) | 38,0 (553) | 29,9 (434) |
| β = 0,7, c = 0,0001 | F* + 10⁻⁸ | 740 | 62,9 | 0,0850 | 9,8·10⁻⁹ | 10,7 (126) | 19,2 (226) | 29,4 (345) | 39,7 (467) | 50,8 (598) | 39,7 (467) |
| β = 0,9, c = 0,0001 | F* + 10⁻⁸ | 739 | 123,2 | 0,1667 | 9,9·10⁻⁹ | 19,3 (116) | 36,5 (219) | 56,1 (337) | 77,1 (463) | 99,1 (595) | 77,1 (463) |
| **β = 0,1, c = 0,01 ★ ☆** | F* + 10⁻⁸ | 478 | 26,8 | 0,0561 | 9,7·10⁻⁹ | **4,1 (72)** | **8,1 (144)** | **12,2 (217)** | **16,8 (300)** | **21,4 (383)** | **16,8 (300)** |
| β = 0,1, c = 0,1 | F* + 10⁻⁸ | 485 | 27,2 | 0,0561 | 9,8·10⁻⁹ | 5,1 (92) | 9,1 (163) | 13,0 (232) | 17,0 (303) | 21,9 (391) | 17,0 (303) |
| β = 0,1, c = 0,3 | F* + 10⁻⁸ | 500 | 28,0 | 0,0561 | 9,9·10⁻⁹ | 5,4 (97) | 9,6 (172) | 13,7 (244) | 17,9 (320) | 22,8 (407) | 17,9 (320) |
| β = 0,5, c = 0,01 | F* + 10⁻⁸ | 684 | 46,8 | 0,0684 | 9,8·10⁻⁹ | 8,4 (123) | 14,9 (218) | 22,1 (323) | 29,8 (436) | 38,0 (555) | 29,8 (436) |
| β = 0,5, c = 0,1 | F* + 10⁻⁸ | 679 | 46,5 | 0,0685 | 9,9·10⁻⁹ | 8,0 (118) | 14,5 (212) | 21,7 (318) | 29,4 (431) | 37,7 (550) | 29,4 (431) |
| β = 0,5, c = 0,3 | F* + 10⁻⁸ | 656 | 44,9 | 0,0684 | 9,8·10⁻⁹ | 7,3 (107) | 12,9 (189) | 20,1 (295) | 27,9 (408) | 36,0 (527) | 27,9 (408) |

### 4.3 Nesterov (Ridge, 15 setup)

Lưới 0,1–24/L. Best 6/L ở ε = 10⁻⁶ và 10⁻⁷; 8/L nhanh hơn ở 10⁻³ và 10⁻⁵ (chênh 0,2–0,3 giây). 24/L vượt cận 1/λ_max ≈ 16/L: dao động, kẹt ở khoảng 10⁻⁵ (chạm 10⁻⁴ một lần lúc đầu nên ô 10⁻⁴ của nó nhỏ nhất, nhưng setup này không hội tụ). Thời gian không tính gradient tại w dùng để kiểm tra dừng.

**Best ở ε = 10⁻⁶:**

- ★ lần đầu chạm ε: **t = 6/L**, 7,7 giây (230 vòng)
- ☆ luôn ≤ ε từ đó: **t = 8/L**, 8,8 giây (272 vòng) — khác ★; setup ★ phải tới 9,5 giây mới ở hẳn dưới ε

| Setup | Kết thúc | Vòng | Giây | Giây/vòng | F − F\* min | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 10⁻⁶ | 10⁻⁷ | Luôn ≤ 10⁻⁶ từ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t = 0,1/L | F* + 10⁻⁸ | 11201 | 372,7 | 0,0333 | 1,0·10⁻⁸ | 10,5 (320) | 21,3 (644) | 36,8 (1108) | 94,1 (2831) | 170,7 (5133) | 107,3 (3229) |
| t = 0,5/L | F* + 10⁻⁸ | 4451 | 148,0 | 0,0333 | 1,0·10⁻⁸ | 4,7 (142) | 9,4 (283) | 16,4 (493) | 33,5 (1008) | 59,8 (1801) | 41,7 (1254) |
| t = 1/L | F* + 10⁻⁸ | 2749 | 91,4 | 0,0333 | 9,9·10⁻⁹ | 3,3 (100) | 6,5 (198) | 11,5 (347) | 23,5 (708) | 42,1 (1269) | 29,2 (880) |
| t = 1,5/L | F* + 10⁻⁸ | 2092 | 69,5 | 0,0332 | 1,0·10⁻⁸ | 2,7 (82) | 5,3 (160) | 9,3 (282) | 19,1 (576) | 34,4 (1034) | 23,7 (714) |
| t = 2/L | F* + 10⁻⁸ | 1690 | 56,1 | 0,0332 | 9,9·10⁻⁹ | 2,3 (71) | 4,5 (138) | 8,1 (244) | 16,5 (498) | 29,7 (894) | 20,4 (615) |
| t = 2,5/L | F* + 10⁻⁸ | 1505 | 50,1 | 0,0333 | 9,9·10⁻⁹ | 2,1 (63) | 4,1 (122) | 7,3 (218) | 14,8 (445) | 26,6 (799) | 18,2 (548) |
| t = 3/L | F* + 10⁻⁸ | 1270 | 42,2 | 0,0332 | 9,8·10⁻⁹ | 1,9 (58) | 3,6 (110) | 6,5 (198) | 13,4 (405) | 24,2 (728) | 16,5 (498) |
| t = 4/L | F* + 10⁻⁸ | 1095 | 36,4 | 0,0332 | 9,7·10⁻⁹ | 1,6 (50) | 3,1 (93) | 5,7 (171) | 11,6 (350) | 20,9 (630) | 14,2 (428) |
| **t = 6/L ★** | F* + 10⁻⁸ | 750 | 25,0 | 0,0333 | 9,8·10⁻⁹ | 1,3 (39) | 2,5 (74) | 4,6 (139) | **7,7 (230)** | **12,5 (376)** | 9,5 (284) |
| **t = 8/L ☆** | F* + 10⁻⁸ | 687 | 22,3 | 0,0324 | 1,0·10⁻⁸ | **1,1 (35)** | 2,4 (74) | **4,3 (134)** | 8,8 (272) | 15,2 (469) | **8,8 (272)** |
| t = 12/L | F* + 10⁻⁸ | 946 | 30,6 | 0,0324 | 9,5·10⁻⁹ | 2,4 (75) | 3,0 (92) | 6,3 (194) | 9,8 (303) | 16,8 (520) | 14,6 (452) |
| t = 14/L | F* + 10⁻⁸ | 934 | 30,5 | 0,0327 | 9,8·10⁻⁹ | 2,7 (82) | 3,1 (94) | 6,1 (189) | 9,5 (290) | 17,6 (538) | 16,9 (516) |
| t = 16/L | F* + 10⁻⁸ | 923 | 30,1 | 0,0326 | 9,5·10⁻⁹ | 2,7 (84) | 4,2 (128) | 7,3 (223) | 10,4 (319) | 18,0 (553) | 16,0 (493) |
| t = 18/L | F* + 10⁻⁸ | 778 | 25,3 | 0,0326 | 9,9·10⁻⁹ | 2,4 (73) | 2,7 (83) | 5,4 (166) | 8,3 (255) | 15,4 (472) | 13,6 (418) |
| t = 24/L | hết trần | 18564 | 600,0 | 0,0323 | 1,3·10⁻⁵ | 2,0 (60) | **2,3 (70)** | – | – | – | – |

### 4.4 Newton (Ridge, 7 setup)

Bước cố định t ∈ {0,1; 0,25; 0,5; 1} và backtracking β ∈ {0,1; 0,5; 0,9}. Backtracking luôn nhận t = 1 nên đường trùng hẳn t = 1 (chỉ khác thời gian đo). t < 1 mất hội tụ bậc hai: số vòng tăng gần tỉ lệ 1/t.

**Best ở ε = 10⁻⁶:**

- ★ lần đầu chạm ε: **t = 1**, 2,2 giây (7 vòng)
- ☆ luôn ≤ ε từ đó: **t = 1**, 2,2 giây (7 vòng) — cùng setup với ★

| Setup | Kết thúc | Vòng | Giây | Giây/vòng | F − F\* min | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 10⁻⁶ | 10⁻⁷ | Luôn ≤ 10⁻⁶ từ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t = 0,1 | F* + 10⁻⁸ | 109 | 35,1 | 0,3222 | 9,7·10⁻⁹ | 16,4 (51) | 20,5 (64) | 24,3 (76) | 27,9 (87) | 31,5 (98) | 27,9 (87) |
| t = 0,25 | F* + 10⁻⁸ | 42 | 13,5 | 0,3207 | 7,9·10⁻⁹ | 6,4 (20) | 8,0 (25) | 9,7 (30) | 11,0 (34) | 12,2 (38) | 11,0 (34) |
| t = 0,5 | F* + 10⁻⁸ | 20 | 6,5 | 0,3261 | 2,8·10⁻⁹ | 3,2 (10) | 4,2 (13) | 4,6 (14) | 5,3 (16) | 5,9 (18) | 5,3 (16) |
| **t = 1 ★ ☆** | F* + 10⁻⁸ | 8 | 2,5 | 0,3130 | 3,9·10⁻¹⁰ | **1,6 (5)** | **1,9 (6)** | **2,2 (7)** | **2,2 (7)** | **2,5 (8)** | **2,2 (7)** |
| bt, β = 0,1 | F* + 10⁻⁸ | 8 | 2,6 | 0,3289 | 3,9·10⁻¹⁰ | 1,7 (5) | 2,0 (6) | 2,3 (7) | 2,3 (7) | 2,6 (8) | 2,3 (7) |
| bt, β = 0,5 | F* + 10⁻⁸ | 8 | 3,4 | 0,4271 | 3,9·10⁻¹⁰ | 2,3 (5) | 2,7 (6) | 3,1 (7) | 3,1 (7) | 3,4 (8) | 3,1 (7) |
| bt, β = 0,9 | F* + 10⁻⁸ | 8 | 2,8 | 0,3490 | 3,9·10⁻¹⁰ | 1,7 (5) | 2,1 (6) | 2,5 (7) | 2,5 (7) | 2,8 (8) | 2,5 (7) |

### 4.5 SGD (Ridge, 18 setup)

Lưới batch {1024; 4096; 16384} × lịch {cố định; 1/√k} × α₀ {0,2; 1; 5}. Vòng = epoch, F đo trên toàn bộ dữ liệu cuối mỗi epoch. Không setup nào tới F* + 10⁻⁸ trong 600 giây. Với lịch cố định, chỉ B = 16384, α₀ = 0,2 tới được 10⁻⁶ (298 giây). Với lịch 1/√k, α₀ tốt tăng theo cỡ batch: B = 1024 hợp α₀ = 0,2, B = 4096 hợp α₀ = 1, B = 16384 hợp α₀ = 5 — vì vậy phải dò α₀ cho từng tổ hợp. Ở 10⁻³ và 10⁻⁴, B = 1024, α₀ = 1 nhanh nhất (cập nhật nhiều lần mỗi epoch), nhưng sau đó nhiễu giữ nó ở mức cao.

**Best ở ε = 10⁻⁶:**

- ★ lần đầu chạm ε: **B = 4096, 1/√k, α₀ = 1**, 111,5 giây (120 vòng)
- ☆ luôn ≤ ε từ đó: **B = 1024, 1/√k, α₀ = 0,2**, 526,0 giây (648 vòng) — khác ★; setup ★ phải tới 599,8 giây mới ở hẳn dưới ε

| Setup | Kết thúc | Vòng | Giây | Giây/vòng | F − F\* min | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 10⁻⁶ | 10⁻⁷ | Luôn ≤ 10⁻⁶ từ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B = 1024, cố định, α₀ = 0,2 | hết trần | 736 | 600,2 | 0,8154 | 2,1·10⁻⁵ | 4,2 (5) | 10,7 (13) | – | – | – | – |
| B = 1024, cố định, α₀ = 1 | hết trần | 736 | 600,1 | 0,8154 | 1,8·10⁻⁴ | 1,6 (2) | – | – | – | – | – |
| B = 1024, cố định, α₀ = 5 | hết trần | 737 | 600,6 | 0,8150 | 3,1·10⁻³ | – | – | – | – | – | – |
| **B = 1024, 1/√k, α₀ = 0,2 ☆** | hết trần | 739 | 600,2 | 0,8122 | 1,8·10⁻⁸ | 5,7 (7) | 22,5 (28) | 59,7 (74) | 126,8 (157) | 268,1 (330) | **526,0 (648)** |
| B = 1024, 1/√k, α₀ = 1 | hết trần | 740 | 600,4 | 0,8114 | 7,7·10⁻⁷ | **1,6 (2)** | **10,5 (13)** | 101,4 (125) | 484,1 (596) | – | – |
| B = 1024, 1/√k, α₀ = 5 | hết trần | 736 | 600,0 | 0,8152 | 2,4·10⁻⁵ | 7,5 (9) | 108,7 (133) | – | – | – | – |
| B = 4096, cố định, α₀ = 0,2 | hết trần | 651 | 600,2 | 0,9219 | 1,0·10⁻⁶ | 14,8 (16) | 34,4 (37) | 62,8 (68) | – | – | – |
| B = 4096, cố định, α₀ = 1 | hết trần | 654 | 600,1 | 0,9176 | 3,3·10⁻⁵ | 3,6 (4) | 11,7 (13) | – | – | – | – |
| B = 4096, cố định, α₀ = 5 | hết trần | 652 | 600,0 | 0,9203 | 4,6·10⁻⁴ | 5,6 (6) | – | – | – | – | – |
| B = 4096, 1/√k, α₀ = 0,2 | hết trần | 654 | 600,5 | 0,9183 | 3,0·10⁻⁵ | 67,6 (74) | 329,2 (359) | – | – | – | – |
| **B = 4096, 1/√k, α₀ = 1 ★** | hết trần | 649 | 600,7 | 0,9256 | 3,9·10⁻⁸ | 4,6 (5) | 20,1 (22) | **51,3 (55)** | **111,5 (120)** | **228,9 (248)** | 599,8 (648) |
| B = 4096, 1/√k, α₀ = 5 | hết trần | 650 | 600,5 | 0,9238 | 1,0·10⁻⁶ | 6,8 (7) | 14,0 (15) | 168,6 (182) | – | – | – |
| B = 16384, cố định, α₀ = 0,2 | hết trần | 770 | 600,6 | 0,7800 | 1,3·10⁻⁷ | 57,0 (62) | 117,6 (144) | 197,3 (248) | 298,0 (373) | – | 596,9 (765) |
| B = 16384, cố định, α₀ = 1 | hết trần | 816 | 600,3 | 0,7356 | 4,7·10⁻⁶ | 9,7 (13) | 24,1 (32) | 51,8 (68) | – | – | – |
| B = 16384, cố định, α₀ = 5 | hết trần | 820 | 600,5 | 0,7324 | 1,0·10⁻⁴ | 12,4 (17) | 32,9 (45) | – | – | – | – |
| B = 16384, 1/√k, α₀ = 0,2 | hết trần | 815 | 600,7 | 0,7371 | 1,3·10⁻³ | – | – | – | – | – | – |
| B = 16384, 1/√k, α₀ = 1 | hết trần | 817 | 600,7 | 0,7352 | 4,9·10⁻⁶ | 34,7 (47) | 166,5 (226) | 468,2 (636) | – | – | – |
| B = 16384, 1/√k, α₀ = 5 | hết trần | 819 | 600,6 | 0,7333 | 1,3·10⁻⁷ | 43,9 (60) | 94,4 (129) | 146,3 (200) | 224,0 (306) | – | 580,8 (792) |

### 4.6 Subgradient (Lasso, 5 setup)

α_k = α₀/√k, chọn subgradient chuẩn nhỏ nhất. Không setup nào tới 10⁻⁶ trong 600 giây; best lấy ở ε nhỏ nhất đạt được (10⁻⁵: α₀ = 30).

**Best ở ε = 10⁻⁵:** (không setup nào tới 10⁻⁶)

- ★ lần đầu chạm ε: **α₀ = 30**, 379,3 giây (11734 vòng)
- ☆ luôn ≤ ε từ đó: **α₀ = 30**, 567,8 giây (17566 vòng) — cùng setup với ★

| Setup | Kết thúc | Vòng | Giây | Giây/vòng | F − F\* min | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 10⁻⁶ | 10⁻⁷ | Luôn ≤ 10⁻⁵ từ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| α₀ = 0,5 | hết trần | 18984 | 600,0 | 0,0316 | 6,5·10⁻³ | – | – | – | – | – | – |
| α₀ = 2 | hết trần | 18623 | 600,0 | 0,0322 | 6,5·10⁻⁴ | 387,5 (12066) | – | – | – | – | – |
| α₀ = 8 | hết trần | 18548 | 600,0 | 0,0323 | 3,9·10⁻⁵ | **120,8 (3730)** | 435,9 (13471) | – | – | – | – |
| **α₀ = 30 ★ ☆** | hết trần | 18563 | 600,0 | 0,0323 | 6,3·10⁻⁶ | 125,4 (3882) | **219,6 (6796)** | **379,3 (11734)** | – | – | **567,8 (17566)** |
| α₀ = 100 | hết trần | 18576 | 600,0 | 0,0323 | 2,2·10⁻⁵ | 394,1 (12204) | 442,3 (13697) | – | – | – | – |

### 4.7 ISTA (Lasso, 13 setup)

Lưới 0,1–2/L, 6/L và 19–128/L. Best 19/L ở mọi ε. Bước lớn hơn (29–128/L) vẫn hội tụ (cận cục bộ 2/λ_max(H_S) ≈ 130/L) nhưng giai đoạn đầu chậm, nên tới 10⁻³ mất 10–61 giây. Giây/vòng của 64–128/L cao bất thường (0,05) — chưa rõ nguyên nhân.

**Best ở ε = 10⁻⁶:**

- ★ lần đầu chạm ε: **t = 19/L**, 20,9 giây (626 vòng)
- ☆ luôn ≤ ε từ đó: **t = 19/L**, 20,9 giây (626 vòng) — cùng setup với ★

| Setup | Kết thúc | Vòng | Giây | Giây/vòng | F − F\* min | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 10⁻⁶ | 10⁻⁷ | Luôn ≤ 10⁻⁶ từ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t = 0,1/L | hết trần | 18889 | 600,0 | 0,0318 | 2,0·10⁻³ | – | – | – | – | – | – |
| t = 0,5/L | hết trần | 18642 | 600,0 | 0,0322 | 1,5·10⁻⁵ | 175,1 (5484) | 413,6 (12867) | – | – | – | – |
| t = 1/L | hết trần | 18614 | 600,0 | 0,0322 | 2,0·10⁻⁷ | 87,5 (2741) | 206,8 (6433) | 321,5 (9986) | 472,1 (14651) | – | 472,1 (14651) |
| t = 1,5/L | hết trần | 18403 | 600,0 | 0,0326 | 3,0·10⁻⁸ | 58,4 (1827) | 137,8 (4289) | 214,3 (6657) | 314,6 (9767) | 449,5 (13894) | 314,6 (9767) |
| t = 2/L | hết trần | 15884 | 600,0 | 0,0378 | 2,0·10⁻⁸ | 52,4 (1370) | 151,0 (3216) | 231,3 (4992) | 309,4 (7324) | 417,4 (10420) | 309,4 (7324) |
| t = 6/L | F* + 10⁻⁸ | 6571 | 219,5 | 0,0334 | 1,0·10⁻⁸ | 14,9 (454) | 35,6 (1070) | 55,6 (1661) | 82,1 (2439) | 117,1 (3483) | 82,1 (2439) |
| **t = 19/L ★ ☆** | F* + 10⁻⁸ | 1923 | 64,4 | 0,0335 | 1,0·10⁻⁸ | **4,6 (138)** | **11,0 (330)** | **15,7 (471)** | **20,9 (626)** | **28,7 (859)** | **20,9 (626)** |
| t = 29/L | F* + 10⁻⁸ | 1298 | 44,1 | 0,0340 | 1,0·10⁻⁸ | 10,6 (315) | 17,9 (530) | 23,7 (705) | 29,7 (876) | 35,8 (1054) | 29,7 (876) |
| t = 32/L | F* + 10⁻⁸ | 1231 | 42,0 | 0,0341 | 9,9·10⁻⁹ | 12,7 (372) | 19,7 (577) | 25,3 (741) | 30,6 (897) | 35,9 (1054) | 30,6 (897) |
| t = 48/L | F* + 10⁻⁸ | 1351 | 45,9 | 0,0340 | 1,0·10⁻⁸ | 26,4 (769) | 31,7 (924) | 35,4 (1037) | 39,0 (1143) | 42,4 (1246) | 39,0 (1143) |
| t = 64/L | F* + 10⁻⁸ | 1116 | 55,6 | 0,0498 | 9,8·10⁻⁹ | 32,8 (680) | 38,8 (797) | 43,1 (881) | 47,2 (960) | 51,4 (1038) | 47,2 (960) |
| t = 96/L | F* + 10⁻⁸ | 1476 | 76,0 | 0,0515 | 9,7·10⁻⁹ | 61,1 (1186) | 65,1 (1264) | 68,0 (1320) | 70,7 (1373) | 73,3 (1424) | 70,7 (1373) |
| t = 128/L | F* + 10⁻⁸ | 1387 | 70,9 | 0,0511 | 9,4·10⁻⁹ | 59,8 (1170) | 62,7 (1228) | 64,9 (1270) | 66,9 (1309) | 68,9 (1348) | 66,9 (1309) |

### 4.8 FISTA (Lasso, 16 setup)

Lưới 0,1–6/L và 19–128/L. Best 6/L ở ε ≥ 10⁻⁵, 32/L ở 10⁻⁶ và 10⁻⁷ (29/L chậm hơn 0,1 giây). 64/L trở lên vượt cận 1/λ_max(H_S) ≈ 65/L: không tới nổi 10⁻³. FISTA bước lớn dao động mạnh quanh F*, nên t_ε (lần đầu chạm ε) lạc quan cho các bước này.

**Best ở ε = 10⁻⁶:**

- ★ lần đầu chạm ε: **t = 32/L**, 9,4 giây (282 vòng)
- ☆ luôn ≤ ε từ đó: **t = 6/L**, 11,9 giây (345 vòng) — khác ★; setup ★ phải tới 16,4 giây mới ở hẳn dưới ε

| Setup | Kết thúc | Vòng | Giây | Giây/vòng | F − F\* min | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 10⁻⁶ | 10⁻⁷ | Luôn ≤ 10⁻⁶ từ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t = 0,1/L | F* + 10⁻⁸ | 9046 | 302,2 | 0,0334 | 9,8·10⁻⁹ | 12,3 (374) | 24,7 (748) | 51,4 (1545) | 94,1 (2824) | 150,8 (4518) | 122,0 (3657) |
| t = 0,5/L | F* + 10⁻⁸ | 4035 | 134,8 | 0,0334 | 9,9·10⁻⁹ | 5,5 (166) | 11,0 (329) | 17,3 (519) | 41,7 (1251) | 67,1 (2009) | 53,7 (1607) |
| t = 1/L | F* + 10⁻⁸ | 2686 | 89,8 | 0,0334 | 9,8·10⁻⁹ | 3,8 (117) | 7,6 (230) | 12,1 (365) | 29,3 (878) | 47,3 (1416) | 33,5 (1004) |
| t = 1,5/L | F* + 10⁻⁸ | 2185 | 73,1 | 0,0335 | 9,8·10⁻⁹ | 3,1 (95) | 6,2 (187) | 9,9 (297) | 23,8 (713) | 38,6 (1153) | 27,3 (817) |
| t = 2/L | F* + 10⁻⁸ | 1757 | 58,8 | 0,0334 | 9,9·10⁻⁹ | 2,7 (82) | 5,3 (161) | 8,5 (256) | 20,5 (615) | 33,3 (997) | 23,6 (705) |
| t = 2,5/L | F* + 10⁻⁸ | 1566 | 52,4 | 0,0334 | 9,9·10⁻⁹ | 2,4 (73) | 4,8 (144) | 7,6 (229) | 18,3 (548) | 29,8 (890) | 21,0 (629) |
| t = 3/L | F* + 10⁻⁸ | 1427 | 48,4 | 0,0339 | 9,6·10⁻⁹ | 2,2 (67) | 4,3 (130) | 7,0 (208) | 16,7 (498) | 27,7 (812) | 19,2 (572) |
| t = 4/L | F* + 10⁻⁸ | 1232 | 42,0 | 0,0341 | 9,5·10⁻⁹ | 1,9 (58) | 3,7 (112) | 6,0 (180) | 14,3 (428) | 23,6 (702) | 16,4 (492) |
| **t = 6/L ☆** | F* + 10⁻⁸ | 935 | 32,7 | 0,0350 | 9,7·10⁻⁹ | **1,6 (46)** | **3,0 (85)** | **5,1 (147)** | 11,9 (345) | 17,6 (508) | **11,9 (345)** |
| t = 19/L | F* + 10⁻⁸ | 777 | 26,2 | 0,0337 | 7,9·10⁻⁹ | 3,7 (110) | 3,9 (117) | 8,5 (255) | 11,7 (352) | 13,3 (400) | 20,9 (620) |
| t = 29/L | F* + 10⁻⁸ | 588 | 19,6 | 0,0333 | 9,4·10⁻⁹ | 2,9 (86) | 3,1 (94) | 5,6 (169) | 9,5 (284) | 10,7 (322) | 15,3 (460) |
| **t = 32/L ★** | F* + 10⁻⁸ | 574 | 19,2 | 0,0334 | 8,6·10⁻⁹ | 3,3 (98) | 3,4 (103) | 5,8 (174) | **9,4 (282)** | **10,7 (320)** | 16,4 (490) |
| t = 48/L | F* + 10⁻⁸ | 514 | 17,2 | 0,0334 | 9,8·10⁻⁹ | 3,3 (98) | 3,4 (102) | 5,3 (160) | 10,2 (306) | 12,2 (365) | 13,9 (417) |
| t = 64/L | hết trần | 15173 | 600,0 | 0,0395 | 5,1·10⁻³ | – | – | – | – | – | – |
| t = 96/L | hết trần | 11696 | 600,0 | 0,0513 | 4,9·10⁻² | – | – | – | – | – | – |
| t = 128/L | hết trần | 9883 | 600,0 | 0,0607 | 7,3·10⁻² | – | – | – | – | – | – |
