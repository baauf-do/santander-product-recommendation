# Điều kiện dừng & biểu đồ $F(w)-F^\star$ — nhật ký thử nghiệm

Ghi lại lần sửa [bai_lam.ipynb](bai_lam.ipynb): thay cách xác định hội tụ bằng **điều kiện dừng**,
vẽ **$F(w)-F^\star$** trên trục log, và đổi quy tắc chọn **setup tốt nhất** cho khớp. Mọi con số dưới
đây lấy từ lần chạy toàn bộ notebook ngày 2026-10-03 (dữ liệu như notebook: 466 243 dòng train ×
161 cột, sản phẩm `ind_recibo_ult1`, $\lambda=\lambda_1=10^{-3}$).

---

## 1. Thay đổi gì

| | Trước | Sau |
|---|---|---|
| Vòng lặp | chạy **đủ** `VONG = 250` vòng | dừng khi $\|r(w_k)\|\le\text{TOL}\cdot\|r(w_0)\|$; `VONG = 500` chỉ là **ngân sách tối đa** |
| Xác định hội tụ (mục 6b) | $k_\varepsilon$: lần đầu $(F-F^\star)/F^\star\le1\%$, với $F^\star$ = min của chính các lần chạy | dừng **trước** khi hết ngân sách ⇔ đạt điều kiện dừng ⇔ hội tụ |
| Trục tung | $F(w)$ | $F(w)-F^\star$, thang log (trừ các hình quét SGD) |
| Setup tốt nhất | $F$ cuối thấp nhất | trong các setup **đã hội tụ**: nhanh nhất theo giây; không cái nào hội tụ → $F$ cuối thấp nhất |
| In kết quả | `F = ...` | thêm cột *hội tụ sau k vòng / chưa hội tụ / phân kỳ* (hàm `trang_thai`) |

Thước đo tối ưu $r$ cho từng thuật toán:

| Thuật toán | $r(w)$ | Chi phí thêm |
|---|---|---|
| GD, GD backtracking, Nesterov, Newton (Ridge) | $\nabla F(w)$ | 0 — gradient vốn đã tính (Nesterov: lấy luôn `g` mà `ridge(w)` trả về cùng $F(w)$) |
| SGD | $\nabla F(w)$ trên **toàn bộ** dữ liệu, kiểm tra cuối mỗi epoch | 1 gradient đầy đủ / epoch, **đồng hồ tạm dừng** (như lúc ghi loss) |
| Subgradient (Lasso) | subgradient chuẩn nhỏ nhất $s$ | 0 — $s$ vốn đã tính |
| ISTA / FISTA (Lasso) | gradient mapping $\big(w-\mathrm{prox}_{t\lambda_1}(w-t\nabla\ell(w))\big)/t$ | 1 phép prox $O(d)$ |

---

## 2. Vì sao chọn như vậy

### 2.1 Vì sao dùng chuẩn gradient, không dùng $F$

- **Cách cũ cần biết trước $F^\star$**, mà $F^\star$ lại lấy từ chính các lần chạy đang được đánh giá
  → vòng luẩn quẩn: lần chạy tốt nhất luôn "hội tụ" theo định nghĩa. Một thuật toán thật thì không
  biết $F^\star$, nên cũng không dùng được tiêu chí đó để quyết định dừng.
- **$\|\nabla F\|$ đo được ngay tại chỗ** và bằng 0 khi và chỉ khi $w$ là nghiệm (Ridge lồi chặt).
  Với Lasso không có gradient nên dùng thước đo tương đương: subgradient chuẩn nhỏ nhất ($=0$ ⇔ thoả
  KKT) và gradient mapping ($=0$ ⇔ $w$ là điểm bất động của bước prox). Tại $w=0$, cả hai thước đo
  Lasso đều bằng soft-threshold của $\nabla\ell(0)$ theo $\lambda_1$ → hai thuật toán Lasso dùng
  **cùng một mốc** $\|r(w_0)\|$.
- **Không dùng "$|F_k-F_{k-1}|$ nhỏ"**: với bài toán xấu điều kiện như ở đây, GD giảm $F$ rất chậm
  nên tiêu chí này sẽ dừng sớm và báo nhầm là đã hội tụ.

### 2.2 Vì sao dùng ngưỡng **tương đối** ($\text{TOL}\cdot\|r(w_0)\|$)

$\|\nabla F(w_0)\|$ phụ thuộc dữ liệu và nhãn (ở đây $\approx0{,}489$). Mục 7 huấn luyện 24 sản phẩm
với tỉ lệ dương rất khác nhau, nên một ngưỡng tuyệt đối sẽ quá chặt với nhãn này và quá lỏng với nhãn
khác. Ngưỡng tương đối nghĩa là "giảm gradient đi $1/\text{TOL}$ lần", và có cùng ý nghĩa cho mọi nhãn.

### 2.3 Thí nghiệm chọn `TOL` và `VONG`

Trước khi sửa, mình đo xem mỗi thuật toán cần bao nhiêu vòng để $\|r(w_k)\|/\|r(w_0)\|$ xuống dưới
từng mức (giới hạn 3000 vòng):

| Thuật toán | $10^{-2}$ | $3\cdot10^{-3}$ | $10^{-3}$ | $3\cdot10^{-4}$ | $10^{-4}$ | giây / 1000 vòng |
|---|---|---|---|---|---|---|
| GD $t=1{,}9/L$ | 700 | 1688 | > 3000 | > 3000 | > 3000 | 33 |
| Nesterov $t=1/L$ | 84 | 119 | 317 | 558 | 915 | 66 |
| Nesterov $t=2/L$ | 59 | 83 | 172 | 388 | 642 | 66 |
| ISTA $t=1/L$ | 1368 | > 3000 | > 3000 | > 3000 | > 3000 | 33 |
| FISTA $t=1/L$ | 88 | 133 | 335 | 507 | 1025 | 65 |
| Newton (backtracking) | — | — | — | — | 9 vòng tới sai số máy ($\|g\|/\|g_0\|\approx4\cdot10^{-17}$) | — |

Khoảng cách giữa GD và Nesterov khớp với bài toán **xấu điều kiện** ($L/\mu\approx6250$, xem 2.4):
GD cần cỡ $\kappa\log(1/\text{TOL})$ vòng, Nesterov chỉ cần cỡ $\sqrt\kappa\log(1/\text{TOL})$. Từ
bảng trên:

- **`TOL = 1e-3`**: là mốc chặt nhất mà các phương pháp tăng tốc (Nesterov, FISTA) vẫn đạt được trong
  vài trăm vòng, trong khi GD và ISTA thì không. Như vậy thí nghiệm **phân biệt được** $O(1/k)$ với
  $O(1/k^2)$, đúng điều bài muốn chỉ ra. Chọn `1e-4` thì gần như chỉ Newton hội tụ, bảng so sánh mất
  thông tin. Chọn `1e-2` thì quá lỏng: $F$ lúc dừng còn cách $F^\star$ cỡ $10^{-3}$.
- **`VONG = 500`** (tăng từ 250): đủ chỗ cho Nesterov 1/L (317 vòng) và FISTA 1/L (335 vòng) dừng thật.
  Nếu để 250, cả hai sẽ bị ghi là *chưa hội tụ* chỉ vì thiếu ngân sách. Tăng lên 3000 cũng không cứu
  được GD/ISTA mà thời gian chạy tăng 6 lần (một lượt quét GD ≈ 16 giây × 6 bước ở 500 vòng).
- **`VONG_NT = 30`, `EPOCH = 42`** giữ nguyên: Newton dừng ở vòng 6, còn SGD bước cố định về lý
  thuyết không hội tụ, nên thêm epoch cũng không thay đổi kết luận.

### 2.4 Bằng chứng bài toán xấu điều kiện

Nói "xấu điều kiện" thì phải đo được. Mình tính trực tiếp trên `X_tr` (466 243 × 161), với $w^\star$
lấy từ Newton chạy tới sai số máy:

| Đại lượng | Giá trị | Ý nghĩa |
|---|---|---|
| Trị riêng nhỏ nhất của $X^\top X/n$ | $\approx 0$ (−4·10⁻¹⁴) | $X$ **suy biến**: có tổ hợp cột gần như trùng nhau |
| Cặp cột có $\lvert\text{corr}\rvert>0{,}99$ / $>0{,}9$ | 86 / 282 | ví dụ `ind_reca_fin_ult1_lag` ~ `ind_reca_fin_ult1_thang`: −1,0000 |
| Trị riêng của Hessian $\nabla^2F(w^\star)$: lớn nhất / nhỏ nhất | 0,392 / **0,001** | trị riêng nhỏ nhất **đúng bằng $\lambda$** |
| Số trị riêng của Hessian $<2\cdot10^{-3}$ | **92 / 161** | hơn một nửa số hướng có độ cong **chỉ đến từ** phần phạt $\lambda$ |
| $\kappa$ cục bộ tại nghiệm $=\lambda_{\max}/\lambda_{\min}$ | **392** | xấu điều kiện ngay cả với thông tin đúng nhất |
| $L=\lambda_{\max}(X^\top X/n)/4$ (bước 1/L dùng) | 6,25 | lớn hơn độ cong thật lớn nhất (0,392) **16 lần** |
| $\kappa$ mà GD bước $1/L$ "nhìn thấy" $=L/\lambda$ | **≈ 6250** | con số quyết định tốc độ của GD/ISTA bước cố định |

**Nguồn gốc 1: đặc trưng trùng lặp.** Hai họ cột `*_lag2..5` (sở hữu sản phẩm ở các tháng trước) và
`*_thang` (số tháng kể từ lần cuối có sản phẩm) gần như là hàm của nhau. Với khách hàng ổn định thì
"có sản phẩm ở $t-1..t-5$" ⇔ "`_thang` = 1". Hướng phẳng nhất của Hessian (trị riêng 0,001) gần như
toàn bộ nằm trên một nhóm như vậy: `ind_ahor_fin_ult1_thang` (hệ số 0,91) cùng 5 cột `ind_ahor_fin_ult1_lag*`
(mỗi cột ≈ 0,18). Dữ liệu **không có thông tin** để phân biệt các hệ số dọc những hướng này, nên
$\ell(w)$ gần như phẳng theo chúng, và chỉ còn $\frac\lambda2\|w\|^2$ "giữ" lại với độ cong $\lambda$.
Không có Ridge thì bài toán thậm chí không lồi chặt.

**Nguồn gốc 2: nhãn rất lệch.** Tỉ lệ dương là 1,2%, nên tại $w^\star$ hầu hết $\sigma(z_i)\approx0$
và $\sigma'(z_i)=\sigma(1-\sigma)$ rất nhỏ: trung bình 0,0102, trung vị 0,0017, trong khi cận dùng để
tính $L$ là 0,25. Hessian $\frac1nX^\top\mathrm{diag}(\sigma')X$ vì vậy bị "co" khoảng 25 lần so với $X^\top X/(4n)$.
Hệ quả có hai mặt:
- Các hướng có độ cong thật đã nhỏ lại càng nhỏ, còn sàn $\lambda$ thì không đổi, nên $\kappa$ cục bộ
  bị nén về phía sàn.
- $L$ là cận trên quá rộng: bước $1/L$ ngắn hơn bước an toàn thật rất nhiều. Đây cũng là lý do
  **$t=4/L$ không phân kỳ** (nhận xét 3). Gần nghiệm, bước ổn định tới $2/0{,}392\approx5{,}1\approx32/L$;
  chỉ ở những vòng đầu, khi $w\approx0$ và $\sigma'=0{,}25$ ở mọi điểm, độ cong mới chạm $L$, nên 4/L
  dao động khoảng 10 vòng đầu rồi mới đi xuống đều.

**Kiểm chứng bằng tốc độ đo được.** Cho GD bước $1/L$ chạy 500 vòng rồi khớp đường thẳng vào
$\log(F-F^\star)$ ở vòng 300–500: mỗi vòng sai số chỉ giảm theo tỉ lệ **0,99737**, tức **khoảng
875 vòng mới giảm 10 lần**. Sau 500 vòng sai số mới từ 0,65 xuống 0,0115. Ngoại suy ra, phải cần
khoảng **4000 vòng** (≈ 2 phút) để tới $10^{-6}$. Cận xấu nhất của lý thuyết là $1-\mu/L=0{,}99984$
(14 392 vòng mỗi bậc). Thực tế nhanh hơn cận này vì sai số tập trung ở các hướng cong vừa, chưa đi
vào các hướng chạm sàn $\lambda$, nhưng vẫn chậm gần 900 vòng mỗi bậc.

**Hệ quả cho các thí nghiệm:**
- GD, ISTA bước cố định: không thể đạt `TOL = 1e-3` trong ngân sách hợp lý → *chưa hội tụ* là kết
  quả đúng, không phải lỗi cài đặt.
- Nesterov, FISTA: phụ thuộc $\sqrt\kappa\approx79$ thay vì $\kappa$, nên hội tụ trong vài trăm vòng.
- Newton: nhân với $H^{-1}$ là "chuẩn hoá" lại mọi hướng, nên $\kappa$ gần như không ảnh hưởng →
  6 vòng.
- GD backtracking: tự tìm bước lớn hơn $1/L$, nên khắc phục được phần "$L$ quá rộng" (16×), dù không
  khắc phục được phần $\kappa=392$ thật.
- Muốn GD bước cố định chạy tốt thì phải sửa **bài toán**, không phải thuật toán: bỏ bớt một trong
  hai họ cột `*_lag2..5` / `*_thang`, hoặc tăng $\lambda$.

<details><summary>Code tính các con số trên (chạy sau ô nạp dữ liệu và ô hàm mục tiêu)</summary>

```python
LAM = 1e-3
phat = np.full(d, LAM); phat[0] = 0
print("eig X^T X/n:", np.linalg.eigvalsh(X_tr.T @ X_tr / n)[[0, -1]])
tol_cu, TOL = TOL, 0.0                      # tam tat dieu kien dung -> Newton chay toi sai so may
w = newton(X_tr, y_tr, LAM, so_vong=15, backtracking=True)[0]
TOL = tol_cu
s = sigmoid(X_tr @ w); s = s * (1 - s)
H = X_tr.T @ (X_tr * s[:, None]) / n + np.diag(phat)
eH, V = np.linalg.eigh(H)
print("sigma' mean/median:", s.mean(), np.median(s))
print("eig H min/max, kappa:", eH[0], eH[-1], eH[-1] / eH[0], " so eig < 2e-3:", (eH < 2e-3).sum())
print("huong phang nhat:", [(TEN[i], V[i, 0]) for i in np.argsort(-abs(V[:, 0]))[:6]])
```
</details>

### 2.5 Vì sao vẽ $F(w)-F^\star$ trên trục log

Khi vẽ $F(w)$, mọi đường đều dồn vào một dải mỏng cỡ $10^{-5}$ sát đáy, không phân biệt được. Trên thang
log của $F-F^\star$:
hội tụ **tuyến tính** là **đường thẳng** (độ dốc = tốc độ), hội tụ **bậc hai** của Newton là đường
**cắm xuống ngày càng dốc**, và có thể đọc trực tiếp sai số lúc dừng.

**$F^\star$ lấy từ đâu** — phải độc lập với các đường được vẽ, không thì đường tốt nhất luôn chạm 0:

- **Ridge**: `LogisticRegression(solver="newton-cholesky", tol=1e-12)` của sklearn →
  $F^\star=0{,}043630121211$, khớp tới 12 chữ số với Newton tự viết chạy tới sai số máy. sklearn
  `lbfgs`/`newton-cholesky` không phạt intercept nên tối thiểu hoá **đúng** hàm $F$ của mình.
- **Lasso**: không có lời giải tham chiếu độc lập, vì `liblinear` **phạt cả intercept** nên giải một hàm
  hơi khác. Vì vậy $F^\star$ = $F$ nhỏ nhất từng thấy trên mọi lần chạy Lasso (= 0,046564887, của
  FISTA 0,5/L). Hệ quả: đường đạt $F^\star$ có điểm cuối $=0$ và bị ẩn trên trục log. Chấp nhận được,
  nhưng là chỗ kém chặt hơn Ridge.

**Chỗ *không* vẽ $F-F^\star$**: các hình quét SGD (`sgd_eta_*`, `sgd_lich_*`) vẫn vẽ $F$. Đại lượng ghi
ở đó là loss **trên mini-batch**, nên có thể nhỏ hơn $F^\star$ do nhiễu, và $F-F^\star$ âm thì không
có nghĩa trên trục log. Trong hình tổng kết Ridge, SGD được lấy **trung bình theo epoch** cho bớt
nhiễu, nhưng sau khoảng 25 epoch giá trị trung bình vẫn lọt xuống dưới $F^\star$ nên phần đuôi bị ẩn.
Đây là bằng chứng trực quan rằng SGD bước cố định chỉ lảng vảng trong "quả cầu nhiễu" quanh nghiệm.

### 2.6 Vì sao đổi quy tắc chọn setup tốt nhất

Lần chạy đầu vẫn chọn theo "$F$ cuối thấp nhất", và kết quả bị **ngược**: Nesterov chọn $t=0{,}5/L$
(454 vòng) thay vì $2/L$ (172 vòng), GD backtracking chọn $\beta=0{,}9$ (110 giây) thay vì $\beta=0{,}1$
(12 giây), FISTA chọn $0{,}5/L$ (486 vòng) thay vì $2/L$ (231 vòng). Khi đã có điều kiện dừng, mọi setup
hội tụ đều dừng ở **cùng một mức** $\|\nabla F\|$. Setup nào chạy càng lâu thì $F$ lúc dừng càng sát
$F^\star$ một chút, nên chọn theo $F$ thấp nhất là chọn đúng cái **chậm nhất**. Quy tắc mới: trong
các setup đã hội tụ, chọn cái **nhanh nhất theo giây**. Nếu không setup nào hội tụ (GD cố định,
SGD, subgradient) thì vẫn chọn theo $F$ cuối như cũ.

---

## 3. Kết quả

### 3.1 Quét tham số (mỗi dòng: trạng thái điều kiện dừng, $F$ lúc dừng, thời gian)

**GD bước cố định** (Ridge) — không bước nào hội tụ trong 500 vòng

| $t$ | 0,1/L | 0,5/L | 1/L | 1,5/L | 2/L | 4/L |
|---|---|---|---|---|---|---|
| $F$ (500 vòng) | 0,16144 | 0,06713 | 0,05517 | 0,05064 | 0,04823 | **0,04491** |

**GD backtracking** — hội tụ với mọi $\beta$

| $\beta$ | 0,1 | 0,3 | 0,5 | 0,7 | 0,9 |
|---|---|---|---|---|---|
| vòng | **164** | 208 | 261 | 315 | 394 |
| giây | **12,5** | 17,5 | 25,6 | 40,8 | 110,1 |
| $F$ lúc dừng | 0,0436874 | 0,0436857 | 0,0436660 | 0,0436468 | 0,0436335 |

**Nesterov** — hội tụ với 4/5 bước

| $t$ | 0,1/L | 0,5/L | 1/L | 1,5/L | 2/L |
|---|---|---|---|---|---|
| vòng | > 500 | 454 | 317 | 205 | **172** |
| giây | 32,5 | 29,7 | 20,7 | 13,4 | **11,2** |

**Newton** — bước 0,1: chưa hội tụ trong 30 vòng; 0,25: 27 vòng; 0,5: 13 vòng; **1: 6 vòng (2,0 s)**;
backtracking với $\beta\in\{0{,}1;0{,}5;0{,}9\}$: 6 vòng, ba đường trùng nhau vì lần nào bước $t=1$
cũng được chấp nhận ngay.

**SGD** (batch 4096, 42 epoch) — không lần nào hội tụ. Bước cố định tốt nhất $\alpha_0=0{,}2$ (loss
epoch cuối 0,044234). Các lịch giảm bước còn tệ hơn: $\alpha_0/\sqrt k$ cho 0,045913, $\alpha_0/k$
cho 0,052650.

**Subgradient** (Lasso) — không lần nào hội tụ. Tốt nhất $\alpha_0=8$: $F=0{,}05004$, 154/160 hệ số khác 0.

**ISTA** — không hội tụ, tốt nhất $2/L$: $F=0{,}05184$.
**FISTA** — hội tụ với 4/5 bước: 0,5/L → 486 vòng, 1/L → 335, 1,5/L → 269, **2/L → 231 vòng
(15,1 s)**, 19/160 hệ số khác 0.

### 3.2 Tổng kết (mục 6 / 6b, setup tốt nhất theo quy tắc mới)

| Thuật toán | Bài | Setup | Điều kiện dừng | Giây | $F$ lúc dừng | log-loss train | log-loss val |
|---|---|---|---|---|---|---|---|
| GD bước cố định | Ridge | $t=4/L$ | chưa (> 500) | > 16,3 | 0,0449078 | 0,043159 | 0,038855 |
| GD backtracking | Ridge | $\beta=0{,}1$ | 164 vòng | 12,5 | 0,0436874 | 0,040929 | 0,036530 |
| GD Nesterov | Ridge | $t=2/L$ | 172 vòng | 11,3 | 0,0436768 | 0,040421 | 0,036007 |
| **Newton** | Ridge | $t=1$ | **6 vòng** | **2,3** | 0,0436497 | 0,040680 | 0,036263 |
| SGD | Ridge | $\alpha_0=0{,}2$ | chưa (> 42 epoch) | > 30,9 | ≈ 0,04423* | 0,040939 | 0,036546 |
| sklearn lbfgs | Ridge | mặc định | 27 vòng | 1,3 | 0,0436392 | 0,040654 | 0,036241 |
| Subgradient | Lasso | $\alpha_0=8$ | chưa (> 500) | > 16,4 | 0,0500414 | 0,042725 | 0,038013 |
| **FISTA** | Lasso | $t=2/L$ | **231 vòng** | 15,1 | 0,0465776 | 0,040801 | 0,036070 |
| sklearn liblinear | Lasso | L1 | 25 vòng | 6,6 | 0,0466888 | 0,041495 | 0,036835 |

\* SGD: trung bình loss mini-batch ở epoch cuối, không phải $F$ trên toàn bộ.

**MAP@7** (mục 7–8, không đổi so với trước khi sửa): logistic Ridge ×24 (Newton) đạt 0,7976 trên khách
có thêm mới và 0,0225 trên toàn bộ khách. LightGBM softmax đạt 0,8197 / 0,0231.

---

## 4. Nhận xét

1. **Điều kiện dừng tách bạch các lớp thuật toán đúng như lý thuyết.** Với cùng `TOL = 1e-3` và
   cùng ngân sách: các phương pháp **bậc hai / tăng tốc / bước thích nghi** (Newton, Nesterov, FISTA,
   GD backtracking) hội tụ; các phương pháp **bậc nhất bước cố định** (GD, ISTA), **subgradient**
   ($O(1/\sqrt k)$) và **SGD bước cố định** thì không. Cùng bước $1/L$, GD sau 500 vòng vẫn ở
   $F=0{,}0552$, còn Nesterov đã dừng ở vòng 317 với $F=0{,}04366$. Tương tự, ISTA chưa hội tụ còn
   FISTA hội tụ.

2. **Newton thắng áp đảo** (6 vòng, ~2 giây). Hình $F-F^\star$ cho thấy rõ: sai số giảm mỗi vòng một
   bậc rồi cắm xuống. Với $d=160$, dựng và giải Hessian rất rẻ nên lợi thế bậc hai không bị chi phí
   mỗi vòng ăn mất.

3. **$t=4/L$ không phân kỳ mà còn là bước cố định tốt nhất.** Mốc $2/L$ chỉ là điều kiện *đủ*, tính
   từ cận xấu nhất $\sigma'(z)\le1/4$. Ở đây tỉ lệ dương chỉ 1,2%, nên phần lớn $\sigma(z_i)$ nằm gần 0
   và $\sigma'(z_i)=\sigma(1-\sigma)\ll1/4$: tại nghiệm, độ cong thật lớn nhất là 0,392, nhỏ hơn
   $L=6{,}25$ tới 16 lần (số liệu ở 2.4). Cũng vì vậy GD backtracking,
   vốn được phép gấp đôi bước mỗi vòng, tự tìm ra bước lớn hơn $1/L$ nhiều lần và hội tụ, trong khi
   GD bước cố định thì không.

4. **Đánh đổi của $\beta$ trong backtracking.** $\beta$ nhỏ co bước mạnh tay, ít lần thử nên mỗi vòng
   rẻ: 164 vòng, 12 giây. $\beta=0{,}9$ chọn bước sát ngưỡng hơn nhưng phải tính $F$ rất nhiều lần
   mỗi vòng: 394 vòng, 110 giây, chậm gần 9 lần theo thời gian.

5. **Cùng một ngưỡng gradient ≠ cùng sai số $F$.** Các setup hội tụ đều dừng ở
   $\|\nabla F\|\le10^{-3}\|\nabla F(w_0)\|$, nhưng $F-F^\star$ lúc dừng lại khác nhau: từ khoảng
   $2\cdot10^{-5}$ (Newton) đến $5{,}7\cdot10^{-5}$ (backtracking $\beta=0{,}1$). Lý do là với hàm
   lồi mạnh $F-F^\star\le\|\nabla F\|^2/(2\mu)$, và $\mu$ rất nhỏ nên cận này lỏng. sklearn
   (`tol=1e-4` mặc định) dừng ở $F-F^\star\approx9\cdot10^{-6}$, chặt hơn ngưỡng của mình. Với Newton,
   chỉ thêm 1–2 vòng là tới sai số máy, nên nếu cần $F$ thật chính xác thì có thể đặt `TOL` riêng
   chặt hơn cho Newton. Đổi lại, log-loss train/val của mọi setup đã hội tụ chỉ lệch nhau ở chữ số
   thứ 4, nên `TOL = 1e-3` là đủ cho mục đích dự đoán.

6. **SGD: lịch giảm bước lại tệ hơn bước cố định** trong 42 epoch. Robbins–Monro đảm bảo hội tụ
   *tiệm cận*, nhưng với $\kappa\sim6000$ thì bước $\alpha_0/k$ teo quá nhanh, trước khi kịp tới gần
   nghiệm. Không cấu hình SGD nào đạt điều kiện dừng: đúng như lý thuyết cho bước cố định (quả cầu
   nhiễu), và cho thấy ngân sách 42 epoch quá ít với bước giảm dần.

7. **Lasso: FISTA vừa hội tụ vừa thưa đúng nghĩa** (19/160 hệ số khác 0, khớp với sklearn liblinear
   cũng 19/160). Subgradient hiếm khi đưa hệ số về **đúng** 0 nên ra 154/160 hệ số khác 0, dù hàm
   mục tiêu giống hệt. $F$ của FISTA thấp hơn $F$ của liblinear vì liblinear phạt cả intercept.

8. **Kết luận MAP@7 không đổi**: mô hình logistic đã hội tụ (Newton), nên khoảng cách so với LightGBM
   nằm ở **mô hình tuyến tính**, không nằm ở thuật toán tối ưu.

---

## 5. Hạn chế & việc còn lại

- **Trường hợp biên**: hội tụ được suy ra từ "dừng trước khi hết ngân sách". Nếu một thuật toán đạt
  điều kiện dừng **đúng ở điểm cuối cùng** của ngân sách thì sẽ bị ghi nhầm là *chưa hội tụ*. Không
  lần chạy nào ở trên rơi vào trường hợp này.
- **SGD**: thời gian *không* tính phép kiểm tra $\|\nabla F\|$ toàn bộ mỗi epoch (đồng hồ dừng). Nếu
  tính vào thì thời gian SGD sẽ tăng lên khoảng gấp đôi.
- **$F^\star$ của Lasso** lấy từ chính các lần chạy (xem 2.5), chưa có lời giải tham chiếu độc lập.
- **Hình trong `overleaf/hinh/` chưa được cập nhật.** Lần chạy kiểm thử lưu hình vào thư mục tạm để
  không ghi đè. Chạy lại notebook trong VS Code sẽ ghi đè các PDF cùng tên, với trục tung mới là
  $F-F^\star$.
- **[overleaf/bao_cao.tex](overleaf/bao_cao.tex) vẫn mô tả tiêu chí cũ** ($k_\varepsilon$, 1%, 250 vòng) ở
  các dòng 121, 186–197, 231–235. Cần sửa lại theo bảng 3.2: ví dụ "Newton hội tụ sau 5 vòng" giờ là
  6 vòng theo điều kiện dừng, FISTA là 231 vòng thay vì 99, và Nesterov/backtracking là 172/164 vòng.

---

## 6. Chạy lại

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib scikit-learn lightgbm ipykernel
```

Cần `train_ver2.csv` (Kaggle, ~2,3 GB) ở thư mục gốc. Mở `bai_lam.ipynb`, chọn kernel `.venv` rồi
*Run All*: toàn bộ notebook chạy khoảng 20 phút trên máy này. Các tham số `TOL`, `VONG`, `VONG_NT`,
`EPOCH` nằm cả trong ô **CẤU HÌNH CHẠY**.
