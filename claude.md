1. chọn feature theo kiểu df['a'] = x, viết ít hàm thôi,
2. ĐÃ BIẾT kết quả là 38 feeatures => tối thiểu hóa code phần này
3. Define cho t hai làm lasso và ridge đơn giản để t tự code các thuạt toán =)) 
4. Code đến đây đã, t sẽ tự code tiếp,/order m code, sau đó m không cần code overengineering nnay
5. Thêm các thuật toán proximal và accelerated gd
6. Trace tại sao SGD lại cho ra đồ thị mịn thế
7. Mở rộng bài toán với toàn bộ tất cả các sản phẩm (tương ứng với từng đó model) => Tính MAP7 và kẻ bảng so sánh accuracy các kiểu
8. Chọn best theo thời gian tới F − F* ≤ ε (chi tiết + kết quả chạy thử: tien_trinh.md, mục 5–9). Kế hoạch lần chạy lớn (ngân sách 10 tiếng):
   - Đã xong: chạy thử 11 setup (mục 6c notebook, đã có output). ε chính = 1e-5; báo cáo ε ∈ {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}
   - F*: Ridge = 0.043630121211 (Newton = sklearn); Lasso = 0.046538162389 (FISTA 600 s, chính xác ~1e-9)
   - Trần mỗi setup: max_giay = 600; max_vong_lap không giới hạn
   - Dừng sớm khi F − F* ≤ 1e-8 (đã làm: tham số F_dung của 7 thuật toán, 6c truyền F* + 1e-8)
   - Lưới: giữ 46 setup cũ + Nesterov/FISTA/GD t ∈ {2.5, 3, 4, 6}/L, SGD batch {1024, 4096, 16384} × lịch {cố định, 1/√k} × α0 {0.2, 1, 5} (mỗi tổ hợp batch × lịch phải dò α0, tien_trinh.md mục 17), GD backtracking thêm hệ số Armijo c
   - Ước tính ~7 giờ cho ô 6c bản đầy đủ (71 setup; SGD 18 lần chạy hết trần 600 s ≈ 3 h). Đã thêm vào 6c: BUOC_THEM t ∈ {2.5,3,4,6}/L cho GD/Nesterov/FISTA, C_ARMIJO c ∈ {1e-4,1e-2,0.1,0.3} × β ∈ {0.1,0.5} (tien_trinh.md mục 18)
   - Báo cáo: bảng best theo từng ε + hình F − F* theo thời gian có đường ngang ε = 1e-5; cập nhật cả bai_lam.ipynb và tien_trinh.md
   - Đã xong: biểu đồ so sánh mới (tien_trinh.md, mục 10–12). Đã sửa: hình SGD vs GD dùng log_x. Lưu ý: thời gian đo lệch tới ~50% giữa các lần chạy → lần chạy lớn báo cáo thêm số vòng/lượt dữ liệu tới ε hoặc đo lặp
   - Đã sửa theo note_nhung_gi_can_sua.md (tien_trinh.md mục 13–14), chưa chạy lại notebook. Còn: Nesterov restart (cân nhắc). Đã xong: hình tổng kết (nay là mục 6e) + 3 hình cặp đôi (mục 6d) vẽ lại từ KQ_6C, best theo t_ε ở ε = 1e-6 (tien_trinh.md mục 24)
   - Đã chạy toàn bộ với CHAY_DAY_DU = False (tien_trinh.md mục 15). Ô 6c lưu/đọc kết quả: kq_6c_dai_dien.pkl / kq_6c_day_du.pkl, CHAY_LAI_6C = True để chạy lại (mục 16)
9. Màu biểu đồ: chỉ dùng danh sách MAU (8 màu, thứ tự cố định), đường thứ i lấy màu thứ i; không dải gradient, không chọn màu theo tên thuật toán/tham số; vượt 8 đường thì thêm kiểu nét/marker
10. Kết quả là vài con số chính xác / nhiều thuộc tính / đường chồng nhau → in BẢNG bằng in_bang() (ô cấu hình); biểu đồ chỉ khi hình dạng đường cong là thông điệp (tien_trinh.md mục 19)
