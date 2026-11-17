# Kịch bản demo

Sáu câu theo thứ tự trình bày, đều lấy từ `docs/v2-6-smoke.txt` (không lấy từ benchmark). Chạy `make demo`, mở
http://localhost:3000. Ảnh chụp tương ứng: `cd web && pnpm screenshots` ghi `1-phu-dinh.png` … `7-ngoai-pham-vi-so-sanh.png` và
`screenshots.json` vào thư mục này (không commit, xem `README.md`). Chạy thử lần cuối: 2026-11-17 (sau khi sửa dịch câu không dấu và thêm từ chối câu ngoài phạm vi, D79), KG
`kg_32e035ed37b0`.

Mỗi câu, mở trang trước khi nói: mỗi lượt diễn giải mất 3–5 s. Lượt diễn giải đầu tiên sau vài phút API rảnh đôi khi
lỗi kết nối (WORKLOG T5); hỏi lại câu đó là được.

| # | Màn | Câu | Điểm cần chỉ ra |
|---|---|---|---|
| 1 | So sánh | `no pneumothorax and no pleural effusion` | B1 và B0 trả cùng 10 ảnh (câu mẫu của B1 trùng nguyên văn câu hỏi), và chính B1 cảnh báo rằng câu chỉ gồm phủ định thì model hay trả ảnh *có* dấu hiệu đó. B2 không trùng ảnh nào với hai hệ kia; mỗi thẻ ghi "không tràn khí màng phổi 94%", tức KG chấm phủ định bằng 1 − p. Thang điểm ghi ở đầu mỗi cột: chỉ so thứ hạng, không so điểm giữa các cột |
| 2 | Tìm kiếm, rồi bấm ảnh hạng 1 | `suy tim` | Bệnh không có cột ảnh riêng: KG suy từ dấu hiệu (tim to 61%, phù phổi 30%, tràn dịch màng phổi 9%), kèm cảnh báo "không phải chẩn đoán". Trong khung chi tiết: điểm trộn = z(KG) + đóng góp CLIP; đường đi "Cardiomegaly —gợi ý→ Heart failure · trọng số 61%" (đưa chuột lên trọng số: tỉ lệ cùng xuất hiện trong report). Câu không dấu: phần cosine trộn đọc bản dịch "heart failure" (khung hiểu câu hỏi ghi rõ). Nếu diễn giải là "một phần" thì guard đã bỏ một câu so sánh mà facts không xác nhận |
| 3 | Tìm kiếm | `tràn khí màng phổi bên phải` | Câu tiếng Việt chấm thẳng bằng cột tràn khí màng phổi. Hệ thống nói rõ "bên hoặc mức độ không dùng để xếp hạng", không âm thầm bỏ chữ "bên phải" |
| 4 | Tìm kiếm | `phim có nốt phổi ở thùy trên` | Nốt phổi nằm ngoài 14 nhãn CheXpert, nhưng có cột riêng trong ma trận mở rộng. Vị trí "thùy trên" được báo là không dùng để xếp hạng. Đường trộn thấy rõ ở đây: hạng 1 có độ khớp nốt phổi thấp hơn hạng 2 nhưng đóng góp CLIP +0,53 (hạng 2 là −0,01), và diễn giải nói đúng "cao nhất là ảnh hạng 2 với 63%" (guard giữ câu này vì facts xác nhận) |
| 5 | Tìm kiếm | `tổn thương ở góc sườn hoành` | Vùng giải phẫu, không phải dấu hiệu: KG lấy các dấu hiệu thường gặp ở góc sườn hoành (mờ phổi 100%, tràn dịch màng phổi 31%, trọng số theo tỉ lệ gặp ở vùng), kèm cảnh báo khớp theo vùng và không xét bên |
| 6 | Tìm kiếm | `tràn dịch màng phổi và sỏi thận` | "sỏi thận" được tô đỏ là không nhận ra (nằm ngoài KG ngực), bị bỏ kèm cảnh báo; phần còn lại vẫn được chấm. Diễn giải cũng nói rõ sỏi thận không được tính |

Câu dự phòng khi còn thời gian:

| Màn | Câu | Điểm cần chỉ ra |
|---|---|---|
| Tìm kiếm, rồi So sánh | `ảnh chó mèo` | Câu ngoài phạm vi: B2 không trả ảnh nào, không viết diễn giải, chỉ báo "Hệ thống chỉ tìm ảnh X-quang ngực" kèm lý do của parser và 3 câu gợi ý (bấm vào là chạy). Trên màn so sánh, B1 và B0 vẫn trả 10 phim cho câu này: đó là chỗ B2 khác hẳn hai đường cũ |
| Tìm kiếm | `sỏi thận` | Câu trong phạm vi mà không nhận ra khái niệm nào: trang chuyển sang xếp theo độ giống văn bản như B1 và không có diễn giải. Khác với "ảnh chó mèo": câu này không bị từ chối |
| Tìm kiếm | `heart failure with small pleural effusions` | Bệnh cộng dấu hiệu, có phần kiến thức [K#] (đưa chuột lên [K#] để xem mục được dẫn) |
