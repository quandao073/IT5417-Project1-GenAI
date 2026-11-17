# Ảnh chụp màn hình demo

`cd web && pnpm screenshots` (khi `make demo` đang chạy) chụp sáu câu của `kich-ban-demo.md` theo thứ tự (màn so
sánh cho câu 1, trang chi tiết cho câu 2) vào thư mục này, kèm `screenshots.json` ghi lại hệ thống đã trả gì cho từng
câu (ý, cảnh báo, hai thành phần điểm, diễn giải và câu bị bỏ).

**Ảnh chụp không commit** (`.gitignore`: `docs/demo/*` trừ `README.md` và `kich-ban-demo.md`), vì:

- Ảnh X-quang là dữ liệu CheXpert. Trang dataset của Stanford AIMI chỉ nói dữ liệu dùng cho nghiên cứu phi thương mại
  "subject to the applicable data use terms"; bản thỏa thuận nằm sau đăng nhập (Redivis), chưa đọc được và chưa có bản
  lưu trong `docs/data-governance/` (kiểm 2026-11-17). Chưa xác minh được quyền đăng lại ảnh trên repo public, nên giữ
  trên máy. Trong báo cáo hay slide thì kiểm thỏa thuận đã ký trước và ghi nguồn Stanford AIMI.
- Trang hiện tên khái niệm UMLS (tên tiếng Anh của MRCONSO); repo public không chứa tập con UMLS (CLAUDE.md §5, T38).
