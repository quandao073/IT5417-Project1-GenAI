# Ảnh chụp màn hình demo

`cd web && pnpm screenshots` (khi `make demo` đang chạy) chụp trang kết quả của 6 câu, trang chi tiết một ảnh và màn so
sánh của 2 câu vào thư mục này, kèm `screenshots.json` tóm tắt từng trang (số ảnh, cảnh báo, trạng thái diễn giải).

**Ảnh chụp không commit** (`.gitignore`: `docs/demo/*`), vì:

- Ảnh X-quang là dữ liệu CheXpert. Trang dataset của Stanford AIMI chỉ nói dữ liệu dùng cho nghiên cứu phi thương mại
  "subject to the applicable data use terms"; bản thỏa thuận nằm sau đăng nhập (Redivis), chưa đọc được và chưa có bản
  lưu trong `docs/data-governance/` (kiểm 2026-11-17). Chưa xác minh được quyền đăng lại ảnh trên repo public, nên giữ
  trên máy. Trong báo cáo hay slide thì kiểm thỏa thuận đã ký trước và ghi nguồn Stanford AIMI.
- Trang hiện tên khái niệm UMLS (tên tiếng Anh của MRCONSO); repo public không chứa tập con UMLS (CLAUDE.md §5, T38).
