# 3. Bỏ hẳn disk guard

Ngày: 2026-09-22

## Trạng thái

Accepted

## Bối cảnh

Kế hoạch V1 §9.2 mô tả một `disk_guard` với `min_free_gb: 40`,
`project_ceiling_gb: 55`, `staging_ceiling_gb: 8`, `model_cache_ceiling_gb: 8`,
yêu cầu pipeline phải dừng trước khi vượt bất kỳ ngưỡng nào. §17 (task Phase 1)
và §19 (danh sách unit test) đều nhắc lại.

Module này từng tồn tại. Nó bị xóa có chủ đích ở commit `9d3dca3`
(`chore: drop the disk guard command and switch code comments to english`), và
bị gỡ khỏi kế hoạch cũ ở `b4b55e5`. Tài liệu kế hoạch V1 mới vô tình đưa nó trở lại.

## Quyết định

**Không hồi sinh disk guard.**

Số đo trên máy thực tế:

- 191 GB trống, trong khi §9.2 giả định "máy còn khoảng 100 GB".
- Raw data 12,1 GB đã tải xong; V1 chỉ thêm dưới 1 GB
  (canonical Parquet + SQLite FTS + Qdrant payload).
- `images.serve_originals: true` nghĩa là **không stage nào ghi file ảnh**.
  Không có bản WebP, không có thư mục processed, không có staging lớn.

Ngân sách của §9.2 được viết cho một kịch bản không xảy ra: nó giả định một
pipeline có materialize ảnh đã xử lý. Kịch bản đó đã bị loại khi phép đo cho thấy
mọi ảnh CheXpert-small đều là mode L, dưới 512 px, không EXIF — nên resize,
grayscale và strip metadata đều là no-op.

Một guard không bao giờ được kích hoạt là mã chết phải bảo trì, và tệ hơn, nó
tạo cảm giác an toàn sai ở những chỗ nó không thực sự canh.

## Hệ quả

- `.env.example` không còn `DISK_MIN_FREE_GB` / `DISK_PROJECT_CEILING_GB`
  (đã gỡ ở `a98669d`).
- §9.2, task "Viết disk guard" của Phase 1, và mục disk guard trong danh sách
  unit test §19 **không được thực hiện**.
- Nếu sau này có một stage thật sự ghi nhiều — ví dụ đổi ý và materialize
  thumbnail cho toàn corpus — thì thêm một check free-space cục bộ ngay tại
  stage đó, chứ không dựng lại một module guard toàn cục.
