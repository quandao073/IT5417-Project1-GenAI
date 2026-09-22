# Giới hạn của POC

> Cập nhật liên tục. Bắt buộc có trước khi bảo vệ.

- Ảnh dùng bản CheXpert-v1.0-small đã downsample: không dùng được cho chi tiết mảnh.
- Nhãn CheXbert sinh tự động từ report, không phải ground truth lâm sàng. Chỉ 234
  study trong `valid/` có nhãn do bác sĩ gán.
- Trục 1 và trục 2 của đánh giá cùng suy ra từ report nên còn tương quan; human
  review tồn tại để bù phần đó.
- Một ảnh frontal đại diện cho mỗi study: report có thể mô tả ảnh khác trong cùng study.
- `search_text` ưu tiên impression. Findings chỉ có ở 26,6% số dòng nên không bao
  giờ được đặt làm điều kiện bắt buộc.
- Report là tiếng Anh. Truy vấn tiếng Việt đi tới report search và evidence
  extraction thông qua 14 tên nhãn chuẩn, không phải bằng dịch tự do.
- RadGraph-XL có sẵn trong `data/raw/chexpert-plus/radgraph_xl/` nhưng **không**
  được ingest trong V1 — nó thuộc phần Knowledge Graph hoãn sang V2. File không
  thiếu, chỉ là chưa dùng.
- Không phải thiết bị y tế, không dùng để chẩn đoán.
