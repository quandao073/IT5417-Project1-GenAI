# Giới hạn của POC

> Cập nhật liên tục. Bắt buộc có trước khi bảo vệ.

- Ảnh dùng bản CheXpert-v1.0-small đã downsample: không dùng được cho chi tiết mảnh.
- Nhãn CheXbert sinh tự động từ report, không phải ground truth lâm sàng. Chỉ 200
  study trong `valid/` (234 ảnh) có nhãn do bác sĩ gán.
- Nhãn CheXbert suy từ section `findings` chỉ dùng được cho 25,6% study thực sự
  có section đó. Với study không có, labeler chạy trên chuỗi rỗng và luôn xuất
  `No Finding = present`; những assertion đó bị loại khỏi `labels.parquet`.
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
- Relevance graded của đánh giá retrieval là proxy suy từ report: bậc là số nguồn nhãn
  (CheXpert/CheXbert, đều đọc cùng report) khẳng định trạng thái yêu cầu. Study có findings có tối
  đa 4 nguồn, không có findings tối đa 3, nên proxy thiên về study có findings. Report leg cũng đọc
  chính report đó, nên phần hybrid hơn label-only trên nDCG (0,607 so với 0,531) không phải bằng
  chứng độc lập; human review (Phase 7) mới là trục độc lập.
- Từ vựng report (`concepts.yaml:report`) phủ không đều. Probe thiết kế: recall 0,27 (Pleural Other)
  đến 0,76 (Support Devices) trên study PRESENT. Với từ vựng cuối, trên 1.000 report thật chỉ 62,5%
  cặp (study, concept PRESENT) có câu evidence, từ 4,4% (Enlarged Cardiomediastinum) đến 77,9%
  (Support Devices). Concept mà report ít dùng đúng từ thì report leg ít tín hiệu.
- Câu tiếng Việt chỉ có exclude không có vector leg: `semantic_query` rỗng nên không có gì để
  encode, và cũng không có term tiếng Anh cho report leg. Mode `vector` trả rỗng; `hybrid` trả theo
  thứ tự cố định của mode `label`, kèm warning.
- Evidence là câu **đầu tiên** khớp, không phải câu "tốt nhất": mỗi concept lấy câu đầu tiên có
  polarity mong muốn, không có thì câu đầu tiên nhắc tới concept. Polarity do rule parser gán trên
  từng câu, nên hedge kiểu "question … versus …" có thể bị đọc là `affirmed`.
- Không phải thiết bị y tế, không dùng để chẩn đoán.
