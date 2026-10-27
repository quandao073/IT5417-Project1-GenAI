# Giới hạn

> Cập nhật liên tục. Bắt buộc có trước khi bảo vệ. Phần đầu là giới hạn của hệ thống đang chạy (B1); phần sau là giới
> hạn đã biết của dữ liệu và nguồn mà V2 dùng. V2 đang xây (`docs/v2-kg-plan.md`); giới hạn của các tính năng V2 sẽ ghi
> khi chúng có.

## Hệ thống đang chạy (B1)

- Ảnh dùng bản CheXpert-v1.0-small đã downsample: không dùng được cho chi tiết mảnh.
- Mỗi study một ảnh frontal đại diện: report có thể mô tả ảnh khác trong cùng study.
- Parser chỉ biết 14 nhãn CheXpert. Khái niệm khác chỉ được tìm khi câu không có nhãn nào (câu tự do của parser, kèm
  cảnh báo); đi cùng một nhãn thì bị bỏ. Câu tiếng Việt không có nhãn nào thì không có gì để tìm.
- Model ảnh zero-shot hiểu phủ định kém: câu "không có tràn khí màng phổi" cho 4/10 ảnh đầu mang chính tràn khí, trong
  khi nền của corpus là 14,4% (thử ngày 2026-10-24, `docs/measured-facts.md`). Câu chỉ có phủ định được tìm kèm cảnh
  báo.
- Câu trích report chỉ để hiển thị, không đổi thứ hạng. Đó là câu **đầu tiên** khớp, không phải câu "tốt nhất";
  polarity do rule parser gán trên từng câu, nên hedge kiểu "question … versus …" có thể bị đọc là `affirmed`.
- Đáp án của đánh giá là nhãn suy từ report: nhãn CheXbert do máy trích, không phải ground truth lâm sàng; chỉ 200
  study `valid/` (234 ảnh) có nhãn bác sĩ. Nhãn CheXbert theo section `findings` chỉ có ở study có section đó (25,7%
  corpus); labeler chạy trên chuỗi rỗng luôn ra `No Finding = present`, nên những dòng đó bị loại.
- Không phải thiết bị y tế, không dùng để chẩn đoán. Điểm là độ khớp truy vấn, không phải xác suất bệnh.

## Dữ liệu và nguồn của V2

- RadGraph-XL là output của model, chưa ai kiểm: chỉ dùng làm trọng số của KG và nhãn bạc khi đánh giá, không coi là
  sự thật. Nhãn bạc và nhãn CheXbert cùng đọc một report, nên các trục đánh giá dựa trên chúng còn tương quan; người
  chấm một mẫu là trục độc lập.
- UMLS không có tiếng Việt (bản 2026AA Level 0 có 24 ngôn ngữ, không có `VIE`): mọi alias tiếng Việt do dự án soạn.
- Level 0 Subset không có SNOMED CT, nguồn có phân cấp finding và giải phẫu tốt nhất. Phân cấp lấy từ MSH, NCI, FMA,
  HPO, và mỗi nguồn ghi quan hệ một kiểu (WORKLOG T39).
- KG không có triệu chứng lâm sàng; không xét bên, vị trí chi tiết hay mức độ của tổn thương.
