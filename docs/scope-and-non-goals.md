# Scope và non-goals

> Phạm vi hiện hành theo thiết kế V2 `docs/v2-kg-design.md` (WORKLOG D41, D45–D50); tiến độ ở `docs/v2-kg-plan.md`.
> [Kế hoạch V1](ke-hoach-du-an-nho-cxr-semantic-retrieval-v1.md) là tài liệu lịch sử: chỗ nào lệch thì `WORKLOG.md` và
> file này đúng.

## Đề tài

Mô tả đề tài, nguyên văn:

> Đề tài nhằm xây dựng một hệ thống truy vấn ảnh y tế cho ảnh X-quang phổi, cho phép người dùng tìm kiếm hình ảnh
> dựa trên mô tả ngôn ngữ tự nhiên thay vì chỉ dựa vào nhãn hoặc từ khóa. Tri thức y khoa về giải phẫu phổi, bệnh lý
> và các dấu hiệu hình ảnh được mô hình hóa thành một Knowledge Graph, đóng vai trò trung gian giữa truy vấn ngôn ngữ
> và đặc trưng hình ảnh. LLM được sử dụng để phân tích truy vấn, ánh xạ sang các khái niệm trong KG và hỗ trợ diễn giải
> kết quả truy vấn. Sinh viên cần trang bị kiến thức về computer vision cho ảnh y tế, image embedding và semantic
> retrieval, xây dựng Knowledge Graph y khoa (UMLS, RadGraph), và LLM/RAG pipeline. Đề tài thiên về ứng dụng nhưng có
> giá trị thực tiễn cao trong hỗ trợ chẩn đoán và quản lý dữ liệu y tế.

Người dùng nói thêm (2026-10-25): khi bàn về một bệnh, ví dụ tràn dịch màng phổi, hệ thống phải nối ảnh với các dấu
hiệu hình ảnh của bệnh đó. Triệu chứng lâm sàng nằm ngoài phạm vi (2026-10-27, xem *Không thực hiện*).

## Hiện trạng

**Đang chạy (B1, baseline của V2):** câu hỏi EN/VI → parser LLM theo 14 nhãn CheXpert → câu mẫu → text encoder
CXR-CLIP → cosine trên Qdrant, lọc tư thế chụp → top-k, kèm câu trích report chỉ để hiển thị (D38–D40). Thứ hạng chỉ
dựa trên ảnh (D39).

**Đang xây (V2):** Knowledge Graph tri thức y khoa từ UMLS và RadGraph, làm trung gian giữa câu hỏi và điểm ảnh của
CXR-CLIP. Mới xong bước lọc UMLS thành Parquet (V2-1). KG, ma trận điểm, các tool và phần diễn giải chưa có.

## Phạm vi V2

| Hạng mục (theo thiết kế V2) | Trạng thái |
|---|---|
| Corpus 25.000 study; pool 88.853 study ngoài corpus | Có |
| Truy vấn ảnh bằng câu tự nhiên EN/VI (B1) | Có |
| UMLS 2026AA Level 0 Subset lọc thành Parquet | Có (V2-1) |
| Từ vựng lõi từ RadGraph trên pool; 14 nhãn CheXpert ánh xạ sang CUI | Chưa có (V2-2) |
| KG: khái niệm giải phẫu, bệnh, finding, thiết bị từ UMLS; trọng số `SUGGESTS`, `TYPICALLY_AT` tính trên pool | Chưa có (V2-3, V2-4) |
| Ma trận điểm CXR-CLIP đã hiệu chỉnh trên pool-val | Chưa có (V2-5) |
| LLM phân tích câu hỏi, chọn khái niệm trong ứng viên của KG, diễn giải kết quả từ đường đi trong KG | Chưa có (V2-6) |
| Benchmark mới, đánh giá B0–B3 | Chưa có (V2-7, V2-8) |
| Web UI | Chưa có; chờ `pnpm` (WORKLOG T5) |

## Ràng buộc

- Report không vào đường truy vấn (D45); thứ hạng chỉ từ ảnh (D39); LLM không quyết định thứ hạng (D47).
- KG chỉ chứa tri thức, không có tầng dữ kiện theo study (D46); mọi trọng số của KG lấy từ pool, không lấy từ corpus.
- Dữ liệu UMLS không vào git, không vào Docker image (D49, T38).

## Không thực hiện

- Chẩn đoán hay đề xuất điều trị. Điểm là độ khớp truy vấn, không phải xác suất bệnh.
- Triệu chứng lâm sàng (ho, sốt…): ảnh không có tín hiệu về chúng, và suy từ triệu chứng sang bệnh là suy luận chẩn
  đoán.
- Đọc report hay nhãn của ca lúc truy vấn.
- Vị trí, bên, mức độ của tổn thương, cho tới khi model ảnh học được; khoanh vùng trên ảnh.
- Nguồn tri thức ngoài UMLS và RadGraph (RadLex để sau); SNOMED CT chỉ khi đã khai báo MLDS (D48).
- Sinh report từ ảnh; tích hợp PACS/HIS/FHIR; triển khai cloud.

## Dữ liệu và nguồn tri thức

| Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| Ảnh CheXpert-v1.0-small | Có | Corpus 25.000 study; pool 88.853 study |
| Report CheXpert Plus | Có | Chỉ dùng offline: trọng số KG và đánh giá (D45) |
| Nhãn 14 bệnh | Có | CheXpert/CheXbert trích từ report; 200 ca `valid/` có nhãn bác sĩ |
| RadGraph-XL | Có | Output của model; phủ 25.000/25.000 ca corpus, độ phủ trên pool đo ở V2-2; nối với CSV bằng text (WORKLOG T29) |
| UMLS 2026AA Level 0 Subset | Có | MD5 khớp; lọc ở `data/umls/2026AA/`; không có tiếng Việt (`docs/measured-facts.md`, *UMLS 2026AA*) |
| CXR-CLIP fine-tune FT6 | Có trên đĩa | Candidate và lệnh dựng ma trận điểm chỉ có trên nhánh `feat/label-aware-image-retrieval` (V2-5) |
| Ground truth cho V2 | Chưa có | Benchmark mới và pooling (V2-7) |

## Câu hỏi nghiên cứu

Theo thiết kế V2: *KG có cải thiện truy vấn chứa khái niệm mà model ảnh không học trực tiếp không (từ đồng nghĩa, khái
niệm hẹp hơn, rộng hơn, vị trí giải phẫu, bệnh không thấy trực tiếp trên phim), so với pipeline hiện tại (B1)?* Các câu
hỏi phụ ở `docs/v2-kg-design.md`.
