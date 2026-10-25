# Scope và non-goals

> Phạm vi có hiệu lực từ 2026-10-25 (WORKLOG D41). Bản trước hoãn Knowledge Graph sang V2 (D28); bản này đưa KG
> vào đề tài. [Kế hoạch V1](ke-hoach-du-an-nho-cxr-semantic-retrieval-v1.md) viết trước thay đổi này: chỗ nào lệch
> thì `WORKLOG.md` và file này đúng.

## Đề tài

Mô tả đề tài, nguyên văn:

> Đề tài nhằm xây dựng một hệ thống truy vấn ảnh y tế cho ảnh X-quang phổi, cho phép người dùng tìm kiếm hình ảnh
> dựa trên mô tả ngôn ngữ tự nhiên thay vì chỉ dựa vào nhãn hoặc từ khóa. Tri thức y khoa về giải phẫu phổi, bệnh lý
> và các dấu hiệu hình ảnh được mô hình hóa thành một Knowledge Graph, đóng vai trò trung gian giữa truy vấn ngôn ngữ
> và đặc trưng hình ảnh. LLM được sử dụng để phân tích truy vấn, ánh xạ sang các khái niệm trong KG và hỗ trợ diễn giải
> kết quả truy vấn. Sinh viên cần trang bị kiến thức về computer vision cho ảnh y tế, image embedding và semantic
> retrieval, xây dựng Knowledge Graph y khoa (UMLS, RadGraph), và LLM/RAG pipeline. Đề tài thiên về ứng dụng nhưng có
> giá trị thực tiễn cao trong hỗ trợ chẩn đoán và quản lý dữ liệu y tế.

Người dùng nói thêm (2026-10-25): khi bàn về một bệnh, ví dụ tràn dịch màng phổi, hệ thống phải liên kết ảnh với các
dấu hiệu và triệu chứng của bệnh đó. Khi người dùng hỏi, hệ thống tra tri thức y khoa để hỗ trợ diễn giải kết quả.

## Hệ thống khi xong

```
câu hỏi (EN/VI)
  → LLM phân tích câu hỏi, ánh xạ sang khái niệm của KG (bệnh, dấu hiệu hình ảnh, giải phẫu, triệu chứng)
  → KG quy khái niệm về thứ encoder ảnh tìm được (vd "góc sườn hoành tù" → tràn dịch màng phổi)
  → câu mẫu → text encoder → cosine với vector ảnh trong Qdrant, lọc tư thế chụp → top-k      [đã có, D38–D39]
  → mỗi kết quả: tra KG (dấu hiệu, triệu chứng, vị trí của bệnh liên quan)
                 + câu trích report và fact RadGraph của chính ca đó
  → LLM diễn giải kết quả chỉ từ những gì đã tra được (RAG), kèm nguồn của từng ý
```

Ba ràng buộc giữ từ phần đã làm:

- **Thứ hạng chỉ từ ảnh** (D39): KG làm việc ở phía câu hỏi và ở phần diễn giải sau xếp hạng, không đọc nhãn hay report
  của ca để xếp hạng. Có nới ràng buộc này không là câu hỏi P16.
- **Câu đưa vào encoder do code dựng**, không do LLM viết (D39). LLM chỉ phân tích câu hỏi và viết phần diễn giải.
- **Triệu chứng là tri thức về bệnh**, không phải khẳng định về bệnh nhân trong ảnh: ảnh không cho biết bệnh nhân có
  khó thở hay không. Dữ liệu từng ca cũng hầu như không ghi triệu chứng (`docs/measured-facts.md`, *Dữ liệu cho KG*), nên triệu chứng
  chỉ nối với ảnh qua bệnh: ảnh → bệnh → triệu chứng trong KG.

## Bắt buộc

| Hạng mục | Hiện trạng |
|---|---|
| Corpus 25.000 study, ảnh kèm report | Đã có (Phase 1) |
| Image embedding và semantic retrieval từ câu tự nhiên | Đã có: CXR-CLIP, một đường tìm kiếm (D38–D39) |
| LLM phân tích câu hỏi EN/VI | Đã có cho 14 khái niệm CheXpert (D22); phải mở rộng sang khái niệm của KG |
| KG tri thức y khoa trong Neo4j, dựng từ UMLS và RadGraph: giải phẫu phổi, bệnh lý, dấu hiệu hình ảnh, triệu chứng của **mọi bệnh có trong corpus** | Chưa có; thiết kế chờ duyệt: `docs/superpowers/specs/2026-10-25-medical-knowledge-graph-design.md`; cần license UMLS (P14) |
| Nối ca với KG | Chưa có. Qua bệnh (14 khái niệm) và qua RadGraph-XL của report |
| Ánh xạ câu hỏi → khái niệm KG | Chưa có |
| Diễn giải kết quả bằng LLM trên tri thức tra từ KG và report (RAG), có trích nguồn | Chưa có |
| Đánh giá: retrieval, ánh xạ câu hỏi, diễn giải | Retrieval và parser đã có; phần KG chưa có (P18) |
| API, web UI, Docker Compose | API đã có; web UI chưa (WORKLOG T5) |

## Không thực hiện

- Tự động chẩn đoán hay đề xuất điều trị. Hệ thống là công cụ tìm kiếm và tra cứu, không phải thiết bị y tế.
- Bệnh không có trong corpus; nguồn tri thức ngoài UMLS và RadGraph.
- Sinh report từ ảnh.
- Tích hợp PACS/HIS/FHIR, triển khai cloud.

## Dữ liệu và nguồn tri thức

Kiểm ngày 2026-10-25. Số đo và cách đo nằm ở `docs/measured-facts.md`.

| Thành phần | Trên đĩa | Ghi chú |
|---|---|---|
| Ảnh: CheXpert-v1.0-small | Có | Corpus 25.000 study; pool 88.853 study ngoài corpus |
| Report: CheXpert Plus | Có | Impression 100% corpus, findings 25,7%, clinical history 78,2% (nhưng chỉ 1–4% ghi một triệu chứng) |
| Nhãn 14 bệnh | Có | CheXpert/CheXbert trích từ report; 200 ca `valid/` có nhãn bác sĩ |
| RadGraph | Có (RadGraph-XL) | Phủ 25.000/25.000 impression; entity giải phẫu/quan sát có present/absent/uncertain, quan hệ `located_at`/`modify`/`suggestive_of`. Không có mã UMLS, là output của model, phải nối với CSV bằng text (WORKLOG T29) |
| UMLS | **Không** | License cấp cho cá nhân, miễn phí, cần đăng ký tài khoản UTS (P14). Không được commit dữ liệu trích ra vì repo public (WORKLOG T38) |
| Quan hệ bệnh → dấu hiệu → triệu chứng | **Chưa biết** | UMLS có những quan hệ này tới đâu thì chỉ xem được khi có license. Report ít khi tả dấu hiệu (góc sườn hoành tù 3,3%), nên không trích được từ corpus. Chỗ UMLS thiếu, 14 nhãn CheXpert có một lõi nhỏ tuyển tay giữa các khái niệm UMLS |
| Ground truth cho phần KG | **Không** | Cần benchmark mới (P18) |
| LLM | Có | `gpt-5-mini` qua OpenAI (D22) |

**Kết luận:** đủ để bắt đầu tầng ca (ảnh, report, RadGraph, nhãn). Tầng tri thức chờ license UMLS; đánh giá phần KG
cần benchmark mới.

## Câu hỏi nghiên cứu

Đề xuất, chốt ở bước thiết kế: *KG tri thức y khoa làm trung gian có giúp (1) tìm đúng ảnh khi câu hỏi nói bằng dấu
hiệu, triệu chứng hay vị trí giải phẫu thay vì tên bệnh, và (2) diễn giải kết quả đúng với tri thức và report hơn LLM
không có KG hay không?*
