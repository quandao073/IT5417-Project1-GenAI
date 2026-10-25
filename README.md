# CXR Semantic Retrieval

Tìm ảnh X-quang ngực bằng câu hỏi tự nhiên, tiếng Anh hoặc tiếng Việt, chỉ dựa vào ảnh.

## Bài toán

Đầu vào là một câu như "tim to và tràn dịch màng phổi, phim PA" hoặc "pulmonary edema without
pleural effusion". Đầu ra là các ảnh X-quang ngực khớp nhất trong kho 25.000 ca, xếp hạng theo độ
giống giữa câu hỏi và nội dung ảnh. Câu hỏi có thể đòi một bệnh, nhiều bệnh cùng lúc, loại trừ một
bệnh, hoặc lọc theo tư thế chụp (AP/PA). Thứ hạng không dùng nhãn hay report của ca, nên cách làm
áp dụng được cho ảnh chưa có report.

## Cách tiếp cận

```
câu hỏi (EN/VI)
  → parser: các bệnh được hỏi (include), bị loại (exclude), tư thế chụp
  → câu mẫu, vd "pulmonary edema and no pleural effusion"
  → text encoder của model CLIP chuyên X-quang ngực
  → cosine với vector ảnh trong Qdrant, chỉ lọc tư thế chụp → top-k
  → trích câu trong report của từng ca, chỉ để hiển thị
```

- **Parser.** LLM (`gpt-5-mini`, output ràng buộc bằng JSON schema theo 14 khái niệm CheXpert);
  thiếu key hoặc LLM lỗi thì dùng rule parser có từ điển song ngữ.
- **Câu mẫu.** Một hàm duy nhất dựng câu từ các ràng buộc: mỗi khái niệm một cụm cố định,
  `no X` cho bệnh bị loại, `normal chest` cho No Finding, thứ tự cố định (nên "A and B" và
  "B and A" là một câu). Câu không có khái niệm nào thì dùng câu tự do của parser. Câu chỉ có
  phủ định, có ý "có thể", hoặc có trên hai ràng buộc vẫn được tìm, kèm cảnh báo trong response.
- **Encoder.** Mặc định là CXR-CLIP SwinT (Kakao Brain, huấn luyện trên MIMIC-CXR). BiomedCLIP là
  encoder thay thế; mỗi lần embed tạo một *build* riêng (`<ngày>_chexpert25k_<encoder>_v1`), và
  server phục vụ build chọn bằng `BUILD_ID`.
- **Bằng chứng.** Mỗi kết quả kèm câu nguyên văn trong report có nhắc tới bệnh được hỏi, gắn
  polarity (`affirmed`, `negated`, `uncertain`). Câu trích được thêm sau khi xếp hạng và không
  làm đổi thứ hạng.
- **Đánh giá.** Đáp án là nhãn 14 bệnh của từng ca (CheXbert trích từ report, nhãn bác sĩ cho 200
  ca `valid/`). Một ảnh là đúng khi nhãn của nó thỏa ràng buộc của câu hỏi. Số chính là nDCG@10 nhị
  phân, báo cáo thêm P@10, MRR và tỉ lệ kết quả mang bệnh bị loại trừ. Benchmark gồm 200 câu
  EN/VI chia dev và heldout.

## Đang mở rộng: tri thức y khoa

Đề tài thêm một Knowledge Graph tri thức y khoa (giải phẫu phổi, bệnh lý, dấu hiệu hình ảnh, triệu chứng) làm trung
gian giữa câu hỏi và ảnh: LLM ánh xạ câu hỏi vào khái niệm của KG và diễn giải kết quả bằng tri thức tra được, ví dụ
nối một ca tràn dịch màng phổi với dấu hiệu và triệu chứng của bệnh này. Phần này chưa được xây; phạm vi và dữ liệu
còn thiếu nằm ở `docs/scope-and-non-goals.md`.

## Dữ liệu

| Phần | Nguồn | Đặt tại |
|---|---|---|
| Ảnh | CheXpert-v1.0-small | `data/raw/CheXpert-v1.0-small/` (`train/`, `valid/`, `train.csv`, `valid.csv`) |
| Report | CheXpert Plus | `data/raw/chexpert-plus/df_chexpert_plus_240401.csv` |
| Nhãn CheXbert | CheXpert Plus | `data/raw/chexpert-plus/labels/` |

Ảnh và report ghép theo `path_to_image`. Corpus gồm 25.000 study, mỗi study một ảnh frontal AP/PA
có impression. Cả hai bộ dữ liệu đều phải đăng ký và chấp nhận điều khoản; repo không chứa dữ liệu nào.

## Cài đặt

Yêu cầu: [`uv`](https://docs.astral.sh/uv/) (tự cài Python 3.11–3.12), Docker cho Qdrant, khoảng
25 GB trống ngoài dữ liệu gốc (12,1 GB). GPU không bắt buộc: mọi bước chạy được trên CPU. Trên Windows,
dùng Git Bash.

```bash
uv sync --all-extras
echo "OPENAI_API_KEY=sk-..." > .env          # tùy chọn; không có thì dùng rule parser
```

Khi chạy trực tiếp trên máy, `.env` chỉ cần dòng trên; mọi đường dẫn và địa chỉ đã có giá trị mặc
định. `.env.example` là cấu hình cho container.

Trọng số CXR-CLIP không có trên Hugging Face. Tải một lần rồi chuyển sang file mà code đọc; URL nằm ở
`configs/models.yaml` (`image_encoder.candidates.cxrclip.url`):

```bash
mkdir -p data/cache/models/cxrclip
curl -L -o data/cache/models/cxrclip/swint_m.tar "<url trong configs/models.yaml>"
uv run --with omegaconf python -c "from pathlib import Path; \
from src.embeddings.cxrclip import convert_checkpoint as c; \
print(c(Path('data/cache/models/cxrclip/swint_m.tar'), Path('data/cache/models/cxrclip/swint_m.pt')))"
```

File đầu ra phải đúng tên `swint_m.pt`: sha256 của nó được ghim trong `configs/models.yaml`, nên
một file khác sẽ bị từ chối khi nạp. BiomedCLIP và cấu hình Swin/BERT tự tải từ Hugging Face theo
revision đã ghim.

## Dựng chỉ mục

```bash
uv run cxr audit                  # đo tỉ lệ ghép ảnh–report
uv run cxr build-corpus           # bảng Parquet của corpus 25.000 study
uv run cxr embed                  # vector ảnh ra shard file, chạy trên CPU, resume được
docker compose up -d qdrant
uv run cxr index                  # nạp shard vào Qdrant, ghi data/indexes/builds/<build_id>/manifest.json
uv run cxr build-report-index     # chỉ mục report để trích bằng chứng
uv run cxr validate-indexes       # kiểm manifest, shard, Qdrant và chỉ mục report khớp nhau
```

`cxr embed` dùng encoder `image_encoder.selected` trong `configs/models.yaml`. Muốn có build cho encoder
khác (vd BiomedCLIP), đặt tạm `selected` và `revision` sang encoder đó khi chạy `cxr embed`, rồi
`cxr index --build-id <build_id>`. Các build cùng tồn tại; `--build-id` hoặc `--encoder` chọn build
cho `validate-indexes`, `score-corpus` và `eval-retrieval`.

## Chạy

```bash
docker compose up -d qdrant
uv run uvicorn src.api.main:app --port 8000
```

Mở http://127.0.0.1:8000/docs để thử trực tiếp, hoặc:

```bash
curl -s http://127.0.0.1:8000/api/v1/search -H "Content-Type: application/json" \
     -d '{"query": "pulmonary edema without pleural effusion", "top_k": 5}'
```

| Endpoint | Chức năng |
|---|---|
| `POST /api/v1/search` | Nhận `query`, `top_k` (1–50), `views` (`AP`/`PA`, tùy chọn). Trả `parsed_query`, `query_text` (câu encoder đã đọc), `results` (ảnh, điểm cosine, `report_evidence`) và `warnings` |
| `POST /api/v1/query/parse` | Chỉ parse câu hỏi |
| `GET /api/v1/images/{image_id}` | Ảnh JPEG |
| `GET /api/v1/health` | Build, encoder, trạng thái Qdrant, chỉ mục report và LLM |

`BUILD_ID=<build_id>` chọn build khác để phục vụ; không đặt thì server dùng build duy nhất của
encoder mặc định. Thử parser riêng: `uv run cxr parse "tim to nhưng không có tràn dịch"`.

## Đánh giá

```bash
uv run cxr eval-parser --split dev --out artifacts/metrics/parser_eval_dev.json
uv run cxr eval-retrieval --split dev --out artifacts/metrics/retrieval_eval_dev.json
uv run cxr eval-retrieval --split dev --encoder biomedclip --out artifacts/metrics/retrieval_eval_dev_biomedclip.json
```

`eval-retrieval` chấm ba cấu hình parser (`gold`, `rule`, `auto`), tách theo nhóm câu (đơn, ghép,
phủ định, tư thế chụp, tiếng Việt), và chấm nhóm phủ định dưới hai cách hiểu nhãn: bệnh bị loại chỉ
cần không được ghi là có, hoặc phải được ghi rõ là không có. `--encoder` cần build của encoder đó đã
được index.

## Kiểm thử

```bash
uv run pytest -q                  # unit test
uv run pytest -m model            # adapter thật trên trọng số đã tải; chạy khi đổi encoder hoặc configs/models.yaml
uv run pytest -m integration      # cần Qdrant
uv run ruff check src tests
```

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/data` | Ghép ảnh–report, corpus, nhãn |
| `src/embeddings` | Adapter encoder, embed, index Qdrant, sổ đăng ký build |
| `src/query` | Parser LLM và rule parser |
| `src/retrieval` | Câu mẫu, tìm kiếm, trích bằng chứng, chỉ mục report |
| `src/api` | FastAPI |
| `src/evaluation` | Benchmark parser và retrieval |
| `src/cli` | Lệnh `cxr` |
| `configs/` | Dữ liệu, model, khái niệm, retrieval, đánh giá |
| `benchmarks/` | 200 câu truy vấn có đáp án |
| `data/`, `artifacts/` | Không commit |

## Lưu ý

- Ảnh CheXpert-small đã giảm độ phân giải, không phù hợp với tổn thương nhỏ.
- Nhãn dùng để chấm do máy trích từ report, không phải bác sĩ đọc ảnh; chỉ 200 ca `valid/` có
  nhãn bác sĩ.
- Trọng số CXR-CLIP dùng giấy phép CC BY-NC 4.0, chỉ dùng cho mục đích phi thương mại.
