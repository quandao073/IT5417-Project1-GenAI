# CXR Semantic Retrieval

[![ci](https://github.com/quandao073/IT5417-Project1-GenAI/actions/workflows/ci.yml/badge.svg)](https://github.com/quandao073/IT5417-Project1-GenAI/actions/workflows/ci.yml)

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

- **Parser.** LLM (`claude-haiku-5-5`, output ràng buộc bằng tool có JSON schema theo 14 khái niệm CheXpert);
  thiếu key hoặc LLM lỗi thì dùng rule parser có từ điển song ngữ.
- **Câu mẫu.** Một hàm duy nhất dựng câu từ các ràng buộc: mỗi khái niệm một cụm cố định,
  `no X` cho bệnh bị loại, `normal chest` cho No Finding, thứ tự cố định (nên "A and B" và
  "B and A" là một câu). Câu không có khái niệm nào thì dùng câu tự do của parser. Câu chỉ có
  phủ định, có ý "có thể", hoặc có trên hai ràng buộc vẫn được tìm, kèm cảnh báo trong response.
- **Encoder.** Mặc định là CXR-CLIP SwinT bản fine-tune FT6 (`cxrclip_ft6_lr1e-04`): checkpoint
  MIMIC-CXR của Kakao Brain, tinh chỉnh trên bệnh nhân ngoài corpus. Bản zero-shot gốc (`cxrclip`) và
  BiomedCLIP là encoder thay thế; mỗi lần embed tạo một *build* riêng
  (`<ngày>_chexpert25k_<encoder>_v1`), và server phục vụ build chọn bằng `BUILD_ID`.
- **Bằng chứng.** Mỗi kết quả kèm câu nguyên văn trong report có nhắc tới bệnh được hỏi, gắn
  polarity (`affirmed`, `negated`, `uncertain`). Câu trích được thêm sau khi xếp hạng và không
  làm đổi thứ hạng.
- **Đánh giá.** Đáp án là nhãn 14 bệnh của từng ca (CheXbert trích từ report, nhãn bác sĩ cho 200
  ca `valid/`). Một ảnh là đúng khi nhãn của nó thỏa ràng buộc của câu hỏi. Số chính là nDCG@10 nhị
  phân, báo cáo thêm P@10, MRR và tỉ lệ kết quả mang bệnh bị loại trừ. Benchmark gồm 200 câu
  EN/VI chia dev và heldout.

## Đang xây: V2 với Knowledge Graph

V2 thêm một Knowledge Graph tri thức y khoa dựng từ UMLS và RadGraph, làm trung gian giữa câu hỏi và điểm ảnh của
CXR-CLIP. Thiết kế ở `docs/v2-kg-design.md`, tiến độ ở `docs/v2-kg-plan.md`. Đã có các bước dựng đầu vào, KG trên
Neo4j và đường truy vấn qua KG (`POST /api/v1/search/kg`); đánh giá chính thức chưa có:

```bash
# Cần license UMLS của chính bạn. Giải nén Metathesaurus Level 0 Subset 2026AA vào data/umls/raw/
# (đường dẫn khác thì đặt UMLS_ZIP, UMLS_META_DIR trong .env)
uv run cxr kg-filter-umls          # data/umls/2026AA/{mrconso,mrsty,mrrel,mrsab}.parquet và manifest.json
uv run cxr kg-pool-vocab           # từ vựng RadGraph-XL trên pool và đường cong độ phủ, ghi data/kg/vocab/
uv run cxr kg-chexpert-candidates  # ứng viên CUI cho 14 nhãn để duyệt tay; bản đã chốt: configs/kg/chexpert_cuis.yaml
uv run cxr build-image-scores --encoder cxrclip_ft6_lr1e-04 \
  --corpus-build-id 20261024_chexpert25k_cxrclip_ft6_lr1e-04_v1 \
  --pool-build-id 20261024_poolval9k_cxrclip_ft6_lr1e-04_v1   # ma trận điểm 25.000 study × 14 nhãn
```

Ma trận điểm là cosine với câu mẫu của từng nhãn, hiệu chỉnh trên pool-val, ghi vào
`data/indexes/image_scores/<build_id>/`. Shard pool-val và trọng số FT6 sinh trên nhánh `feat/label-aware-image-retrieval`.

Nạp KG vào Neo4j và hỏi thử (cần `ANTHROPIC_API_KEY` trong `.env`; không có key thì chỉ còn rule parser, không diễn giải):

```bash
docker compose --profile kg up -d neo4j        # Neo4j Community, volume neo4j_data, chỉ mở trên 127.0.0.1
uv run cxr kg-quality                          # data/kg/graph/quality.json: AUROC lenient từng cột, cờ weak / borderline
uv run cxr kg-load-neo4j                       # nạp bảng KG và top-N PREDICTED mỗi khái niệm (kg.yaml: neo4j.top_n)
uv run cxr validate-indexes --kg --kg-matrix data/indexes/image_scores/<build_id>_ext1_probe1
uv run cxr kg-query "tràn dịch màng phổi, không có tim to"              # hoặc --file câu_hỏi.txt --out x.jsonl
uv run cxr kg-bench-scoring                    # chấm điểm + top-k trên ma trận giả 1 triệu ảnh x 37 cột
uv run python scripts/kg_label_shadows.py    # dạng của 14 nhãn trùng alias của khái niệm con đã scorable (kg.yaml: ambiguous_forms)
uv run cxr eval-parser --split dev --parsers llm --kg-prompt prompts/query_parser_v3.txt   # prompt parser KG, chỉ bộ dev
```

API đọc ma trận `_ext1_probe1` (cột probe, D72) và bảng KG của `data/kg/graph/` (compose mount sẵn). Mất Neo4j hay LLM thì `/search/kg` vẫn trả kết
quả theo 14 nhãn kèm cảnh báo; `/health` báo `neo4j`, `kg_build_id`, `matrix_version`.

Dữ liệu UMLS chỉ nằm trong `data/umls/` (gitignored), không bao giờ vào git hay Docker image. Nguồn UMLS dùng: tên tiếng
Anh từ các nguồn mức 0 (`SRL = 0`), quan hệ từ MSH, NCI, FMA, HPO (`configs/kg.yaml`).

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
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env # tùy chọn; không có thì dùng rule parser
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

Trọng số FT6 của encoder mặc định (`runs/FT6_lr1e-04/model.pt`, sha256 ghim ở
`image_encoder.candidates.cxrclip_ft6_lr1e-04.revision`) không được phát hành: file sinh ra từ checkpoint trên bằng code
fine-tune của nhánh `feat/label-aware-image-retrieval`. Thiếu file này thì phục vụ bản zero-shot:
`BUILD_ID=20261023_chexpert25k_cxrclip_v1`.

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
| `POST /api/v1/search` với `"system": "B0"` | B0 của đánh giá: bản dịch EN của câu hỏi, không parser; `parsed_query` là `null` |
| `POST /api/v1/search/kg` | Tìm qua KG (B2); NDJSON: sự kiện `results` (mỗi ảnh kèm xác suất từng ý và từng cột) rồi `explanation` |
| `GET /api/v1/studies/{study_id}/report` | Report của một study, chỉ để đối chiếu; trả 403 trừ khi `configs/demo.yaml: show_report: true` |
| `POST /api/v1/query/parse` | Chỉ parse câu hỏi |
| `GET /api/v1/images/{image_id}` | Ảnh JPEG |
| `GET /api/v1/health` | Build, encoder, trạng thái Qdrant, chỉ mục report và LLM |

`BUILD_ID=<build_id>` chọn build khác để phục vụ; không đặt thì server dùng build duy nhất của
encoder mặc định. Thử parser riêng: `uv run cxr parse "tim to nhưng không có tràn dịch"`.

## Chạy demo

Giao diện web (`web/`, Next.js) cho câu hỏi EN/VI: kết quả hiện ngay, diễn giải tới sau; khung "Hệ thống hiểu câu hỏi"
(cụm → khái niệm → cách mở rộng → cột và trọng số, cảnh báo); chi tiết một ảnh (điểm từng cột, đường đi KG, phiên bản);
màn so sánh B1 · B2 · B0 cùng thứ hạng với `cxr eval-kg`.

1. Cần: Docker Desktop đang chạy; Node ≥ 20 và `pnpm` (`corepack enable`, hoặc `npm i -g pnpm` với Node 25 không còn
   corepack); chỉ mục đã dựng (mục *Dựng chỉ mục*), KG ở `data/kg/graph/` và ma trận điểm; `ANTHROPIC_API_KEY` trong
   `.env` (không có thì chạy rule parser, không diễn giải).
2. Lần đầu: `cd web && pnpm install`, và nạp KG vào Neo4j: `docker compose --profile kg up -d neo4j`, rồi
   `uv run cxr kg-load-neo4j`.
3. Mỗi lần: `make demo` (hoặc `./scripts/demo.sh`). Lệnh này bật Qdrant và Neo4j, chạy API ở `127.0.0.1:8000` (lần đầu
   khoảng 1 phút để nạp encoder, SapBERT, ma trận), chờ `/health` sẵn sàng rồi chạy `pnpm dev`.
4. Mở http://localhost:3000. Thanh trên cùng phải ghi "API sẵn sàng · KG kg_… · Neo4j ok · LLM ok". Neo4j
   `unreachable` thì KG chỉ chấm 14 nhãn: chạy lại bước 2.
5. Ctrl+C dừng API và web; `docker compose --profile kg stop` dừng container.

Nút "Xem report gốc (chỉ để đối chiếu)" chỉ hiện khi `configs/demo.yaml` có `show_report: true` (khởi động lại API).
Ảnh chụp màn hình: `cd web && pnpm screenshots` khi demo đang chạy (dùng Microsoft Edge có sẵn; ảnh nằm ở `docs/demo/`,
không commit, xem `docs/demo/README.md`). Kiểm web: `cd web && pnpm test && pnpm typecheck`.

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

CI (`.github/workflows/ci.yml`) chạy `uv sync --all-extras --locked`, `ruff` và `pytest -q` (bỏ các test cần mô
hình, Qdrant, Neo4j, LLM hay dữ liệu theo marker sẵn có), cùng `pnpm typecheck` và `pnpm test` của `web/`, mỗi lần push và
mỗi PR; khoảng 1,5 phút.

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
