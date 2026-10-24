# Pipeline một đường CXR-CLIP — bảng trạng thái từng bước

> Một bước một lúc; dừng sau mỗi bước để người dùng review. `⏳` chưa tới · `🔄` đang làm · `✅` xong · `⛔` chặn.
> Nhánh `feat/cxrclip-image-only` tách từ `main` `90cd62a`. Chỉ commit khi người dùng gọi `/commit`.
> Đọc trước: `WORKLOG.md` (*Trạng thái*, D38–D40, T36), rồi file này.

**Đích:** câu hỏi → parser (gpt-5-mini, rule dự phòng) → câu mẫu `render(include, exclude)` → text encoder → cosine trên
Qdrant, lọc view → top-k. Bỏ lọc nhãn và mode `label`/`hybrid`/`image` khỏi API. Giữ trích câu report làm bằng chứng
(sau xếp hạng, không ảnh hưởng thứ hạng). Giữ `label_filter` (chấm điểm), `probe_pool.py`, `build-probe-pool`,
`patient_split`. BiomedCLIP vẫn là một candidate encoder.

| # | Bước | Trạng thái | Kết quả |
|---|---|---|---|
| 1 | Index CXR-CLIP + sổ đăng ký build `data/indexes/builds/<build_id>/manifest.json` | ✅ đã duyệt | adapter từ `b71e587`; `selected: cxrclip`; hai build đăng ký: cxrclip 7/8 PASS (`image_probe` FAIL: probe là của BiomedCLIP, đi cùng mode `image` ở bước 2), biomedclip 8/8 PASS; manifest cũ của nhánh V1.1 không đổi (sha256); 940 passed, ruff sạch; integration 3/4 (probe live FAIL như trên, PASS với `BUILD_ID` BiomedCLIP). Sau review: mọi lệnh encode text nạp encoder qua `index.build_encoder` (từ chối revision khác build); `--build-id`/`--encoder` dùng chung, mâu thuẫn thì từ chối; `score-corpus --encoder biomedclip` ra lại 0,599; `.gitignore` chỉ bỏ `notes/*.log`; D38 trong WORKLOG; 952 passed |
| 2 | Rút pipeline/API về một đường; chốt hành vi: không khái niệm, chỉ phủ định, UNCERTAIN, ≥ 3 ý | ✅ chờ review + commit | `templates.render` (no X, normal chest, thứ tự chuẩn, từ chối A and no A); pipeline một đường, warning cho các biên; request bỏ `mode` (422 nói rõ), response bỏ nhãn, API không nạp `labels.parquet`; gỡ hybrid / mode `image` / FTS / mode `label` / `train-probe`; `eval-retrieval` một đường (số chính nDCG@10 nhị phân). 865 passed, integration 3, model 6, ruff sạch; `validate-indexes` 7/7 PASS cả hai build; smoke 7 câu qua API thật (`notes/cxrclip_step2_smoke.log`); dev CXR-CLIP nDCG@10 nhị phân 0,482, trùng `image_joint` V1.1 ở 36/38 câu. WORKLOG: D39, D40, T36. Sửa luôn một model test bước 1 để lọt (khối `preprocessing` dời vào candidate) |
| 3 | Port benchmark câu mẫu: `template_bench` (R10, S12, P@k, hit@k, HNRR) + phần chấm mode ảnh của `retrieval_eval` từ `feat/label-aware-image-retrieval`; câu mẫu qua `templates.render` | ⏳ | |
| 4 | Đo zero-shot CXR-CLIP và BiomedCLIP trên cùng pipeline (`--encoder`): dev, câu mẫu trên corpus, pool-val. **Không heldout** | ⏳ | |
| 5 | WORKLOG: ghi số đo của bước 4 (các quyết định bị đảo đã ghi ở bước 1–2: D38–D40) | ⏳ | |
| 6 | Mang code fine-tune sang, rồi fine-tune; heldout chạy một lần ở M2 cho zero-shot và bản fine-tune đã chọn | ⏳ | |

### Hành vi của bước 2 (người dùng duyệt 2026-10-24; ghi ở WORKLOG D39)

| Trường hợp | Hành vi |
|---|---|
| Không có khái niệm | Encode `semantic_query` tiếng Anh của parser, kèm warning; `semantic_query` rỗng (câu VI không khái niệm) → rỗng, kèm warning |
| Chỉ có phủ định | Câu mẫu `"no pleural effusion"`, không xử lý riêng; warning ghi rõ *mô hình hiện tại dễ trả về ảnh có bệnh này*. No Finding không phủ định được → bỏ, kèm warning |
| UNCERTAIN | Coi như PRESENT trong câu mẫu, kèm warning |
| ≥ 3 ý | Đủ mọi ý theo thứ tự chuẩn, kèm warning "trên 2 ràng buộc"; bench báo riêng nhóm này |
| No Finding được include | Câu mẫu dùng `normal chest`, không dùng "no finding"; áp cho mọi nơi qua **một** hàm render, hàm đó cũng quyết định thứ tự chuẩn |
| Một khái niệm vừa include vừa exclude | Không bao giờ dựng "A and no A": `SearchQuery` từ chối parse như vậy (LLM thì retry rồi về rule parser; rule parser tự giải: "edema and no edema" → chỉ exclude), và `render` cũng từ chối |

Nhắc: giữ trích câu report làm bằng chứng (chỉ hiển thị sau xếp hạng, response ghi rõ không ảnh hưởng thứ hạng; ảnh
không có report → trống); giữ lọc view; không xóa `label_filter`, `probe_pool.py`, `build-probe-pool`, `patient_split`.

Phạm vi bước 2 (người dùng duyệt 2026-10-24):
- Response bỏ nhãn và `constraints_satisfied`; API không nạp `labels.parquet`.
- Request bỏ `mode`; còn gửi `mode` thì 422 nói rõ lý do, không âm thầm bỏ qua.
- Xóa `hybrid.py`, phần tìm FTS của `report_search` (giữ `report_index`, `fetch_texts`), `image_concepts.py`,
  `probe_eval.py`, lệnh `train-probe`. Giữ `probe.py`, `probe_pool.py`, `build-probe-pool`, `label_filter`,
  `build-report-index`.
- `eval-retrieval` rút về một đường, giữ `relevance_for`, hai policy, ba cấu hình parser, tách theo nhóm và ngôn ngữ.
  Metric chính: nDCG@10 nhị phân; graded chỉ là số phụ. Ngưỡng hybrid trong `configs/evaluation.yaml` gỡ hoặc đánh
  dấu hết hiệu lực. `bench-vector` bỏ `--mode`.

### Định hướng sau bước này (người dùng, 2026-10-24; S1–S4 là tên giai đoạn của người dùng)

- Encoder chọn qua `BUILD_ID` ở server, không qua request.
- **S3** thêm cách chấm: probe và decomposed quay lại dưới **một** lệnh `build-image-scores --method {decomposed, probe}`,
  ghi cùng một định dạng ma trận điểm.
- **S4** thêm `label_filter`; chỉ khi đó API mới nạp lại nhãn, và chỉ sau núm `label_filter`.
- **Bước 4 không chạy heldout.** Chỉ đo dev, benchmark câu mẫu trên corpus và pool-val. Heldout để dành cho **M2**,
  chạy cùng lúc cho zero-shot và bản fine-tune đã chọn.

## Danh sách kiểm trước mỗi commit

```bash
uv run pytest -q && uv run ruff check src tests
uv run pytest -m model          # BẮT BUỘC khi đổi configs/models.yaml hoặc src/embeddings/ (WORKLOG T37)
uv run pytest -m integration    # cần Qdrant Docker
uv run cxr validate-indexes && uv run cxr validate-indexes --encoder biomedclip
```

## Điểm dừng (2026-10-24 tối)

Dừng sau bước 2 theo yêu cầu. Người dùng review, gọi `/commit`, rồi merge vào `main`.

**TIẾP THEO:** bước 3 (benchmark câu mẫu) khi người dùng mở lại.
