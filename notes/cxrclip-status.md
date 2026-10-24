# Pipeline một đường CXR-CLIP — bảng trạng thái từng bước

> Một bước một lúc; dừng sau mỗi bước để người dùng review. `⏳` chưa tới · `🔄` đang làm · `✅` xong · `⛔` chặn.
> Nhánh `feat/cxrclip-image-only` tách từ `main` `90cd62a`. Chỉ commit khi người dùng gọi `/commit`.
> Đọc trước: `WORKLOG.md` (của `main`; quyết định mới sẽ ghi ở bước 5), rồi file này.

**Đích:** câu hỏi → parser (gpt-5-mini, rule dự phòng) → câu mẫu `render(include, exclude)` → text encoder → cosine trên
Qdrant, lọc view → top-k. Bỏ lọc nhãn và mode `label`/`hybrid`/`image` khỏi API. Giữ trích câu report làm bằng chứng
(sau xếp hạng, không ảnh hưởng thứ hạng). Giữ `label_filter` (chấm điểm), `probe_pool.py`, `build-probe-pool`,
`patient_split`. BiomedCLIP vẫn là một candidate encoder.

| # | Bước | Trạng thái | Kết quả |
|---|---|---|---|
| 1 | Index CXR-CLIP + sổ đăng ký build `data/indexes/builds/<build_id>/manifest.json` | ✅ đã duyệt | adapter từ `b71e587`; `selected: cxrclip`; hai build đăng ký: cxrclip 7/8 PASS (`image_probe` FAIL: probe là của BiomedCLIP, đi cùng mode `image` ở bước 2), biomedclip 8/8 PASS; manifest cũ của nhánh V1.1 không đổi (sha256); 940 passed, ruff sạch; integration 3/4 (probe live FAIL như trên, PASS với `BUILD_ID` BiomedCLIP). Sau review: mọi lệnh encode text nạp encoder qua `index.build_encoder` (từ chối revision khác build); `--build-id`/`--encoder` dùng chung, mâu thuẫn thì từ chối; `score-corpus --encoder biomedclip` ra lại 0,599; `.gitignore` chỉ bỏ `notes/*.log`; D38 trong WORKLOG; 952 passed |
| 2 | Rút pipeline/API về một đường; chốt hành vi: không khái niệm, chỉ phủ định, UNCERTAIN, ≥ 3 ý | ⏳ | |
| 3 | Port `template_bench` (R10, S12, P@k, hit@k, HNRR) + phần chấm mode ảnh của `retrieval_eval` từ `feat/label-aware-image-retrieval` | ⏳ | |
| 4 | Đo zero-shot CXR-CLIP và BiomedCLIP trên cùng pipeline (`--encoder`) | ⏳ | |
| 5 | WORKLOG: quyết định bị đảo D11, D23, D29, một phần D30; hash theo candidate (hẹp hơn D37) | ⏳ | |
| 6 | Mang code fine-tune sang | ⏳ | |

### Hành vi của bước 2 (người dùng duyệt 2026-10-24; ghi vào WORKLOG ở bước 5)

| Trường hợp | Hành vi |
|---|---|
| Không có khái niệm | Encode `semantic_query` tiếng Anh của parser, kèm warning; `semantic_query` rỗng (câu VI không khái niệm) → rỗng, kèm warning |
| Chỉ có phủ định | Câu mẫu `"no pleural effusion"`, không xử lý riêng; warning ghi rõ *mô hình hiện tại dễ trả về ảnh có bệnh này*. No Finding không phủ định được → bỏ, kèm warning |
| UNCERTAIN | Coi như PRESENT trong câu mẫu, kèm warning |
| ≥ 3 ý | Đủ mọi ý theo thứ tự chuẩn, kèm warning "trên 2 ràng buộc"; bench báo riêng nhóm này |
| No Finding được include | Câu mẫu dùng `normal chest`, không dùng "no finding"; áp cho mọi nơi qua **một** hàm render, hàm đó cũng quyết định thứ tự chuẩn |
| Một khái niệm vừa include vừa exclude | Báo lỗi, hoặc trả rỗng kèm warning; không bao giờ dựng "A and no A" |

Nhắc: giữ trích câu report làm bằng chứng (chỉ hiển thị sau xếp hạng, response ghi rõ không ảnh hưởng thứ hạng; ảnh
không có report → trống); giữ lọc view; không xóa `label_filter`, `probe_pool.py`, `build-probe-pool`, `patient_split`.

**TIẾP THEO:** bước 2.
