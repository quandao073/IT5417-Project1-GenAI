# Train trên Colab Pro

Laptop chỉ dùng để đóng gói dữ liệu và nhận kết quả về. Phần train chạy trên GPU của Colab. Notebook dừng lại ở bất kỳ bước kiểm tra nào không đạt, nên CU không bị đốt vào một lượt chạy hỏng.

## 1. Đóng gói trên laptop

```bash
# Mốc zero-shot của laptop (đã có ở runs/zero_shot_val.json; chỉ chạy lại nếu checkpoint đổi)
uv run --project training python training/zero_shot_val.py --out runs/zero_shot_val.json
# Gói ≈ 4,7 GiB (5,0 GB) ở data/colab_bundle/cxr_v11 (dữ liệu là hard link nên không tốn thêm đĩa)
uv run --project training python training/colab/pack.py --out data/colab_bundle
```

Gói gồm:
- `code.zip`: `src/`, `training/`, `configs/`. Không có `.env` hay `.venv`. Có `training/colab/pylock.colab.toml`: URL và sha256 của từng file thư viện, xuất từ `training/uv.lock`.
- Cache ảnh pool, nhãn pool, checkpoint CXR-CLIP, và config Swin/BERT đã ghim.
- `zero_shot_val.json`.
- `manifest.json`: sha256 của từng file.

## 2. Tải lên Google Drive

**Đã tải gói lên rồi mà code thay đổi:** không cần tải lại 5 GB. Chỉ thay `code.zip` và `manifest.json` trong
`My Drive/cxr_v11` (ô 3 so sha256 của `code.zip` với `manifest.json`, nên phải thay cả hai), và tải lại notebook.
Thư mục `data/` không đổi.

- Kéo thả **cả thư mục `cxr_v11`** vào **My Drive**, giữ nguyên tên. Dùng drive.google.com hoặc Google Drive for desktop.
- **Không chia sẻ thư mục này.** Dữ liệu CheXpert và trọng số CXR-CLIP (CC BY-NC) chỉ được dùng riêng.
- Mở Colab, chọn *File → Upload notebook*, rồi tải lên `training/colab/train_colab.ipynb`.

## 3. Chạy notebook

Trước tiên chọn *Runtime → Change runtime type → **A100 GPU***. Nếu không có A100 thì chọn L4. Sau đó chạy lần lượt từng ô:

| Ô | Việc | Mất bao lâu | Không đạt thì |
|---|---|---|---|
| 1 | Gắn Drive, xem GPU | < 1 phút | Báo `BUNDLE MISSING`: kiểm tra tên thư mục |
| 2 | Giải nén code; dựng venv **Python 3.12** riêng bằng `uv`, cài đúng từng phiên bản của laptop (torch 2.4.1+cu124, transformers 4.44.2…). Python và thư viện có sẵn của Colab không được dùng | 2–4 phút | **Dừng**, gửi output |
| 3 | Chép dữ liệu sang đĩa máy Colab, **kiểm sha256 từng file** | 1–3 phút | Tải lại gói |
| 4 | **Kiểm môi trường:** P@5 zero-shot trên pool-val phải khớp laptop (trung bình 0,377, dung sai 0,01) | ≈ 1 phút | **Dừng**, gửi output |
| 5 | **Cổng kiểm tra:** chạy FT6 ngắn (20.000 ảnh × 2 epoch). Không được sụp (cosine giữa các caption < 0,8) và không được thấp hơn zero-shot quá 0,02. Lượt ngắn chỉ để bắt lỗi; có vượt zero-shot hay không là việc của các lượt đầy đủ ở ô 6. Ô này in số s/bước và ước tính số phút, số CU cho một lượt đầy đủ | ≈ 5–10 phút | **Dừng**, gửi output |
| 6 | Dò lr cho FT6 với {3e-5, 1e-4, 1e-5}, dữ liệu đầy đủ, tối đa 10 epoch, dừng sớm khi 3 epoch liền không tiến bộ | xem số in ở ô 5 | — |
| 7 | FT1 (đúng công thức bài báo) ở lr mà FT6 chọn: dòng đối chứng | 1 lượt | — |
| 8 | **Trả GPU** để không tốn thêm CU | — | — |

Ở ô 5, điền `CU_PER_HOUR` theo số hiện trong bảng *Resources* của Colab để có ước tính chi phí.

**Nếu mất kết nối:** chạy lại ô 1 → 4 vì máy Colab mới có đĩa trống (và chưa có venv), rồi chạy lại đúng ô train đang dở. Lượt đó tự chạy tiếp từ epoch cuối đã lưu trên Drive. Lượt nào đã có `model.pt` thì được bỏ qua.

## Chạy bản miễn phí qua đêm (T4, chưa mua CU)

Cùng gói, cùng notebook. Chỉ khác ở mấy điểm sau:

- **Chọn GPU T4.** T4 không có bf16 thật, nên code tự chuyển sang fp16 kèm GradScaler, và tắt gradient checkpointing (T4 có 14,7 GiB). Cả hai đường đã thử trên laptop: train fp16 không có bước lỗi; P@5 zero-shot ở fp16 chỉ lệch 1 ảnh top-5 so với mốc bf16, vẫn trong dung sai của ô 4.
- **Tốc độ T4 xấp xỉ laptop.** Ô 5 in số đo thật. Mỗi lượt đầy đủ ước tính khoảng 1,5–2,5 giờ, nên một đêm (8–10 giờ) chạy được khoảng **3 lượt dò lr của ô 6**. Ô 7 (FT1) để sau, khi đã có CU.
- **Bản miễn phí chỉ chạy khi tab trình duyệt còn mở.** Một phiên tối đa khoảng 12 giờ, và có thể bị ngắt sớm hơn. Hãy để laptop bật và tắt chế độ ngủ; laptop chỉ giữ tab nên không nóng. Bị ngắt thì sáng hôm sau chạy lại ô 1–4 và ô 6: các lượt xong được bỏ qua, lượt dở chạy tiếp từ epoch cuối đã lưu.
- **Có lúc Colab không cấp GPU** khi đã dùng nhiều. Khi đó chờ vài giờ rồi thử lại.
- **Drive cần trống khoảng 7 GB:** gói 5,0 GB, cộng khoảng 0,6 GB cho mỗi lượt xong (riêng `model.pt` đã 547 MB).
- Khi mua CU, chỉ cần đổi runtime sang A100 và chạy tiếp đúng notebook này. Các lượt đã xong trên T4 vẫn giữ nguyên.

## 4. Về CU

- Runtime còn kết nối là còn trừ CU, kể cả khi không chạy gì. Xong việc hoặc phải đi đâu lâu thì chạy ô 8, hoặc chọn *Runtime → Disconnect and delete runtime*.
- 100 CU được ước tính là đủ cho ô 5–7, và còn dư cho FT2–FT5 sau này. Con số chắc chắn là con số ô 5 in ra trên đúng GPU bạn được cấp. Nếu ước tính vượt ngân sách, dò lr trên 20.000 ảnh trước.

## 5. Lấy kết quả về laptop

Trong Drive, `cxr_v11/runs/<tên lượt>/` có `model.pt`, `model.sha256`, `metrics.json` và `train.log`. Tải 4 file đó về `runs/<tên lượt>/` của repo. Các thư mục `adapter_epoch*` và file `state_last.pt` chỉ dùng để chạy tiếp, không cần tải. Embed và đánh giá (B16) chạy trên laptop từ `model.pt`.

## Những gì bảo đảm kết quả đúng

1. **Đã sửa nguyên nhân gốc.** FT1 sụp vì hai lý do. Thứ nhất, đích chỉ lấy từ caption nên 33,8% cặp âm thật ra khớp ảnh. Thứ hai, `logit_scale` 24,16 làm mô hình gốc tệ hơn đoán đều, nên cách giảm loss rẻ nhất là ép mọi caption về một điểm (WORKLOG, mục *FT1 kẹt ở ln 128*). FT6 sửa cả hai: lấy đích từ nhãn đầy đủ của ảnh, và cố định scale ở 5. Đã thử ngắn trên laptop (20.000 ảnh × 2 epoch): không sụp, P@5 trung bình 0,377 → 0,386, phủ định 0,267 → 0,358 (WORKLOG, mục *Sửa B14*).
2. **Dữ liệu và trọng số đúng từng byte** nhờ sha256 ở ô 3.
3. **Colab chạy đúng bộ thư viện của laptop:** venv Python 3.12 cài từ lock của laptop ở ô 2, không dùng
   torch/transformers có sẵn của Colab. Ô 4 kiểm lại bằng P@5 zero-shot phải khớp laptop.
4. **Công thức học thật trên đúng GPU này** nhờ cổng kiểm tra ở ô 5.
5. **Mỗi epoch đều được kiểm:** P@5 và cosine giữa các caption trên pool-val. Checkpoint được giữ là epoch tốt nhất theo pool-val (D33), có dừng sớm, và bước nào ra loss không hữu hạn sẽ bị bỏ qua.
