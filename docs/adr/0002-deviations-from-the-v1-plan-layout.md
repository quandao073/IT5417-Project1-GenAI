# 2. Ba lệch so với §15 của kế hoạch V1

Ngày: 2026-09-22

## Trạng thái

Accepted

## Bối cảnh

Kế hoạch V1 §7.5 yêu cầu chia corpus theo patient thành 18000/3500/3500. §15 mô tả
cấu trúc `src/cxr_search/` + `apps/{api,web}/`, và liệt kê `compose.cpu.yaml`.

## Quyết định

**1. Không chia corpus.** Toàn bộ 25.000 study nằm trong một index. Hold-out ở
mức query: 150 validation / 250 test, patient-disjoint. Hệ thống không train gì,
nên chia corpus chỉ thu nhỏ không gian tìm kiếm và làm Recall/nDCG khó so sánh
giữa các mode, trong khi không ngăn được thứ leakage thực sự tồn tại — việc fit
trọng số fusion, prompt và synonym bằng tay. Hệ quả: canonical tables không có
cột `project_split`; việc gán patient vào validation/test nằm trong
`evaluation/queries.py`.

**2. Một package duy nhất tên `src/`, không có tầng `cxr_search/` lẫn `apps/`.**
§15 mô tả `src/cxr_search/` + `apps/{api,web}/`. Repo đã mang tên dự án nên lặp
lại nó thành tên package là thừa, và hai tầng thư mục đó không mang thông tin gì.
`src` vẫn là *một package thật* (`packages = ["src"]`, import `src.data.audit`,
console script `src.cli.main:main`) chứ không phải source root trần, vì cả `cxr`
lẫn hai Docker image (api, worker) đều `pip install` nó và cần đúng một namespace.
Phương án bỏ hẳn namespace — để `data/`, `query/`, `cli/` thành top-level — bị
loại: `data/` ở gốc repo đang là thư mục dataset 12 GB, và việc cài những tên
chung đó vào site-packages của hai container là không nên. `api/` nằm trong
package vì `api.Dockerfile` vốn đã cài package rồi trỏ uvicorn vào module của nó
— tách ra sẽ tạo hai đường nạp mã trong cùng một image.

Đánh đổi đã biết: `src` là một tên top-level rất chung trong site-packages. Vô
hại ở đây vì mỗi venv/image chỉ cài đúng một project; sẽ thành xung đột nếu sau
này venv cài thêm một project khác cũng dùng cách đặt tên này.

**3. Giữ `compose.gpu.yaml` thay vì `compose.cpu.yaml`.** §15 ngụ ý base compose
là GPU. Nhưng runtime — api, web, qdrant — không bao giờ cần GPU; chỉ worker lúc
embed mới cần. Base CPU-safe + một override GPU cho worker là đúng chiều phụ
thuộc và giữ được yêu cầu "đường CPU phải chạy được" của §6.3.

## Hệ quả

`docs/evaluation-design.md` phải nêu rõ cách đọc "corpus không bị chia", để câu
hỏi phản biện "các anh index cả test set à?" có câu trả lời trên giấy tờ.
