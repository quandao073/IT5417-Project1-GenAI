# Thiết kế đánh giá

> Ba trục, nguồn ground truth, metric và cách diễn giải; xem §18 của
> [kế hoạch](ke-hoach-du-an-nho-cxr-semantic-retrieval-v1.md) và
> `configs/evaluation.yaml`.

## Ba trục và tính độc lập của chúng

| Trục | Ground truth | Vai trò |
|---|---|---|
| 1. Retrieval quality | CheXbert labels | P@5/10, nDCG@10, MRR |
| 2. Constraint compliance | CheXpert/CheXbert labels | exclude violation, constraint satisfaction |
| 3. Human judged | 200 study có nhãn bác sĩ | Precision@10, Cohen's kappa |

**Trục 1 và trục 2 cùng suy ra từ report nên tương quan với nhau.** Hybrid thắng
trục 2 *theo thiết kế*, vì nó lọc trên đúng nguồn đó — phải báo cáo như một tính
chất của hệ thống, không phải một kết quả retrieval. **Trục 3 là trục độc lập
duy nhất** và là trục có trọng lượng khi bảo vệ.

## Corpus không bị chia; query mới bị chia

Toàn bộ 25.000 study nằm trong một index và đều searchable, đúng như lúc dùng
thật. Không có tham số nào được học từ corpus nên không có leakage ở mức corpus.
Thứ *được* fit bằng tay là trọng số fusion, prompt và synonym — nên **query** mới
là thứ phải hold out: 150 validation để tuning, 250 test chạy đúng một lần ở
Phase 7, và hai tập không chia sẻ patient.

200 study `valid/` (234 ảnh) là ground truth người duy nhất; chúng được ghim vào tập test
và không bao giờ dùng để tuning.

## Vector mode áp ràng buộc gì

`vector` là baseline mà `hybrid` phải vượt qua, nên nó **không** áp ràng buộc
nhãn. Nó **có** áp bộ lọc view AP/PA, vì view là một metadata facet mà người dùng
kỳ vọng hoạt động ở mọi mode, không phải ràng buộc đang được đo. Tính công bằng
của con số nDCG chủ đạo phụ thuộc vào phân biệt này.

## Relevance graded và tập câu của Phase 5

Với relevance nhị phân (study thỏa ràng buộc của gold), label-only và hybrid cùng có p@10 = 1,0,
nên nó không trả lời được câu hỏi thứ hạng có ích hay không. Phase 5 thêm bậc cho nDCG@10:
`grade = 1 + trung bình theo concept bị ràng buộc của số nguồn nhãn khẳng định đúng trạng thái yêu
cầu`, trong đó exclude đếm `ABSENT` để "không có" nói rõ xếp trên im lặng. Đây là **proxy suy từ
report**, cùng nguồn với trục 1 và 2, và thiên về study có findings (tối đa 4 nguồn so với 3); nó
không thay trục 3.

Phase 5 đo trên 200 câu của benchmark parser (`benchmarks/parser_queries_v1.jsonl`: dev 40 câu để
tune fusion, heldout 160 câu), không phải tập 150/250 ở trên, vì tập đó chưa có. Phase 7 mở rộng
lên tập 400 câu và thêm human review. Số đo: [retrieval-evaluation.md](retrieval-evaluation.md).

## Khi hybrid không thắng

Báo cáo error analysis trung thực theo từng nhóm query, thay vì đổi test set
hoặc đổi metric.
