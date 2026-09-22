# Thiết kế đánh giá

> Ba trục, nguồn ground truth, metric và cách diễn giải; xem §18 của
> [kế hoạch](ke-hoach-du-an-nho-cxr-semantic-retrieval-v1.md) và
> `configs/evaluation.yaml`.

## Ba trục và tính độc lập của chúng

| Trục | Ground truth | Vai trò |
|---|---|---|
| 1. Retrieval quality | CheXbert labels | P@5/10, nDCG@10, MRR |
| 2. Constraint compliance | CheXpert/CheXbert labels | exclude violation, constraint satisfaction |
| 3. Human judged | 234 study có nhãn bác sĩ | Precision@10, Cohen's kappa |

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

234 study `valid/` là ground truth người duy nhất; chúng được ghim vào tập test
và không bao giờ dùng để tuning.

## Vector mode áp ràng buộc gì

`vector` là baseline mà `hybrid` phải vượt qua, nên nó **không** áp ràng buộc
nhãn. Nó **có** áp bộ lọc view AP/PA, vì view là một metadata facet mà người dùng
kỳ vọng hoạt động ở mọi mode, không phải ràng buộc đang được đo. Tính công bằng
của con số nDCG chủ đạo phụ thuộc vào phân biệt này.

## Khi hybrid không thắng

Báo cáo error analysis trung thực theo từng nhóm query, thay vì đổi test set
hoặc đổi metric.
