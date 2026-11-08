# Smoke V2-6 và rà đường cho Disease

Hai phần: bảng smoke 24 câu (lượt cuối) và tóm tắt lượt rà đường của các Disease lõi chưa chấm được. Số đo độ trễ ở
`docs/measured-facts.md`; quyết định ở `WORKLOG.md` (D64, D66–D69). Theo CLAUDE.md §5, khái niệm UMLS chỉ ghi bằng CUI
kèm mô tả tự viết; tên nhãn CheXpert và cụm từ report thì ghi nguyên.

## Smoke 24 câu

Câu hỏi: `docs/v2-6-smoke.txt`. Đây là 24 câu viết mới, không lấy từ heldout của parser, và không dùng lại cho V2-7. Kỳ
vọng được ghi trước lượt chạy đầu.

```bash
uv run cxr kg-query --file docs/v2-6-smoke.txt --out data/kg/smoke/v2-6i.jsonl
```

Có chín lượt, từ a tới i. Bảng dưới là **lượt i** (2026-11-08): sau khi lookup gộp ba bước (D68) và sau khi thêm 22 cạnh
`manual` từ lượt rà đường bên dưới. Chú thích cột "Kết quả": *thẳng* nghĩa là chấm thẳng cột của khái niệm; *rộng hơn*
là noisy-OR các con có cột; *suy từ dấu hiệu* là SUGGESTS ngược tới các cột; *vùng* là noisy-OR có trọng số của các dấu
hiệu `TYPICALLY_AT`; "loại" là phủ định.

| # | Câu | Kỳ vọng | Kết quả lượt i | Ghi chú |
|---|---|---|---|---|
| 1 | ảnh X-quang có khí phế thũng | cột emphysema | C0013990 (cột emphysema), thẳng | cụm không có alias exact, chỉ có hit SapBERT; chooser chọn |
| 2 | phim có nốt phổi ở thùy trên | cột nodule + vị trí | C0034079 (cột nodule), thẳng | vị trí đi cạnh dấu hiệu chỉ cảnh báo (D67) |
| 3 | is there water on the lung | Pleural Effusion | Pleural Effusion (UNCERTAIN) | — |
| 4 | tràn khí màng phổi bên phải | Pneumothorax, cảnh báo bên | Pneumothorax, cảnh báo bên | — |
| 5 | enlarged heart size with clear lungs | Cardiomegaly | Cardiomegaly, loại Lung Opacity ("clear lungs") | — |
| 6 | lobar pneumonia in the right lower lobe | hẹp hơn Pneumonia + vị trí | Pneumonia (yếu), cảnh báo bên và vị trí | "lobar pneumonia" không thành term ở lượt này |
| 7 | loculated pleural effusion | effusion | Pleural Effusion, "Not considered: loculated." | D69 |
| 8 | bệnh lý màng phổi | rộng hơn: noisy-OR con | C0032226 (nhóm bệnh màng phổi), rộng hơn, 3 cột | alias vi + cạnh manual (D67) |
| 9 | interstitial lung disease | bệnh → dấu hiệu | C0206062 (bệnh mô kẽ lan tỏa), suy từ 3 dấu hiệu mô kẽ | cạnh manual (D67) |
| 10 | suy tim | bệnh suy từ dấu hiệu | C0018801, suy từ Cardiomegaly, Edema, Pleural Effusion | — |
| 11 | heart failure with small pleural effusions | effusion + suy tim | Pleural Effusion + C0018801 suy từ dấu hiệu; "Not considered: small." | term dao động giữa các lượt (nhiễu parser) |
| 12 | abnormality at the lung apex | vùng đỉnh phổi | C0225703 (đỉnh phổi), vùng: tràn khí, sẹo, nốt | D68, D69 |
| 13 | tổn thương ở góc sườn hoành | vùng góc sườn hoành | C0230151 (góc sườn hoành), vùng: Lung Opacity, Pleural Effusion | hết bão hòa sau D68, D69 |
| 14 | no pneumothorax and no pleural effusion | loại hai nhãn | loại Pneumothorax, loại Pleural Effusion | — |
| 15 | không có nốt phổi nhưng có khí phế thũng | loại nodule, có emphysema | loại C0034079, có C0013990 | luật nhãn cha/con (D67) |
| 16 | phim PA, không tim to, có xẹp phổi | Atelectasis, loại Cardiomegaly, PA | đúng, cảnh báo Atelectasis yếu | — |
| 17 | atelectasis in the lower lobes | Atelectasis + vị trí | Atelectasis (yếu), vị trí chỉ cảnh báo | — |
| 18 | pacemaker leads | Support Devices | Support Devices | — |
| 19 | phim có xương bị gãy | Fracture (D63) | Fracture: noisy-OR 3 cột, cảnh báo yếu | — |
| 20 | no lobar pneumonia, with consolidation | có Consolidation, loại Pneumonia | đúng; thêm term "lobar pneumonia" → C0032300, không có đường, cảnh báo | C0032300 có dưới 50 lượt nhắc, không phân xử (plan) |
| 21 | tràn dịch màng phổi và sỏi thận | bỏ cụm không nhận ra | Pleural Effusion; "sỏi thận" bị bỏ kèm cảnh báo | — |
| 22 | mediastinum looks wider than normal | cột widened mediastinum | C0746459 (cột widened mediastinum), thẳng | SapBERT xếp trên hit fulltext phủ 1/3 (D68) |
| 23 | giảm thể tích phổi trái | Atelectasis (D64) | Atelectasis, cảnh báo bên, cảnh báo yếu | chú giải cho chooser (D67) |
| 24 | giảm thể tích phổi hai bên | C5563137 (thể tích phổi thấp) | C5563137, cảnh báo bên | — |

Vẫn còn hở:

- Câu 6 và 20: "lobar pneumonia" khi thì thành term, khi thì không, tùy lượt; khi thành term thì không có đường tới cột.
- Câu 11: parser lúc tách "small pleural effusions" thành term, lúc không.
- Mọi câu có vị trí: vị trí chưa được dùng để xếp hạng.

Mỗi lượt đã sửa gì:

| Lượt | Sửa | Câu đổi |
|---|---|---|
| a–d | đường truy vấn KG đầu tiên (D66) | — |
| e | nhãn cha không nuốt khái niệm con, vị trí chỉ cảnh báo, cạnh manual, chú giải, `sides` (D67) | 2, 4, 8, 9, 15, 17, 23 |
| f | SapBERT là bước cuối, vùng có trọng số, grounding theo alias (D68) | 12, 13, 22 |
| g | vùng chuẩn hóa theo max, cảnh báo `not_considered` (D69) | 7, 11, 12, 13; câu 22 hồi quy (fulltext che SapBERT) |
| h | lookup gộp ba bước, độ phủ fulltext, làm nóng SapBERT (D68 sửa) | 22 |
| i | 22 cạnh manual từ lượt rà đường | không câu nào đổi kết quả: 24 câu không hỏi tới bệnh nào vừa có đường |

## Rà đường cho Disease chưa chấm được (2026-11-08)

Đầu vào là các Disease lõi có lượt nhắc mà `expand_concepts` không đưa được tới cột nào (`scripts/kg_disease_routes.py`):
67 bệnh, không tính 4 bệnh chờ cột vòng hai (plan, X-1). Trong đó 29 bệnh có ≥ 50 lượt nhắc.
`cxr kg-graph-llm --task route-review` cho `claude-opus-5-5` đề xuất một trong ba: `is_a` (con của một cột), `suggests`
(một hay nhiều cột là dấu hiệu của bệnh) hoặc `none`. Claude bên chat phân xử. Chỉ những cột phân xử giữ lại mới thành
cạnh `manual` (`configs/kg.yaml: graph.manual_edges`; SUGGESTS có trọng số đều p = 1/k). Lượt này dùng prompt v1, chưa
có danh sách cột yếu; từ v2 thì cột yếu không được làm nguồn `suggests` (T42).

| CUI | Mô tả | Opus | Phân xử | Cột giữ | Lý do ngắn |
|---|---|---|---|---|---|
| C0020498 | bệnh cốt hóa dây chằng cột sống | suggests | suggests | C0011164 | thấy trên phim như thay đổi thoái hóa cột sống; bỏ cột vôi hóa vì quá chung |
| C1868833 | dày thành phế quản | suggests | none | — | là bệnh đường thở, không phải mô kẽ |
| C0006267 | giãn phế quản | suggests | suggests | C1849561 | tăng đậm vân phế quản–mô kẽ; bỏ Atelectasis (yếu) và các cột mờ không đặc hiệu |
| C0242699 | mất khoáng xương do bệnh | is_a | is_a | C0029453 | — |
| C0264541 | vôi hóa màng phổi | is_a | is_a | C0006663 | — |
| C0456973 | hạch rốn phổi to | suggests | suggests | C0746459; C0577559 | rốn phổi to lồi như khối, hay kèm trung thất rộng; bỏ Enlarged Cardiomediastinum (yếu) |
| C0264558 | tràn khí màng phổi có áp lực | is_a | is_a | C0032326 | — |
| C0155733 | xơ vữa động mạch chủ | suggests | suggests | C0006663; C0741171 | vôi hóa quai động mạch chủ, động mạch chủ ngoằn ngoèo |
| C0010674 | bệnh xơ nang | suggests | suggests | C1849561 | bỏ Atelectasis (yếu), Pneumothorax (hiếm), các cột mờ không đặc hiệu |
| C0577702 | viêm phổi thùy dưới | is_a | is_a | C0032285 | cha là cột yếu; leo IS_A kèm cảnh báo (D55) |
| C0003486 | phình động mạch chủ | suggests | suggests | C0746459 | trung thất rộng; bỏ cột yếu, cột vôi hóa và cột khối (thứ phát) |
| C0004153 | xơ vữa động mạch nói chung | suggests | suggests | C0006663; C0741171 | như C0155733 |
| C0034067 | khí phế thũng phổi | **none** | is_a | C0013990 | cột emphysema học từ chữ "emphysema" trong report, tức khí phế thũng phổi; Opus trả none là sai (T42) |
| C1285162 | bệnh thoái hóa nói chung | is_a | is_a | C0011164 | — |
| C0521515 | vôi hóa gân quanh khớp vai | suggests | suggests | C0006663 | thấy ở góc phim |
| C0265512 | đảo xương | suggests | none | — | cột nodule mang nghĩa nốt phổi; nối vào sẽ trả về nốt phổi sai |
| C0699949 | bệnh đường thở nói chung | none | none | — | dấu hiệu của nó (dày thành phế quản, ứ khí) không có cột |
| C0340766 | tăng áp tĩnh mạch (đọc là tĩnh mạch phổi) | suggests | suggests | C0034063; C0018800 | sung huyết tĩnh mạch phổi, phù mô kẽ, tim to; bỏ cột vân mô kẽ (Edema đã phủ) |
| C0037580 | sưng mô mềm | none | none | — | — |
| C0741276 | giãn nhĩ | is_a | is_a | C0018800 | một dạng to buồng tim; hạ về Cardiomegaly, kèm cảnh báo "hẹp hơn" |
| C0158266 | thoái hóa đĩa đệm | is_a | is_a | C0011164 | — |
| C0038353 | giãn dạ dày | suggests | none | — | không phải dấu hiệu X-quang được công nhận của cơ hoành cao; quan hệ gián tiếp |
| C0746592 | vôi hóa vòng van hai lá | is_a | is_a | C0006663 | — |
| C0340643 | bóc tách động mạch chủ | suggests | suggests | C0746459 | trung thất rộng; bỏ cột yếu, effusion và vôi hóa (thứ phát) |
| C0029408 | thoái hóa khớp | is_a | is_a | C0011164 | thay đổi thoái hóa cột sống và khớp vai trên phim |
| C0034072 | suy tim phải do bệnh phổi | suggests | suggests | C0018800 | bỏ Enlarged Cardiomediastinum (yếu) |
| C0575269 | gù cột sống | suggests | none | — | bất thường trục, không chấm được, nhất quán với scoliosis; liên hệ với osteopenia hay fracture là gián tiếp |
| C0032319 | khí trong khoang màng ngoài tim | **is_a** | none | — | Opus bám vào tên chung (khí tích tụ trong mô) của C0013990; cột emphysema chấm khí phế thũng phổi (T42) |
| C0018021 | bướu giáp sau xương ức | suggests | suggests | C0746459 | trung thất trên rộng; bỏ cột yếu và cột khối |

Cột được nhắc trong bảng: C0006663 vôi hóa, C0011164 thay đổi thoái hóa, C0013990 emphysema, C0018800 Cardiomegaly,
C0029453 osteopenia, C0032285 Pneumonia, C0032326 Pneumothorax, C0034063 Edema, C0577559 khối, C0741171 tortuous aorta,
C0746459 widened mediastinum, C1849561 vân mô kẽ.

Kết quả: 22 cạnh (10 IS_A, 12 SUGGESTS). Disease lõi có lượt nhắc mà không có đường: 71 → 48. Trong 48 bệnh này, 4 chờ cột
vòng hai, 7 có ≥ 50 lượt nhắc nhưng phân xử là `none`, 37 có dưới 50 lượt nhắc; 37 bệnh này không phân xử thêm trong V2
(plan). Một bệnh dưới 50 lượt nhắc (C0029410) có đường gián tiếp qua cạnh mới. Độ nối D59 theo lượt nhắc 0,9455 → 0,948
(`docs/measured-facts.md`).
