# GaugeEEG — Tổng quan ý tưởng & chi tiết kỹ thuật

> Tài liệu tổng hợp bằng tiếng Việt cho toàn bộ dự án GaugeEEG: ý tưởng nền
> tảng, cơ sở khoa học, kiến trúc kỹ thuật, chuỗi thí nghiệm E0–E15, kết luận
> từng bước và trạng thái hiện tại. Cập nhật đến kết quả E15 và kế hoạch khóa
> benchmark frozen-REVE đa baseline.

---

## 1. Ý tưởng cốt lõi (base idea)

### 1.1. Câu hỏi nghiên cứu

GaugeEEG là một **benchmark ưu tiên tính khả thi (feasibility-first)** trả lời
một câu hỏi rất cụ thể:

> **Biểu diễn (representation) của mô hình nền tảng EEG có ổn định không khi
> cùng một bản ghi được biểu diễn dưới một quy ước điện thế tham chiếu
> (voltage reference) khác nhưng vẫn hợp lệ về mặt vật lý?**

### 1.2. Đối xứng gauge (gauge symmetry) — nền tảng vật lý

Điện thế da đầu **không có mốc 0 tuyệt đối**. Nếu một mẫu EEG được biểu diễn bởi
ma trận `X` gồm `C` kênh, thì:

```text
X' = X + 1·a(t)
```

mô tả **cùng một trường điện thế theo cặp (pairwise voltage field)** với mọi tín
hiệu chung `a(t)` biến thiên theo thời gian. Một phép tái tham chiếu tuyến tính
với trọng số `w` (thỏa `wᵀ1 = 1`) là:

```text
R_w(X) = X − 1·(wᵀX)
```

- **Hiệu điện thế giữa các cặp kênh là bất biến** với phép biến đổi này.
- **Common Average Reference (CAR)** chính là phép chiếu chuẩn tắc lên không gian
  con trực giao với vector all-ones (vector toàn số 1) theo chiều kênh.

Đây là một **đối xứng gauge**: nhiều mảng số khác nhau cùng biểu diễn một hiệu
điện thế đo được thực sự như nhau. Việc thay đổi tham chiếu là một **biến nhiễu
có cấu trúc (structured nuisance variable)**, không làm thay đổi nhãn nhiệm vụ.

### 1.3. Nguồn gốc ý tưởng: chuyển giao nguyên lý từ NAI-SSL

- **NAI-SSL** lập luận rằng một mục tiêu tự giám sát (self-supervised) sẽ mạnh
  hơn khi tín hiệu giám sát được suy ra từ **một tính chất khoa học của dữ liệu**
  thay vì sao chép từ học biểu diễn ảnh tự nhiên. Quy tắc cụ thể của NAI-SSL là
  hiệp phương sai cấu trúc liên bán cầu trong ảnh MRI.
- GaugeEEG **giữ nguyên nguyên lý thiết kế** nhưng **thay đổi**: phương thức dữ
  liệu (EEG thay vì MRI), quy tắc khoa học (bất biến tham chiếu thay vì đối xứng
  bán cầu), benchmark, và baseline mỏ neo.
- **NAI-SSL không được dùng làm baseline** — nó chỉ là nguồn cảm hứng nguyên lý.

### 1.4. Baseline mỏ neo: REVE

- Baseline mỏ neo là **REVE** (NeurIPS 2025), một mô hình nền tảng EEG có mã
  nguồn công khai, trọng số đã phát hành, và kết quả PhysioNetMI đã công bố.
- REVE là mục tiêu phù hợp vì nó **xử lý tường minh các bố cục điện cực không
  đồng nhất** (heterogeneous electrode layouts) qua positional encoding 4D
  (điện cực *ở đâu* và patch *khi nào*).
- **Điểm giới hạn có thể kiểm chứng:** positional encoding 4D và loss MAE của
  REVE **không** buộc mô hình coi các tín hiệu chỉ khác nhau bởi một toán tử
  tham chiếu là tương đương. Do đó, hỗ trợ bố cục tùy ý **không tự động** kéo
  theo bất biến quy ước tham chiếu. Đây là giả thuyết cần **bác bỏ (falsify)**,
  không phải sự thật được giả định.

### 1.5. Bộ dữ liệu

- Benchmark đầu tiên dùng **PhysioNet EEG Motor Movement/Imagery Dataset
  (EEGMMIDB)** — mở, tải tự động qua MNE.
- Nhiệm vụ **4 lớp motor imagery** khớp định nghĩa của REVE: **left fist (nắm tay
  trái), right fist (nắm tay phải), both fists (cả hai tay), both feet (cả hai
  chân)**.
- Chia dữ liệu **theo subject (đối tượng), không bao giờ theo trial** để tránh rò
  rỉ (leakage).

---

## 2. Bậc thang đóng góp (contribution ladder)

1. **Benchmark RefShift-EEG.** Các "view" tham chiếu ghép cặp, bảo toàn nhãn,
   kèm chỉ số hiệu năng nhiệm vụ và độ trôi biểu diễn (representation drift).
2. **Baseline sanity chính xác.** CAR canonicalization chứng minh benchmark đang
   đo đúng thành phần gauge cộng tính đã dự định.
3. **Benchmark prior-confounding & hiệu chỉnh toán tử an toàn theo lớp.** Phơi
   bày ranh giới định danh (identifiability) giữa bias tham chiếu và dịch chuyển
   nhãn chưa biết (label shift), rồi điều chuẩn (regularize) các hiệu chỉnh
   không chắc chắn bằng toán tử quan sát + một "bộ bảo vệ lớp tệ nhất
   (worst-class safeguard)".
4. **Vượt ra ngoài dịch chuyển common-mode lý tưởng.** Mở rộng sang bipolar
   derivations, montage thiếu kênh, và chuyển giao chéo bộ dữ liệu — nơi CAR
   canonicalization chính xác một mình là không đủ.

> Chỉ mục 1 và baseline chính xác được hiện thực trong v0.1. Phương pháp học chỉ
> được thêm vào nếu pilot chứng minh có một chế độ thất bại (failure mode) có ý
> nghĩa.

---

## 3. Chi tiết kỹ thuật đã hiện thực

### 3.1. Toán tử tái tham chiếu (reference operators)

- **CAR** (common average).
- **Single-electrode** (Cz, Pz, FCz, Fz…).
- **Linear reference xác định (deterministic linear reference):** trọng số được
  cố định bởi một seed.
- Kiểm thử tổng hợp (synthetic test) chứng minh CAR canonicalization loại bỏ
  thành phần tham chiếu cộng tính (gauge) **tới sai số máy** (`< 1e-5`).

### 3.2. Tiền xử lý & encoder

- Tải + tiền xử lý PhysioNetMI tự động qua MNE.
- **Bandpower baseline:** log-bandpower nhẹ, chạy **không cần GPU** hay quyền
  truy cập mô hình — dùng làm smoke test không bị gate.
- **REVE (đóng băng - frozen):** theo interface đã phát hành — EEG 200 Hz, tọa
  độ điện cực từ `brain-bzh/reve-positions`, đầu vào microvolt chia 100, đầu
  attention-pooling của mô hình. Trọng số REVE có thỏa thuận sử dụng có trách
  nhiệm (responsible-use) → cần yêu cầu quyền trên HuggingFace và đăng nhập.

### 3.3. Các bộ đọc (probes / readouts) đã dùng qua các giai đoạn

- **Logistic probe nhẹ** trên embedding REVE đóng băng.
- **Token-level PyTorch probe kiểu chính thức:** token features đóng băng, khởi
  tạo query-token pretrained, query attention, flatten tokens, RMSNorm,
  dropout, đầu tuyến tính, huấn luyện tới 20 epoch (AdamW thay cho StableAdamW).
- **Attention-pool + linear probe** (E7b) — thất bại clean gate.
- **PMA (Pooling by Multihead Attention) / "set probe" q-query** (E7c trở đi):
  một bank query học được với multihead attention gộp một **tập token REVE có số
  lượng biến thiên** thành đầu vào phân loại có chiều cố định. Không flatten,
  không zero-pad. Đây là readout cho phép xử lý **montage bản địa (native)** với
  số kênh khác nhau.

### 3.4. Chỉ số đo (metrics)

- Reference-shift **accuracy, balanced accuracy (BAcc), macro-F1, AUROC**.
- **Cosine drift**, **relative L2 drift**, **linear CKA** (độ trôi biểu diễn).
- Đại lượng stress-test chính:

```text
reference_gap = balanced_accuracy(CAR test) − balanced_accuracy(shifted test)
```

### 3.5. Giao thức đánh giá & tái lập

- Chia theo subject, không theo trial.
- Bộ phân loại được **chọn trên tập validation CAR**, rồi refit trên train+val,
  và đánh giá **một lần** trên mỗi test reference.
- Seed xác định trọng số linear reference.
- Bản ghi tải về, trọng số mô hình, và features sinh ra đều bị Git bỏ qua.
- **Không** so sánh trực tiếp với bảng số của bài báo REVE cho tới khi dùng đủ
  cấu hình đầy đủ và nhiều seed.

---

## 4. Cấu trúc repo

| Đường dẫn | Nội dung |
|---|---|
| `src/gaugeeeg/` | Mã nguồn: reference operators, encoders, `set_probe.py`, `operator_consistency.py`, `cli.py`, `experiment.py`, `config.py`, các audit |
| `configs/` | Một file YAML cho mỗi giai đoạn thí nghiệm |
| `scripts/` + `Makefile` | Các runner (`make set-operator-consistency`, `make consistency-multiseed`, …) |
| `outputs/` | Kết quả JSON/CSV đã commit theo từng run (feature cache & trọng số bị git-ignore) |
| `docs/RESEARCH_PLAN.md` | Giả thuyết, khoảng trống nghiên cứu, bậc thang đóng góp, tiêu chí bác bỏ |
| `docs/EXPERIMENTS.md` | Trình tự thí nghiệm chính xác + tiêu chí chấp nhận từng bước |
| `conversation.md` | Nhật ký quyết định theo dòng thời gian |
| `README.md` | Hướng dẫn chạy + tường thuật kết quả từng giai đoạn |

**Mỗi run** tạo ra: `metrics.csv` (một dòng/defense × test reference),
`summary.json` (gap best/worst + metadata), `resolved_config.yaml` (cấu hình
chính xác đã dùng), `feature_cache/` (features đóng băng tái sử dụng),
`predictions.csv`, `subject_metrics.csv`, `paired_subject_bootstrap.csv`.

---

## 5. Sáu giả thuyết cốt lõi (H1–H5)

- **H1:** Embedding REVE đóng băng và hiệu năng linear-probe **thay đổi** dưới
  phép tái tham chiếu đơn cực hoặc tuyến tính hợp lệ.
- **H2:** CAR canonicalization chính xác loại bỏ thành phần tham chiếu đơn giản
  và phục hồi biểu diễn "view sạch", qua đó **hợp thức hóa benchmark**.
- **H3:** Dưới các thay đổi quy ước khó hơn (không thể canonicalize chính xác),
  học nhất quán gauge (gauge-consistency) cải thiện hiệu năng ở reference tệ
  nhất **mà không** giảm hiệu năng ở reference sạch.
- **H4:** Dưới tỉ lệ lớp mục tiêu chưa biết, một hiệu chỉnh có điều kiện theo
  toán tử (operator-conditioned) có thể cải thiện lỗi trung bình khi dịch chuyển
  nghiêm trọng, trong khi giữ nguyên điều kiện random và balanced.
  → **E14 đã BÁC BỎ** giả thuyết này cho phương pháp hiệu chỉnh E12 qua các
  probe seed chưa đụng tới.
- **H5:** Nhất quán CAR-teacher **ở thời điểm huấn luyện** cải thiện độ chính xác
  nhiệm vụ native16 **vượt trội hơn** augmentation đa view có giám sát, đồng
  thời giữ nguyên CAR sạch, native32, và recall lớp tệ nhất.
  → **E15 là phép kiểm định (chỉ phát triển - development-only) cho giả thuyết
  thay thế này.**

---

## 6. Chuỗi thí nghiệm E0 → E15 (chi tiết & kết luận từng bước)

> Nguyên tắc xuyên suốt: **thang bác bỏ (falsification ladder)**. Mỗi thí nghiệm
> khai báo trước một "cổng (gate)"; phần lớn giai đoạn **thu hẹp hoặc giết chết**
> một tuyên bố thay vì xác nhận nó. Không tiêu tốn GPU cho giai đoạn sau nếu sanity
> check trước đó thất bại.

### E0 — Kiểm chứng đại số tham chiếu
- **Lệnh:** `gaugeeeg synthetic`
- **Chấp nhận:** sai số phục hồi CAR < `1e-5`; sai số bất biến hiệu-cặp < `1e-5`;
  mọi ma trận reference triệt tiêu vector all-ones.
- **✅ Kết luận:** Đại số đúng tới độ chính xác số.

### E1 — Smoke test + baseline cổ điển
- **E1a** (`smoke.yaml`, bandpower): pipeline chạy end-to-end, đủ 4 nhãn trong
  mọi split. **E1b** (`pilot.yaml`, bandpower): hiệu năng CAR test trên mức ngẫu
  nhiên (>25%); `car_canonicalize` cho features gần như y hệt cho mọi view tham
  chiếu đơn giản.
- **✅ Kết luận:** Pipeline + benchmark hoạt động; CAR canonicalization ≈ identity.

### E2–E3 — Độ nhạy REVE đóng băng & clean gate
- Đo độ nhạy REVE đóng băng (pilot), rồi khẳng định trên official split.
- **Clean gate** yêu cầu BAcc ≥ 0.45 (bài báo REVE-Base pooled báo cáo
  0.537 ± 0.005). **✅ Đã vượt:** test BAcc **0.5567** trên full split.

### E4–E5 — Reference stress đa seed & audit thiên lệch theo lớp
- Reference stress: CAR, Cz, Pz, FCz + linear reference ngẫu nhiên xác định, một
  probe chỉ huấn luyện trên CAR. Screen 3 seed (7/21/42).
- **Kết quả marginal:** Cz giảm BAcc ~3.2 điểm trung bình, nhưng view tệ nhất &
  quyết định ngưỡng thay đổi giữa các probe seed.
- **E5 (class-bias audit, không cần GPU):** BAcc tổng hợp **che giấu** một dịch
  chuyển **theo lớp lớn hơn nhiều và lặp lại được** — cụ thể là **gap recall
  của left-fist do Cz gây ra**. Cz được giữ hoàn toàn ngoài các view huấn luyện.

### E6a–E6c — Loss nhất quán vs augmentation (giữ Cz ngoài)
- **E6a (seed 7):** giữ Cz ngoài cả huấn luyện lẫn validation. Huấn luyện probe
  CAR/Pz/FCz với multi-view cross-entropy thường (CE) và với cùng loss + nhất
  quán Jensen–Shannon. Cổng: giảm ≥30% gap recall giữ-ngoài-Cz *và* BAcc CAR
  giảm ≤1 điểm. Đóng góp của rule loss chỉ được công nhận nếu `rule_consistency`
  **cũng thắng** `multi_view_ce`; nếu không, quy phục hồi cho augmentation thường.
- **E6b (đa seed 7/21/42):** bootstrap phân cấp lấy mẫu lại cả probe seed lẫn
  test subject. Nhất quán thắng augmentation trên metric chính ở **cả 3 seed**,
  nhưng CI 95% phân cấp **hẹp nhưng cắt qua 0** → có hướng nhưng **chưa kết luận
  thống kê**.
- **E6c (ablation λ, chỉ validation):** lưới `λ ∈ {0, 0.3, 1, 3, 10}` (0 = multi-
  view CE, 1 = run nhất quán hiện có). Chọn λ **mù với test**: theo BAcc
  validation trung bình trên CAR/Pz/FCz và 3 seed; hòa thì chọn λ nhỏ hơn.
  → **Validation chọn λ = 10**, và lợi thế giữ-ngoài-Cz của nó so với multi-view
  CE thường **vượt qua** bootstrap phân cấp đã khai báo trước.

### E7a — Sparse montage bằng zero-fill
- Áp reference trước, rồi **zero-fill** các điện cực thiếu, giữ thứ tự 64 kênh.
  Đánh giá montage 32/16/8 quanh vùng vận động + drop trái/phải bất đối xứng.
  View chính cố định `sparse16@cz`, lớp mục tiêu left-fist.
- **❌ Kết luận (thất bại có ích):** gần như mọi điều kiện sparse **sụp về gần
  ngẫu nhiên 4 lớp**, gần như chỉ dự đoán một lớp. Zero-fill tạo pattern OOD
  không đáng có → đây là **hiện vật của mô hình đầu vào (input-modeling
  artifact)**, KHÔNG phải giới hạn nội tại của mô hình nền tảng.

### E7b — Tập con kênh bản địa (native) + attention-pool linear probe
- Sửa benchmark: chọn điện cực giữ lại, áp reference trong montage native đó,
  chỉ đưa tín hiệu + tọa độ của chúng vào REVE (**không zero-pad**). Dùng
  attention pooling của REVE + linear probe để chiều feature cố định.
- **❌ Kết luận:** **Không vượt clean gate** — attention-pool + linear chỉ đạt
  **0.3988** BAcc CAR sạch, `native16@cz` = **0.2606**. Plumbing native đúng,
  nhưng kết quả bị nhiễu bởi readout yếu.

### E7c — Cổng readout tập biến thiên (variable-set)
- Thay pooling single-query bằng **PMA**: bank query học được attend vào tập
  token REVE biến thiên. Query count chọn từ `{4, 8, 16}` theo **chỉ** BAcc
  validation CAR; hòa chọn nhỏ hơn. Clean CAR test BAcc dùng làm cổng 0.45.
- **✅ Kết luận (vượt mọi cổng khai báo trước ở seed 7):** chọn **q4** (validation
  BAcc **0.5595**), clean CAR test **0.6112**, `native16@cz` **0.3850**. Gap
  subject ghép cặp chính **0.2263**, CI 95% **[0.1790, 0.2724]**. q4 vượt hẳn q8
  và q16 trên validation.
- **Quan sát hậu nghiệm (post-hoc):** mô hình 16 kênh **vẫn giữ AUROC hữu ích**
  nhưng gần như không bao giờ chọn các lớp hai bên (bilateral).

### E7d — Đóng closure tham chiếu & theo lớp cho q4
- Đóng băng q4, thêm các view full-montage Cz/Pz/FCz, hình thức hóa chẩn đoán
  class-collapse. Yêu cầu **tái lập chính xác** dự đoán CAR của E7c.
- **Kết luận:** Full-montage **Pz** là dịch chuyển tổng hợp lớn nhất (gap BAcc
  **0.0452**, bất đồng dự đoán **27.1%**). Native16 CAR mất **0.2268** BAcc so
  với full CAR và gán **99.7%** dự đoán cho 2 lớp (**sụp hai lớp về mặt chức
  năng**), trong khi các lớp bilateral bị sụp vẫn giữ AUROC **0.615 / 0.681** →
  **bias biên quyết định theo lớp có thể phục hồi**, không phải mất thông tin.

### E7e — Chẩn đoán hình học tham chiếu native
- Kiểm tra xem gap BAcc native16 CAR-vs-Cz gần 0 có tổng quát cho các reference
  xa tâm montage không. Suite: Cz, Pz, Fz trong montage native 16 & 32. Pz/Fz
  chọn **trước** khi xem kết quả native (chúng được giữ trong cả hai montage và
  ít trung tâm hơn Cz).
- **Cổng phạm vi joint gauge/montage:** cần native16 Pz/Fz cho **hoặc** gap BAcc
  tuyệt đối ≥ 0.03 **hoặc** gap recall lớp ≥ 0.10 với CI subject-bootstrap loại
  trừ 0. Bất đồng dự đoán ≥ 0.15 là bằng chứng hỗ trợ nhưng không tự kích hoạt.

### E8 — Kiểm soát hiệu chuẩn chỉ-validation (calibration control)
- Kiểm tra giả thuyết thay thế của E7d/E7e: thay đổi tham chiếu có thể dịch
  **biên tương đối giữa các lớp** trong khi **giữ nguyên phần lớn thứ hạng
  trong-lớp**. Fit **nhiệt độ (temperature)**, **class-bias (zero-sum)**, và
  **vector scaling** chỉ bằng NLL validation. Class-bias là control chính
  (predeclared); temperature là control âm (không đổi argmax → chỉ đo NLL/ECE);
  vector scaling là phân tích độ nhạy. **Nhãn test không bao giờ** vào tối ưu.
- Hai giao thức: **hiệu chuẩn theo view mục tiêu (oracle known-view)** và
  **leave-one-view-out** (mới thực sự kiểm tra chuyển giao reference chưa thấy).
- **Kết luận:** **Chỉ** cổng class-bias theo-view vượt qua. Nó giảm gap recall
  tệ nhất từ **0.353 → 0.116** và tăng BAcc native tệ nhất từ **0.373 → 0.432**.
  Nhưng leave-one-view-out **làm gap recall tệ nhất tăng lên 0.614** (Pz bị
  over-correct). → Khoảng trống nghiên cứu được định nghĩa lại: **dự đoán một
  hiệu chỉnh có điều kiện theo reference/montage mà KHÔNG có nhãn từ toán tử
  quan sát mục tiêu.**

### E9 — Manifold bias tham chiếu (chỉ validation)
- Trích lưới điện-cực-×-reference đầy đủ cho **chỉ** validation subject. Chia:
  71–80 fit target bias oracle, 81–89 đánh giá. Giữ ngoài một danh tính điện cực
  qua cả hai montage. So sánh identity, global-mean, pooled bias (chiến lược E8
  mở rộng), ridge topology 10–20, ridge thống-kê-logit không nhãn, kết hợp, và
  oracle nhãn.
- **✅ Cả 3 cổng ứng viên đều vượt:** trên 48 view mục tiêu giữ-ngoài, RMSE bias
  topology/logit/combined = **0.223 / 0.037 / 0.044** so với **0.559** của
  global-mean. Logit-only & combined nâng BAcc trung bình **0.420 → 0.462** và
  giảm gap recall tệ nhất **0.478 → 0.205 / 0.181**.
- **Phát hiện then chốt (định danh):** bias oracle KHÔNG phải target phụ thuộc
  nhãn tổng quát. Ba tham số của nó tương quan **0.996–0.999** với logit-mean
  trung tâm hóa. Quan trọng hơn: **NLL của additive-bias phụ thuộc nhãn CHỈ qua
  tỉ lệ lớp (class prior)**. Vì lịch cue motor-imagery gần cân bằng, một **prior
  đều (uniform) đã biết** tái tạo tham số oracle giám sát tới sai số số (RMSE
  bias trung bình **0.0035**). → Ridge logit học được **không** phải baseline để
  mang tiếp.

### E10 — Prior đã biết & stress batch nhỏ
- Dùng lại logit validation E9 (chỉ CPU). Dùng **prior đều 4 lớp đã biết**. Kiểm
  tra batch không nhãn ngẫu nhiên 16–900 trial, batch cân bằng ở kích thước
  chính & stress, và skew đơn-lớp 40%/70% có kiểm soát. Trọng số prior-matching
  đặt **chỉ từ lỗi source-reference**: `w_prior = MSE_topology / (MSE_topology +
  MSE_prior)`.
- **Điều kiện chính:** random `n = 32`. **✅ Vượt cổng batch nhỏ:** topology
  shrinkage giảm RMSE bias trung bình **38.0%**, giảm gap recall tối đa trung
  bình **19.5%**, đổi BAcc trung bình **+0.0022**, CI RMSE-delta ghép cặp
  **[−0.154, −0.069]**.
- **Giới hạn bắt buộc báo cáo:** skew nghiêm trọng ở `n = 128` cho RMSE **gấp
  3.00 lần** batch cân bằng → **prior confounding** được xác nhận.

### E11 — Định danh prior chéo subject (cross-subject)
- Kiểm tra xem một **toán tử soft-confusion huấn luyện trên source** có thể trích
  một pseudo-prior thận trọng từ logit mục tiêu đóng băng và dùng nó sửa dịch
  chuyển prior nghiêm trọng không. Model class-probability & topology dùng
  71–75; batch thích nghi (adaptation) dùng 76–80 tách rời; hiệu ứng nhiệm vụ
  audit trên 81–89. Confusion matrix ước lượng bằng leave-one-subject-out.
- **Kết luận:** Hỗ trợ **robustness trung bình** khi skew nghiêm trọng (giảm
  RMSE **8.95%**, CI ghép cặp **[−0.0219, −0.0113]**) nhưng **KHÔNG** class-
  uniform: 3/4 hướng lớp cải thiện, riêng **right-fist tệ đi** (0.1452 → 0.1482).
- **Lỗi giao thức phát hiện sau:** run E11 lưu trữ ban đầu dùng subjects 76–80
  **cả** trong fit topology **lẫn** adaptation → số liệu đó chỉ là **chẩn đoán
  thăm dò**, không phải bằng chứng chéo-subject nghiêm ngặt. E12 chạy lại E11 nội
  bộ với split 71–75 / 76–80 / 81–89.

### E12 — Bộ bảo vệ tin cậy lớp/toán tử chỉ-source (safeguard)
- Với mỗi danh tính reference giữ-ngoài, học **4 "trần tin cậy (trust cap)"
  đường chéo** bằng bình phương tối thiểu trong không gian bias zero-sum 4 lớp,
  **chỉ** dùng subjects 71–75. Điện cực mục tiêu bị loại khỏi cả ví dụ source lẫn
  mỗi fit topology lồng nhau. Khi triển khai, cap **chỉ có thể GIẢM** trọng số
  pseudo-prior đã có của E11 (fall back về topology, không khuếch đại).
- **✅ Vượt cổng repeated-seed:** RMSE skew nghiêm trọng **0.3489 → 0.2013
  (−42.3%)**, đánh bại topology-only (0.2239). **Cả 4 hướng lớp cải thiện** theo
  điểm ước lượng, không phát hiện hại theo lớp.
- **Nhưng:** CI cụm right-fist **vẫn cắt 0** (~[−0.0266, 0.0206]) →
  `paper_level_class_uniform_claim_supported = false`. Đây là screen một-seed hứa
  hẹn, cần lặp seed + bộ dữ liệu ngoài.

### E13 — Audit baseline mạnh nhất (post-hoc)
- **Không** refit model. Dùng lại toàn bộ metric E12, áp bootstrap ghép cặp
  repeat-×-reference so với **cả hai** baseline mạnh: **strict E11
  operator-confusion shrinkage** và **topology-only ridge**.
- **Kết luận:** E12 có RMSE trung bình thấp hơn cả hai baseline mạnh ở cả 3 chế
  độ (random/balanced/severe). Nhưng ở điều kiện chính random `n=32`, delta RMSE
  so với strict E11 là **−0.0076**, CI **[−0.0193, 0.0033]** → một seed **chưa
  xác nhận** tuyên bố trung bình. Quan trọng hơn: **right-fist severe RMSE cao
  hơn** strict E11 **0.0182**, CI **[0.0006, 0.0361]**. → E13 chỉ đưa **giả
  thuyết mean-only** lên seed mới và **bác bỏ** tuyên bố class-uniform hiện tại.
- Vì cổng mạnh hơn này được định nghĩa **sau** khi xem seed 7, E13 chỉ có thể
  bác bỏ, **không** thể xác nhận một tuyên bố paper.

### E14 — Xác nhận probe seed chưa đụng tới (untouched)
- Sửa một giới hạn giao thức: probe q4 cũ dùng subjects 71–89 để early-stopping,
  trong khi E12 lại chia đúng nhóm này thành source/adaptation/evaluation → seed
  7 không thể coi là xác nhận độc lập. E14 dùng **split 4 chiều**: probe train
  **1–60**, probe validation **61–70**, audit downstream **71–89**, test dự trữ
  **90–109 (không bao giờ fit/score)**. Chạy **2 probe seed chưa đụng: 21 & 42**.
- Cổng (đóng băng theo luật E13): mỗi seed mới phải cải thiện RMSE trung bình so
  với **cả** strict E11 lẫn topology ở **cả 3** chế độ đóng băng; CI RMSE phân
  cấp dưới 0 cho cả 6 so sánh; noninferiority BAcc & gap-recall; không hại
  trung bình.
- **❌ Kết luận (âm tính):** **Seed 21 trượt** cổng điểm all-regime; **seed 42
  vượt**. Qua 2 seed, RMSE candidate-trừ-topology ≈ **−0.0064** (random),
  **+0.0001** (balanced), **+0.0001** (severe) — **cả 3 CI 95% phân cấp đều cắt
  0**. Noninferiority nhiệm vụ cũng trượt. → **Phương pháp cap hậu nghiệm E12 bị
  ĐÓNG LẠI**, không tinh chỉnh thêm trên audit subject.
- (Ghi chú kỹ thuật: trong quá trình E14, đã sửa một lỗi hội tụ optimizer
  `fit_known_prior_bias` khi một lớp có khối lượng bằng 0 — bỏ hard-clip
  `[−5, 5]` gây "false line-search failure", giữ log-sum-exp ổn định + L2 dương
  + backtracking; thêm regression test. Xem commit `d759668`.)

### E15 — Nhất quán toán tử quan sát ở thời điểm huấn luyện (ĐÃ CHẠY — FAIL)
- **Pivot ý tưởng lên thượng nguồn (upstream):** thay vì hiệu chỉnh logit hậu
  nghiệm (dòng E8→E14 đã chết), **dạy** cho set-probe q4 rằng native32/native16
  là các **toán tử quan sát lồng nhau (nested observation operators)** của trial
  full-CAR **ngay trong huấn luyện**.
- **Ba nhánh, cùng probe/reference seed 7** (train 1–60, early-stop 61–70, screen
  71–89, test 90–109 khóa lại):
  1. **CAR-only CE** (chỉ CAR, cross-entropy có giám sát).
  2. **Multi-view CE** trên CAR/native32/native16 (baseline augmentation quyết
     định — control tính mới then chốt).
  3. **Operator consistency**: multi-view CE **+** dùng dự đoán full-CAR làm
     **teacher tách gradient (stop-gradient)** cấp target KL cho native32 &
     native16 với **trọng số cố định 0.5 và 1.0**:

     ```text
     CE(mọi view) + 0.5·KL(CAR ‖ native32) + 1.0·KL(CAR ‖ native16)
     ```

     Hệ số nhất quán cố định = 1.0 **trước** khi xem kết quả audit.
- **Cổng (chỉ phát triển, subjects 71–89):** giữ clean CAR trong 0.01 (điểm &
  CI); **native16@CAR BAcc ≥ +0.02** so với CAR-only với CI ghép cặp subject
  **trên 0**; CI **trên 0** so với multi-view CE; giữ recall lớp tệ nhất
  native16 trong 0.01; **không** mất BAcc native32 theo điểm.
- **Kiểm tra tính mới then chốt:** nếu multi-view CE **bằng hoặc thắng** operator
  consistency → bằng chứng ủng hộ augmentation, **KHÔNG** phải rule loss; không
  được báo cáo rule như đóng góp.
- **Kết quả chính (seed 7, audit 71–89):** trên `native16@CAR`, CAR-only =
  **0.3470**, multi-view CE = **0.4561**, operator consistency = **0.4546**
  BAcc. Operator consistency cải thiện so với CAR-only (**+0.1076**, CI 95%
  **[+0.0674, +0.1506]**) nhưng không thắng multi-view CE (**−0.0015**, CI
  **[−0.0195, +0.0175]**). Trên clean CAR, candidate cũng thấp hơn CAR-only
  **−0.0102**, CI cắt 0 và không đạt gate noninferiority đã định trước.
- **❌ Kết luận:** rule KL teacher không có bằng chứng mang giá trị vượt
  augmentation thường. Giữ **multi-view CE** làm baseline hiện tại và bỏ tuyên
  bố đóng góp rule của E15. Kết quả này chỉ chứng minh giới hạn của thiết kế
  readout frozen-REVE cụ thể, không chứng minh giới hạn của mọi EEG foundation
  model hay mọi phương pháp rule-informed.
- **Lệnh tái lập & artifact quyết định:**

  ```bash
  make test
  make set-operator-consistency        # hoặc DEVICE=cuda:1 make set-operator-consistency
  ```

  Quyết định ghi tại `operator_consistency_summary.json`. Bốn thư mục artifact:
  `reve_set_operator_screen_car_only_s7`, `reve_set_operator_screen_multi_view_ce_s7`,
  `reve_set_operator_screen_consistency_s7`, `reve_set_operator_consistency_screen_s7`.

---

## 7. Tiêu chí Go/No-Go & mối đe dọa tính hợp lệ

### 7.1. Đi tiếp lên phương pháp học nếu (qua nhiều seed & full split):
- ít nhất một reference hợp lệ gây **giảm ≥ 3 điểm phần trăm BAcc** ở mô hình
  đóng băng không bảo vệ;
- hướng giảm **ổn định qua các seed**;
- độ trôi embedding không tầm thường (vd cosine ghép cặp trung bình < 0.95 hoặc
  CKA giảm rõ);
- CAR canonicalization phục hồi **≥ 80%** gap reference đơn giản.

### 7.2. Chuyển hướng (pivot) sang bipolar/thiếu-kênh nếu mọi drop reference đơn
giản < 1 điểm và cosine ghép cặp > 0.98. Dừng hướng này nếu các dịch chuyển khó
hơn cũng không gây trôi/giảm đáng kể.

### 7.3. Mối đe dọa tính hợp lệ (threats to validity)
- PhysioNetMI là **một** nhiệm vụ, **một** họ thu nhận → bài báo cần ít nhất một
  bộ dữ liệu mở khác.
- Trọng số REVE cần chấp thuận responsible-use → bandpower là smoke test không
  bị gate.
- Một kênh reference đơn lẻ có thể chứa nhiễu sensor → báo cáo cả reference đặt
  tên xác định lẫn linear reference phân tán.
- CAR chỉ chính xác khi giữ mọi kênh và thay đổi tham chiếu thuần common-mode.
- Kết quả pilot 12-subject chỉ để debug/ước lượng effect-size, **không** phải
  tuyên bố paper.
- E12/E13 vẫn một probe seed & một bộ dữ liệu. Cap chỉ-source được cố định trước
  audit, nhưng chuyển giao ngoài & bất định đa-seed là **bắt buộc**.
- E13 định nghĩa luật baseline mạnh **sau** khi xem seed 7 → chỉ loại bỏ được
  tuyên bố không hỗ trợ.
- E15 theo sau việc xem E14 và dùng audit subjects 71–89 → **không thể** xác
  nhận tuyên bố paper; vai trò hợp lệ duy nhất là chọn/bác bỏ phương pháp. Teacher
  full-CAR có thể chỉ chuyển giao chính lỗi của nó, và các view lồng nhau tương
  quan làm giảm đa dạng hiệu dụng — nhánh multi-view CE phân biệt giá trị nhất
  quán-toán-tử với augmentation thường nhưng không loại bỏ được giới hạn "một họ
  thu nhận".

---

## 8. Kết luận tổng thể & trạng thái hiện tại

1. **Benchmark + baseline canonicalization chính xác** (mục 1–2 của bậc thang)
   **vững chắc và đã được hợp thức hóa** (E0–E3).
2. Một **loss nhất quán được chọn qua validation** cải thiện rõ robustness của
   full-montage Cz giữ-ngoài **so với augmentation thường** (E6). Đây vẫn là một
   **readout bền vững trên encoder đóng băng**.
3. **Dòng hiệu chỉnh logit hậu nghiệm (E8 → E14) là ngõ cụt:** nó trông mạnh trên
   *một* probe fit nhưng **không chuyển giao** qua các seed tối ưu probe. E14 đã
   **bác bỏ dứt điểm H4** cho phương pháp E12.
4. **E15 đã bác bỏ H5 ở dạng hiện tại:** multi-view CE phục hồi native montage,
   nhưng KL teacher full-CAR không thắng multi-view CE và không qua gate clean.
5. **Công việc hiện tại:** khóa benchmark frozen-REVE với các baseline reference,
   montage có cấu trúc/ngẫu nhiên, region dropout, joint augmentation và generic
   JS consistency qua seed 7/21/42. Sau đó mới thiết kế phương pháp mới và xác
   nhận trên một dataset ngoài PhysioNetMI.

### 8.1. Trạng thái kết quả E15

- E15 đã có đủ ba run CAR-only, multi-view CE và operator consistency cùng file
  quyết định `outputs/reve_set_operator_consistency_screen_s7/operator_consistency_summary.json`.
- Trường quyết định là
  `operator_consistency_development_gate_supported: false`; khuyến nghị được lưu
  là `retain_multiview_ce_drop_unjustified_rule_loss`.
- Subjects 90–109 không được fit/score trong E15, nhưng vì đã được xem ở E3–E8,
  chúng không còn là test paper hoàn toàn untouched. Cần dataset ngoài để xác
  nhận paper-level.

### 8.2. Lệnh khóa baseline tiếp theo

```bash
make test
DEVICE=cuda make benchmark-baselines
```

Runner ghi nhận chính xác fingerprint preprocessing, revision của REVE và
position model, chỉ score development audit 71–89, rồi xếp hạng theo mean BAcc
trên `native16@{CAR,Cz,Pz,Fz}` với ràng buộc clean CAR. Chi tiết và literature
review nằm ở `docs/BASELINE_PLAN.md`.

---

## 9. Tài nguyên & giấy phép

- Mã REVE: giấy phép **MIT**; trọng số có thỏa thuận responsible-use riêng,
  **không** được repo này phân phối lại.
- File PhysioNet EEGMMIDB: **Open Data Commons Attribution License v1.0**.
- Mã GaugeEEG: **MIT License**.

---

## 10. Bảng tra cứu nhanh (cheat-sheet lệnh)

| ID | Mục đích | Lệnh | Kết quả chính |
|---|---|---|---|
| E0 | Đại số tham chiếu | `gaugeeeg synthetic` | ✅ sai số < 1e-5 |
| E1a/b | Smoke + bandpower | `gaugeeeg run --config configs/{smoke,pilot}.yaml --encoder bandpower` | ✅ pipeline OK |
| E2/E3 | REVE sensitivity + clean gate | `gaugeeeg run --config configs/reve_clean_gate.yaml --device cuda` | ✅ BAcc 0.5567 |
| E4 | Reference stress 3-seed | `configs/reve_reference_stress.yaml` | Cz ~ −3.2đ (marginal) |
| E5 | Class-bias audit | `gaugeeeg class-bias-audit …` | Gap recall left-fist ổn định |
| E6a | Held-out-Cz method screen | `configs/reve_consistency_screen.yaml` | ✅ pass |
| E6b | Đa seed | `make consistency-multiseed` | Có hướng, CI cắt 0 |
| E6c | Ablation λ | `make consistency-lambda-ablation` | ✅ chọn λ=10 |
| E7a | Sparse zero-fill | `make montage-screen` | ❌ sụp ~ngẫu nhiên |
| E7b | Native subset + linear | `make native-montage-screen` | ❌ clean 0.3988 |
| E7c | Variable-set PMA | `make set-native-screen` | ✅ q4, clean 0.6112 |
| E7d | Reference/class closure | `make set-reference-closure` | Sụp 2 lớp, AUROC còn |
| E7e | Native geometry | `make set-reference-geometry` | Quyết định phạm vi joint |
| E8 | Calibration control | `make set-calibration-control` | Chỉ known-view pass |
| E9 | Bias manifold | `make set-bias-manifold` | ✅ nhưng prior-only đủ |
| E10 | Known-prior / batch nhỏ | `make set-prior-stress` | ✅ n=32; skew ×3 RMSE |
| E11 | Cross-subject prior | `make set-prior-identifiability` | Mean OK, không class-uniform |
| E12 | Class/operator safeguard | `make set-class-safeguard` | −42.3% severe; right-fist CI cắt 0 |
| E13 | Strongest-baseline audit | `make set-strong-baseline-audit` | Chỉ mean-only đủ điều kiện |
| E14 | Untouched probe seeds | `make set-probe-seed-confirmation` | ❌ bác bỏ (seed 21 trượt) |
| **E15** | **Operator consistency (train-time)** | **`make set-operator-consistency`** | **❌ rule không thắng multi-view CE** |
| **Khóa baseline** | **7 baseline × seed 7/21/42** | **`make benchmark-baselines`** | **⏳ code sẵn sàng, chờ GPU run** |

---

*Tài liệu tham chiếu chéo: `README.md`, `docs/RESEARCH_PLAN.md`,
`docs/EXPERIMENTS.md`, `conversation.md`, và các `outputs/*/summary.json`.*
