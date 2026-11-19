# Đối Chiếu Thông Số Bài Báo SCOPE (ACCV 2026) & Kho Mã Nguồn

Tài liệu này đối chiếu toàn diện từng phương trình toán học, kiến trúc mạng, số lượng tham số, hàm mất mát, phép chuẩn hóa mức độ nén (`severity`), kịch bản đánh giá/đo thời gian và toàn bộ **11/11 trọng số chính thức** giữa:
1. **Bài báo gốc & Phụ lục (Supplementary)**: *SCOPE: Semantic COdec Pre-Editing for Object-Preserving Compression* — Silvia Dani, Leonardo Galteri, Marco Bertini (`ACCV 2026`, `arXiv:2610.39778v1`).
2. **Trọng số chính thức trên Google Drive của nhóm tác giả** (`FULL` + 10 mô hình ablation tại `1YbB1M0Bdp1QZl-aoGNWCCsJRMgzROGES`).
3. **Kho mã nguồn gốc `miccunifi/SCOPE`** (`commit 453a40b`) và **các điểm lệch thông số / lỗi thực thi đã được chuẩn hóa trong kho này (`munnn01/scope_again`)**.

---

## 1. Kiến Trúc Mạng & Toàn Bộ 11/11 Checkpoint Chính Thức (Section 3.1, Fig. 5 & Table 3)

| Thành phần | Bài báo (`arXiv:2610.39778v1`) | Checkpoint chính thức (`FULL/best_val.pt`) | `miccunifi/SCOPE` gốc (`scope/model.py`) | Trạng thái sau chuẩn hóa |
|:--|:--|:--|:--|:--|
| Đầu vào ảnh $I$ | RGB $\in [0, 1]^{3 \times H \times W}$ | `[B, 3, H, W]` float `[0, 1]` | Khớp | **Khớp 100%** |
| Nhúng họ codec $e_f$ | $\mathbb{R}^8$ (`3` họ huấn luyện) | `family_emb.weight`: `(3, 8)` | `family_emb_dim = 8` | **Khớp 100%** |
| Cơ sở mức nén $\phi(s)$ (Eq. 1–2) | $[s, s^2, s^3, h_{0.50}(s), h_{0.75}(s), h_{0.90}(s)] \in \mathbb{R}^6$, $h_\tau(s) = \mathrm{clip}(\frac{s-\tau}{1-\tau}, 0, 1)$ | `severity_basis_dim = 6` | `_expand_scalar_severity` trả về `[B, 6]` | **Khớp 100%** |
| MLP điều kiện (Eq. 3) | $\mathrm{MLP}([e_f, \phi(s)]) \in \mathbb{R}^{64}$ | `Linear(14, 64) -> GELU -> Linear(64, 64) -> GELU` | `cond_dim = 64` | **Khớp 100%** |
| U-Net đa tỷ lệ & FiLM (Eq. 4) | $H \times W$ (`48`), $H/2$ (`96`), $H/4$ (`192`), Bilinear upsample + Conv $3\times 3$, skip concat, $\mathrm{FiLM}(x, c) = x \odot (1 + \gamma(c)) + \beta(c)$ | `base_ch = 48`, khởi tạo FiLM đồng nhất ($\gamma=0, \beta=0$) | Khớp | **Khớp 100%** |
| Biên độ dư tối đa (Sec. 3.2) | $\|\Delta_{\mathrm{obj}}\|_\infty \le 0.09$, $\|\Delta_{\mathrm{bg}}\|_\infty \le 0.03$ | `max_delta_obj = 0.09`, `max_delta_bg = 0.03` (`tanh`) | Khớp | **Khớp 100%** |
| Nhánh nền làm mờ (Eq. 7 & Sec. 3.2) | $I_{\mathrm{bg}} = \mathrm{clip}(\mathrm{Blur}(I) + \Delta_{\mathrm{bg}}, 0, 1)$ | `bg_mode = "blur_replace"`, `bg_blur_kernel = 51`, `attention_bias = -4.0` ($\sigma(-4) \approx 0.018$) | **Lệch mặc định trong `SCOPE.__init__`**: để cũ `bg_mode="residual"`, `bg_blur_kernel=41`, `attention_bias=-3.0` | **Đã sửa mặc định `SCOPE.__init__` sang `blur_replace`, `51`, `-4.0`** |

### Bảng kiểm chứng thực tế cả 11/11 Checkpoint trên Google Drive (`load_scope_model(..., strict=True)`)

| Nhóm trong bài báo | Tên mô hình / Checkpoint trên Drive | `conditioning_mode` | `codec_family_to_id` | Số tham số | Trạng thái nạp `strict=True` |
|:--|:--|:--|:--|--:|:--|
| **SCOPE (Full)** — Table 3 (`−15.98%` BDBR) | `..._FULL_20260701_140557/best_val.pt` | `family_severity` | `{diff_jpeg:0, bmshj:1, bpg_like:2}` | **`1,879,119`** | **OK** |
| **w/o bg/rate** — Fig. 5 ($\lambda_{\mathrm{bg\_flat}}=\lambda_{\mathrm{bg\_smooth}}=\lambda_{\mathrm{rate}}=0$) | `..._NOBG_20260702_220324/best_val.pt` | `family_severity` | `{diff_jpeg:0, bmshj:1, bpg_like:2}` | **`1,879,119`** | **OK** |
| **w/o task** — Fig. 5 ($\lambda_{\mathrm{det}}=0$) | `..._NODET_20260702_164808/best_val.pt` | `family_severity` | `{diff_jpeg:0, bmshj:1, bpg_like:2}` | **`1,879,119`** | **OK** |
| **Family** — Table 3 (`−15.42%` BDBR) | `..._familyonly_scratch_20260828_165524/best_val.pt` | `codec_family_only` | `{diff_jpeg:0, bmshj:1, bpg_like:2}` | **`1,878,735`** | **OK** |
| **Severity** — Table 3 (`−13.28%` BDBR) | `..._SEVERITYONLY_scratch_20260829_022023/best_val.pt` | `severity_only` | `{diff_jpeg:0, bmshj:1, bpg_like:2}` | **`1,878,583`** | **OK** |
| **Per-codec (JPEG / BPG / BMSHJ)** — Table 3 | `..._{JPEG,BPG,BMSHJ}_UNCONDITIONED_scratch_.../best_val.pt` (3 ckpts) | `none` | Đơn codec (`{diff_jpeg:0}` / `{bpg_like:0}` / `{bmshj:0}`) | **`1,742,935`** | **OK (3/3)** |
| **Per-codec+Sev. (JPEG / BPG / BMSHJ)** — Table 3 | `..._{JPEG,BPG,BMSHJ}_SEVERITYONLY_percodec_scratch_.../best_val.pt` (3 ckpts) | `severity_only` | Đơn codec (`{diff_jpeg:0}` / `{bpg_like:0}` / `{bmshj:0}`) | **`1,878,583`** | **OK (3/3)** |

---

## 2. Siêu Tham Số Huấn Luyện & Hàm Mất Mát (Section 3.2 & Section 3.3)

Trong `miccunifi/SCOPE` gốc, tệp shell `training/train_scope.sh` chứa cấu hình đúng của bài báo, nhưng bộ phân tích tham số Python `scope/train_scope.py` (khi gọi trực tiếp `python training/train_scope.py ...` trên Windows hoặc Colab) lại giữ các giá trị mặc định cũ từ giai đoạn thử nghiệm YOLO trước đó:

| Siêu tham số | Bài báo (`main.tex`) & `FULL/config.json` | `training/train_scope.sh` | `scope/train_scope.py` gốc (Lệch!) | Sau chuẩn hóa (`scope/train_scope.py`) |
|:--|:--|:--|:--|:--|
| Tập dữ liệu & Crop | COCO 2017, crop $512 \times 512$ quanh đối tượng | `crop_size=512`, `object_crop_prob=0.60`, `crop_margin=4.0` | Khớp CLI (`data.py` để cũ `0.90 / 1.8`) | **`512`, `0.60`, `4.0` đồng nhất cả CLI & `CocoBoxCropDataset`** |
| Họ codec huấn luyện $\mathcal{F}$ (Eq. 10) | `{Diff_JPEG, BMSHJ, BPG_like}` lấy mẫu đều | `diff_jpeg,bmshj,bpg_like` (`1/3` mỗi họ) | Khớp | **Khớp 100%** |
| Lưới `severity` huấn luyện | Tuyến tính đều trên $[0, 1]$ | `0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0` (11 mức) | `0.0,0.25,0.5,0.75,1.0` (5 mức) | **`0.0,0.1,...,1.0` (11 mức)** |
| Lưới `severity` validation | Đánh giá định kỳ chọn `best_val.pt` | `0.0,0.2,0.4,0.6,0.8,1.0` (`val_every=5`, `val_steps=24`) | `""` (`val_every=1`, `val_steps=50`) | **`0.0,0.2,0.4,0.6,0.8,1.0` (`val_every=5`, `val_steps=24`)** |
| Dải chất lượng Diff-JPEG | `q_min = 1`, `q_max = 99` | `1 .. 99` | **`10 .. 99`** (làm lệch công thức `s(q)`) | **`1 .. 99` (cả `train_scope.py` & `codecs.py`)** |
| Dải QP BPG proxy | `qp_min = 0`, `qp_max = 51`, `medium` | `0 .. 51`, `medium` | `codecs.py` fallback để `28 .. 48` | **`0 .. 51`, `medium` đồng nhất** |
| BMSHJ proxy | `bmshj2018-hyperprior`, `q = 1..8`, `bpp_scale = 0.03` | Khớp | Khớp | **Khớp 100%** |
| Epochs / Batch / Samples | `40` epochs, batch `8`, AdamW, cosine warmup `3` epochs, grad clip `1.0`, AMP | `epochs=40`, `batch=8`, `samples_per_epoch=12000` (`1500` steps/epoch) | **`samples_per_epoch=20000`** | **`epochs=40`, `batch=8`, `samples_per_epoch=12000`** |
| Learning rate (`--lr`) | **$10^{-4}$** (`main.tex` Sec. 3.2) | `1e-4` | **`2e-4`** (gấp đôi bài báo!) | **`1e-4`** |
| Giáo viên đóng băng $D$ (Sec. 3.2) | **Detectron2 Faster R-CNN ResNeXt-101-32x8d-FPN** (`p2..p6`, Smooth L1, `0.7` masked + `0.3` full) | `detectron2`, `faster_rcnn_X_101_32x8d_FPN_3x.yaml` | **`yolo`** với `detector_weights=""` (khiến $\mathcal{L}_{\mathrm{det}} \equiv 0$!) | **`detectron2` (`faster_rcnn_X_101_32x8d_FPN_3x.yaml`)** |
| Làm mềm mặt nạ $M_{\mathrm{attn}}, M_{\mathrm{det}}$ | $M_{\mathrm{attn}}$ sát hộp GT, $M_{\mathrm{det}}$ mở rộng ngữ cảnh | `attn_blur=3`, `det_mask_blur=41` | `attn_blur=7`, `det_mask_blur=31` | **`attn_blur=3`, `det_mask_blur=41`** |
| Trọng số hàm mất mát (Eq. 11–16) | Không dùng reweighting theo `severity` (`gain/drop = 0`) | `lambda_bg_flat = 0.25`<br>`lambda_bg_smooth = 0.35`<br>`lambda_rate = 0.50`<br>`lambda_attn = 1.00`<br>`lambda_attn_sparse = 0.15`<br>`lambda_attn_area = 0.50`<br>`lambda_delta = 0.02`<br>`lambda_delta_tv = 0.05`<br>`lambda_det = 4.00` | `0.75`<br>`1.00`<br>`2.00`<br>`1.00`<br>`0.75`<br>`2.00`<br>`0.02`<br>`0.05`<br>`2.00` | **Chuẩn hóa 100% theo `FULL/config.json` & `train_scope.sh`** |

---

## 3. Đánh Giá Thực Nghiệm, Đo Thời Gian & BDBR (Section 4 & Supplementary Sections 1–2)

Tất cả điểm hoạt động trong `configs/eval.yaml` và `configs/ablation.yaml` khớp chính xác với Section 4 của bài báo (5,000 ảnh `COCO 2017 val2017`, 3 kiến trúc detector: **Faster R-CNN X101-FPN**, **RT-DETR-l**, **YOLO26x**; đo bitrate thực từ luồng bit đã mã hóa; BDBR nội suy PCHIP trên miền AP chung không ngoại suy):

| Bộ mã hóa | Trạng thái | Điều kiện họ (`codec_family`) | Các điểm hoạt động trong bài báo | Công thức chuẩn hóa `severity` $s \in [0, 1]$ |
|:--|:--|:--|:--|:--|
| **JPEG** | Seen | `block_dct` $\to$ `diff_jpeg` | $q \in \{10, 15, 20, 30, 40, 50\}$ | $s = 1 - \frac{q - 1}{99 - 1} = \frac{99 - q}{98} \in \{0.908163, \dots, 0.500000\}$ |
| **BPG** | Seen | `intra_smooth` $\to$ `bpg_like` | $\mathrm{QP} \in \{28, 31, 34, 37, 41, 44\}$ | $s = \frac{\mathrm{QP} - 0}{51 - 0} = \frac{\mathrm{QP}}{51} \in \{0.549020, \dots, 0.862745\}$ |
| **BMSHJ** | Seen | `learned` $\to$ `bmshj` | $q \in \{1, 2, 3, 4, 5\}$ (`bmshj2018-hyperprior`) | $s = 1 - \frac{q - 1}{8 - 1} = \frac{8 - q}{7} \in \{1.000000, \dots, 0.428571\}$ |
| **Cheng** | Zero-shot (Unseen) | `bmshj` | $q \in \{1, 2, 3, 4, 5, 6\}$ (`cheng2020-attn`) | $s = 1 - \frac{q - 1}{6 - 1} = \frac{6 - q}{5} \in \{1.000000, \dots, 0.000000\}$ |
| **JPEG2000** | Zero-shot (Unseen family) | Thử cả 3 họ đã học (`diff_jpeg`, `bmshj`, `bpg_like`) | OpenJPEG `-r` $\in \{32, 48, 64, 96, 128, 192\}$ | $s = \frac{r - 1}{192 - 1} = \frac{r - 1}{191} \in \{0.162304, \dots, 1.000000\}$ |

- **Đồng bộ `severity` ở mọi nhánh đánh giá**: Đã chuẩn hóa dải `JPEG [1, 99]` và `BPG [0, 51]` đồng nhất trên cả `scope/evaluation/conditioning.py`, `scope/evaluation/transmission.py` và `scope/evaluation/rate_distortion.py` (`evaluate_categories.py` — Supplementary Sec. 2).
- **Đo thời gian chuẩn Supplementary Sec. 1 (`evaluation/benchmark_scope.py`)**: `configs/timing.yaml` gồm 3 họ × 5 mức `severity` $\{0.00, 0.25, 0.50, 0.75, 1.00\}$, `--warmup 20` và **`--repeat 1`** mặc định (tổng `75,000` phép đo trên `5,000` ảnh COCO val2017 đúng như Table 1 & Table 2 phần Phụ lục); đồng thời hỗ trợ benchmark cả các checkpoint đơn codec (`Per-codec` / `Per-codec+Sev.`).
- **Bảng BDBR & LaTeX (`analysis/compute_bdbr.py`)**: Sửa lỗi lệch số cột LaTeX của `interpolated_cheng.tex` và hỗ trợ đầy đủ nhãn phương pháp cho cả Fig. 5 (`full`, `no_task`, `no_bg`) lẫn Table 3 (`family`, `severity`, `per_codec`, `per_codec_sev`).

---

## 4. Tương Thích Google Colab & Công Cụ Kiểm Định Tự Động

1. **An toàn AMP (`float16`) cho `Diff-JPEG` (`scope/codecs.py`)**: Bọc `_call_diffjpeg` trong `torch.amp.autocast("cuda", enabled=False)` ở `float32` giống `_call_bmshj` (ngăn lỗi `linalg_inv_out_cuda not implemented for 'Half'` trong `diff_jpeg/color.py` khi bật `--amp` trên GPU Colab).
2. **Tương thích `torch.load(..., weights_only=False)` trên PyTorch 2.6+**: Đồng bộ ở cả `scope/inference.py`, `scope/training.py` và `scope/evaluation/transmission.py`.
3. **Sửa lỗi chạy trực tiếp các script `tools/` & kiểm tra `bash` đa nền tảng**: `python tools/check_runtime.py`, `python tools/smoke_real.py`, `python tools/run_full_tests.py` và `python tools/validate_environment.py --mode static` đều chạy trực tiếp sạch sẽ cả trên Linux/Colab lẫn Windows (tự động bỏ qua stub WSL lỗi để dùng Git Bash thực).
4. **Bổ sung `requirements-colab.txt` & `notebooks/SCOPE_Colab.ipynb` (an toàn ABI NumPy 2.x trên Colab+)**: Giữ nguyên PyTorch CUDA và hệ sinh thái C-extension biên dịch theo `numpy>=2.0` (`opencv-python>=5.0`, `scipy`, `pandas`, `matplotlib`) có sẵn của Google Colab+; cài đặt các gói phụ trợ trước qua `requirements-colab.txt` rồi cài `compressai==1.2.8` với `--no-build-isolation --no-deps` (tránh ràng buộc cũ `numpy<2.0` trên PyPI gây hạ cấp NumPy và hỏng ABI mà không cần tạo lại build-env hay Restart Runtime), đồng thời khóa phiên bản tương thích (`numpy<2.0`, `opencv-python<4.12`, `kornia<0.9`, `omegaconf<2.4`, `iopath<0.1.10`, `cmake<4.0`) trong `requirements.txt` cho môi trường độc lập Python 3.10 + PyTorch 2.4.1. Đính kèm bảng ID tải tự động cả 11/11 checkpoint chính thức từ Google Drive và kiểm chứng sẵn các lệnh CLI.
