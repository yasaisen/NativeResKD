# NativeResKD

NativeResKD（NRKD，R27 MVLM 實驗代號 **4.01**）將 CONCH v1.5 的 Absolute Positional Embedding 移除，換成作用於 Q/K 的 learnable 2D Mixed RoPE，再以固定 512×512 Teacher 對真正 variable-length、可為長方形的 Student 做 representation KD。

本目錄保留原 `CONCHNativeRoPEKD` 的資料、模型、loss、representation fidelity 與 BRACS linear-probe 功能；訓練管理改用 CLEE 3.18 與 DRepGenVLM 3.19 共同模式：flat project JSON、timestamp run directory、`keepTrain`／`reTrain`、best/latest/periodic checkpoint、TensorBoard + JSONL/CSV、finite checks，以及 model/trainer 原子 checkpoint pair。原 `CONCHNativeRoPEKD` 不會被修改。

## 正式設定

- 專案：NativeResKD；縮寫：NRKD；版本：4.01。
- KD metadata：指定的 NAS6 train/test RepGenVLM JSON。
- 正式抽樣：train 40,000、valid 4,000、2 epochs。
- effective batch：32。
- 本機：1 GPU × per-device 1 × accumulation 32。
- HPC：4 GPUs × per-device 1 × accumulation 8。
- 每 epoch 1,250 optimizer steps，總數 2,500；與先前 20,000×4 epochs 設計具有相同 sample exposures 與 training-step 總數。
- Teacher：固定 512×512、凍結、`eval()`、`no_grad()`。
- Student：全參數訓練；32×32 到 24×24 的真實 variable patch grid，沒有 token/image padding 偽造的 sequence shortening。

正式設定檔：

- `NativeResKD/projects/NativeResKD_config.json`
- `NativeResKD/projects/NativeResKD_config_hpc.json`

相對路徑（例如 `projects/generated_kd_manifests`、`projects/bracs_metadata.csv`）以本 README 所在的 project root 為工作目錄。

## 環境與測試

```bash
conda activate MVLM_NRKD
cd /home/yasaisen/Desktop/27_research/research_main_v4/lab_01/NativeResKD

python -m pytest -q tests

python R27_MVLM_preflightNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json
```

Preflight 會驗證實際 checkpoint 架構、APE→RoPE surgery、四種 input shape、Teacher 無梯度、Student 全參數梯度、KD loss 與 metadata QC。需要較快的 smoke test 時可加：

```bash
--set train_size_expected=128 --set valid_size_expected=64 --set num_workers=0
```

## 訓練與恢復

本機單卡：

```bash
python R27_MVLM_trainNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json
```

HPC 四卡：

```bash
sbatch NRKD_train_4GPU.slurm
```

同一 run 精確續訓（`keepTrain`）：

```bash
python R27_MVLM_trainNRKD_v0.0.py \
  --latest-checkpoint-path /path/to/checkpoints/YYMMDDHHMMSS \
  --use-this-dir
```

只載入 Student、重置 optimizer/scheduler/epoch 並建立新 run（`reTrain`）：

```bash
python R27_MVLM_trainNRKD_v0.0.py \
  --best-checkpoint-path /path/to/checkpoints/YYMMDDHHMMSS \
  --retrain
```

每個 checkpoint 是三件式 pair：`[model]*.pth`、`[trainer]*.pth`、`[checkpoint]*.json`。Manifest 保存 checkpoint ID、檔案大小與 SHA-256；keepTrain 另外比對 metadata/model/loss/DDP topology/steps signature，並恢復各 rank 的 Python、NumPy、PyTorch、CUDA、DataLoader generator 與 sampler state。

## Representation fidelity

```bash
python R27_MVLM_evalRepresentationNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json \
  --model-id S0 --output-dir representation_eval/S0

python R27_MVLM_evalRepresentationNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json \
  --model-id S1 \
  --checkpoint /path/to/run/best_model.pth \
  --output-dir representation_eval/S1
```

`--checkpoint` 可指向未加 prefix 的虛擬名稱（如 `best_model.pth`）或實際 `[model]best_model.pth`。

## BRACS frozen linear probe

先建立官方 4,539 ROI 的 metadata：

```bash
python build_bracs_metadata.py \
  --root /media/yasaisen/NAS8/for_research/datasets/BRACS/BRACS_RoI/latest_version \
  --output projects/bracs_metadata.csv
```

之後對 T/S0/S1 分別執行 `matched_512` 與 `native_roi`：

```bash
python R27_MVLM_evalBRACSNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json \
  --model-id T --protocol matched_512

python R27_MVLM_evalBRACSNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json \
  --model-id S1 --protocol native_roi \
  --checkpoint /path/to/run/best_model.pth

python aggregate_bracs_results.py --root bracs_eval
```

Aggregation 會對兩個 protocol 分別執行三組 true-label 分層 paired bootstrap：

- `DeltaMacro-F1(T,S0) = Macro-F1(S0) - Macro-F1(T)`
- `DeltaMacro-F1(S0,S1) = Macro-F1(S1) - Macro-F1(S0)`
- `DeltaMacro-F1(T,S1) = Macro-F1(S1) - Macro-F1(T)`

並額外直接從不同 prediction directories 執行下列跨 protocol 比較：

- `DeltaMacro-F1(matched_512:T,native_roi:S1) = Macro-F1(native_roi:S1_native) - Macro-F1(matched_512:T)`

預設使用 10,000 次重抽、seed 42 與 percentile 95% CI。每次 bootstrap 都在同一組 sample indices 上同步計算兩個模型，並維持各 true-label class 的原始樣本數。Within-protocol `native_roi` 會將 `T_fixed512`、`S0_native`、`S1_native` 正規化為 T、S0、S1；cross-protocol 則明列 `protocol_a`、`protocol_b` 與實際 evaluated model ID。

若 coverage 不一致，within-protocol 統計使用 T/S0/S1 三模型共同成功的 test-sample intersection；cross-protocol 統計使用指定兩模型的共同成功 intersection。輸出會保留 test union、共同樣本數、各模型 usable 數與排除數，不會靜默把不同 cohort 的 Macro-F1 相減。

結果輸出為：

- `bracs_eval/paired_bootstrap_macro_f1.csv`
- `bracs_eval/paired_bootstrap_macro_f1.json`

CSV/JSON 以 `comparison_scope` 區分 `within_protocol` 與 `cross_protocol`，並保存 `protocol_a`、`protocol_b`。已有完整 prediction files 時，也可只重跑 paired bootstrap，不需重新抽取影像特徵：

```bash
python R27_MVLM_evalBRACSPairedBootstrapNRKD_v0.0.py \
  --root bracs_eval --iterations 10000 --seed 42 --confidence-level 0.95
```

BRACS 與 KD split 完全分離。Native ROI 報告另包含 coordinate range、sequence length >1024、coverage 與 OOM 分析。

## 重要 fail-fast 條件

- checkpoint 不是預期 ViT-L/16、24 layers 時停止。
- Student surgery 除 APE 外有任何 missing/unexpected/mismatched weight 時停止。
- KD 資料不是 Breast H&E、MPP 0.5、512×512，或路徑進入 BRACS/逃離 ROI root 時停止。
- train/valid sample 或 case overlap 時停止。
- effective batch 不等於 32（正式設定）、loss/gradient/parameter/optimizer state 非有限，或 DDP keepTrain topology/signature 改變時停止。

CONCH v1.5 權重仍受其原始授權與 gated-access 條款約束，本專案不散布模型權重。
