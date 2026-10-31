# NativeResKD

**繁體中文** | [English](README.md)

NativeResKD（NRKD，R27 MVLM 實驗 4.01）旨在將固定解析度的 CONCH v1.5 視覺編碼器轉換為可處理真正可變長度序列的編碼器。方法移除 Student 的 Absolute Positional Embedding（APE），在 attention 的 query 與 key 導入可學習的二維 Mixed RoPE，再由凍結於 512 × 512 的 Teacher 執行知識蒸餾（Knowledge Distillation, KD），以恢復原有表徵能力。

本實驗的核心問題是：RoPE Student 是否能在接受長方形、原生解析度 ROI 的同時保留 CONCH 的病理表徵能力，並推廣到遠長於 KD 訓練上限 1,024 spatial tokens 的序列。在 BRACS frozen linear-probe 評估中，訓練後 Student 恢復了 APE→RoPE 轉換造成的大幅性能損失，整體表現達到與 Teacher 相當的水準：

- Matched 512 × 512 Macro-F1：Teacher **0.5917**、未蒸餾 Student **0.5107**、蒸餾後 Student **0.6004**。
- Native ROI Macro-F1：固定 512 Teacher **0.5917**、未蒸餾 native Student **0.5236**、蒸餾後 native Student **0.6030**。
- 在兩種協定中，S1 相對 S0 的 paired-bootstrap 95% 信賴區間皆完全高於零。
- S1 相對 Teacher 的信賴區間則跨越零。因此，證據支持「恢復至 Teacher 水準」，尚不足以宣稱統計上優於 Teacher。

## 目錄

- [方法概述](#方法概述)
- [模型版本定義](#模型版本定義)
- [KD 資料與原生視野建構](#kd-資料與原生視野建構)
- [蒸餾目標函數](#蒸餾目標函數)
- [訓練設定與實際收斂結果](#訓練設定與實際收斂結果)
- [BRACS 評估設計](#bracs-評估設計)
- [BRACS 實驗結果](#bracs-實驗結果)
- [結果解讀與限制](#結果解讀與限制)
- [重現方式](#重現方式)
- [Repository 結構](#repository-結構)

## 方法概述

CONCH v1.5 使用具有 learned APE 的 ViT-L/16 image tower，因此受限於固定位置 grid。NativeResKD 保留固定解析度且凍結的 Teacher；Student 則複製 Teacher 權重、移除 APE，並以可學習的 2D Mixed RoPE 取代每一個 self-attention block 的位置處理方式。

```text
共享的 512 × 512 Breast H&E ROI
                |
          共享幾何增強
                |
        +-------+-----------------------+
        |                               |
凍結 Teacher                       可訓練 Student
APE、32 × 32 grid                  無 APE、24–32 × 24–32 grid
固定 1,024 patches                真實長方形序列
        |                               |
        +---- global / patch / hidden --+
                    cosine KD
```

實作遵守下列不變條件：

- Teacher 永遠維持 `eval()`、沒有可訓練參數，並在 `no_grad()` 下執行。
- Student 除移除 APE 外，其他初始權重皆須與 Teacher 相同；若有額外 missing、unexpected、shape mismatch 或數值不一致，程式會立即停止。
- Student attention 禁止 attention mask。序列縮短必須來自實際較小的影像視野，不能用 padding token 搭配 mask 偽裝。
- Patch correspondence 是顯式的：每個 Student patch 都保存其在 Teacher 32 × 32 grid 中的 `(y, x)` 座標。
- Prefix token 不套用 RoPE；RoPE 僅作用於 Q、K，不作用於 V。
- 訓練前會驗證 patch size 為 16 × 16、Transformer 深度為 24。

### CONCH backbone

本專案的 CONCH v1.5 image tower 為 ViT-L/16：

| 元件 | 設定 |
|---|---:|
| 輸入 anchor | 512 × 512 |
| Patch size | 16 × 16 |
| Teacher grid | 32 × 32 |
| Transformer 深度 | 24 |
| Embedding width | 1,024 |
| Attention heads | 16 |
| 公開影像特徵 | 768 維 attentional-pool 輸出 |

模型先依公開架構以 448 × 448 建立，Teacher APE 再以 bicubic interpolation 固定至 512 × 512，也就是由 28 × 28 插值成 32 × 32。完成後 Teacher 僅在此固定 grid 上運作。

### 可學習 2D Mixed RoPE

NativeResKD 為每一層 Transformer、每一個 attention head 建立混合 x/y rotary frequencies，初始化 seed 為 42、`theta = 10`。座標採 row-major 排列，並以 Student 當前長方形視野的局部座標為準。所有 frequencies 皆可學習，且使用獨立於 backbone 的 learning rate。

每個 patch 座標 `(x, y)` 會透過該 layer/head 的可學習軸向混合產生旋轉角，再以 cosine/sine 旋轉 Q、K 的 spatial-token pairs。由於座標是在執行時依實際 grid 產生，Student 不需要插值固定位置表，也能直接接受長方形 grid。

## 模型版本定義

結果報告使用三個模型狀態：

| ID | 定義 | 位置編碼 | 評估輸入 |
|---|---|---|---|
| **T** | 凍結的 CONCH v1.5 Teacher | 固定於 512 的 APE | 僅固定 512 × 512 |
| **S0** | 完成 Teacher→RoPE surgery、尚未 KD 的 Student | deterministic 初始化的可學習 2D Mixed RoPE | Matched 512 或 native ROI |
| **S1** | S0 完成兩個 epochs representation KD 後的模型 | 已訓練 2D Mixed RoPE，Student 全參數 fine-tune | Matched 512 或 native ROI |

S0 是重要的 control。其下游性能反映移除 APE、加入尚未訓練之 rotary frequencies 所造成的即時表徵破壞；S1 則量測 KD 能修復多少此類損失。

## KD 資料與原生視野建構

### 資料來源與抽樣

正式設定從指定的 NAS6 RepGenVLM metadata 選取 Breast H&E ROI：

| Split | 候選 ROI | 選取 ROI | 涵蓋 cases |
|---|---:|---:|---:|
| Train | 47,460 | 40,000 | 911 |
| Validation | 12,234 | 4,000 | 226 |

抽樣策略為 `case_aware_deterministic_round_robin`，seed 42；保存的 manifest 顯示沒有樣本因 QC 被跳過。每一個接受的 ROI 必須符合：

- Breast 組織與 H&E stain；
- 20×、MPP 0.5；
- 尺寸必須正好為 512 × 512，禁止 resize；
- 路徑必須位於指定 ROI root；
- 路徑不得位於 BRACS forbidden root；
- HSV-saturation Otsu tissue mask 不得為空。

程式同時檢查 train/validation 的 sample 與 case 均不重疊。BRACS 被明確排除於 KD metadata 之外，使下游評估與表徵訓練維持分離。

### 共享資料增強

Teacher 與 Student 先接受相同、依 epoch 決定且可重現的增強：

- horizontal flip；
- vertical flip；
- 隨機 0°、90°、180° 或 270° rotation。

完成共享轉換後才抽取 Student view，因此 Teacher target 與 Student 保留的 patches 仍有正確空間對應。

### Student views

每一筆訓練樣本從三類視野中抽樣：

| View | 抽樣機率 | 行為 |
|---|---:|---|
| Full | 0.30 | 保留完整 32 × 32 grid。 |
| Background trim | 0.50 | 反覆移除背景比例至少 95% 的邊界 row/column，任一軸不得少於 24 patches；若無法移除則退回 full。 |
| Lossy trim | 0.20 | 抽取 24–32 × 24–32 長方形 grid，並選擇保留最多組織量的 crop。 |

當 patch 的 tissue fraction 小於 0.05 時視為背景。Student 的訓練序列長度因此介於 576–1,024 spatial tokens，且可以是長方形。程式不會以 image/token padding 偽造較短序列。

## 蒸餾目標函數

模型在最終公開 image feature，以及第 6、12、18、24 層 Transformer 表徵上比較 Teacher 與 Student。令 cosine distance 為 `d(a,b) = 1 - cos(a,b)`，總 loss 為：

```text
L = 1.00 L_global + 0.50 L_patch + 0.25 L_hidden
```

- `L_global`：對齊 768 維、post-final-normalization、pre-projection 的 CONCH image feature。
- `L_patch`：依顯式 mapping 對齊每個 Student patch 與對應 Teacher patch，並對四個指定層取平均。
- `L_hidden`：對齊相同四層的 pooled hidden representation。

只有 lossy crop 的逐樣本 global semantic distance 會乘上 `tissue_retention_ratio²`；patch 與 hidden target 不加權。Loss 實作會對每個指定層與最終 objective 執行 finite-value checks。

## 訓練設定與實際收斂結果

### 正式最佳化設定

| 設定 | 數值 |
|---|---:|
| Epochs | 2 |
| Train / validation samples | 40,000 / 4,000 |
| Effective batch size | 32 |
| 本機 run | 1 GPU × batch 1 × accumulation 32 |
| HPC 替代設定 | 4 GPUs × batch 1 × accumulation 8 |
| Optimizer | AdamW |
| Backbone learning rate | 1 × 10⁻⁵ |
| RoPE learning rate | 1 × 10⁻⁴ |
| Weight decay | 0.05 |
| Adam betas | (0.9, 0.95) |
| Precision | bfloat16 autocast |
| Gradient clipping | global norm 1.0 |
| Scheduler | linear warm-up 後接 cosine annealing |
| Warm-up | 2,500 optimizer steps 中的前 125 steps（5%） |
| Seed | 42，deterministic mode |

每個 epoch 有 1,250 optimizer steps，兩個 epochs 共 2,500 optimizer steps；以樣本計算共處理 80,000 次訓練輸入。

### 對應的正式訓練 run

本報告將提供的 BRACS 結果與 run `260830105807` 一併分析：

| Provenance 項目 | 數值 |
|---|---|
| Host / GPU | `MISAKA-gpu24`、單張 NVIDIA GeForce RTX 4090 |
| Python / PyTorch | 3.10.21 / 2.6.0+cu124 |
| CUDA / cuDNN | 12.4 / 90100 |
| Best checkpoint | 第 2 epoch（`epoch_idx=1`）、optimizer step 2,500 |
| Best checkpoint ID | `d888a8b34b0649f18a5e52b53649d522` |
| Best model SHA-256 | `360c1edf8e43d47dbe6f97fa7c76649f4e0fa8bfb0d05487bf200219a7affbac` |
| 記錄的 Git commit | 無（`null`） |

BRACS 結果檔本身沒有保存 checkpoint 路徑或 hash。因此，上述內容是此次提供的關聯訓練 provenance，但無法只由 `bracs_eval` 獨立驗證結果與 checkpoint 的密碼學連結。

### 訓練與驗證軌跡

標準訓練迴圈中的 validation 使用 deterministic 完整 32 × 32 Student view。這是 representation-KD validation，不是 BRACS classification 結果。

| Epoch | Train total loss | Validation total loss | Validation global cosine | Validation patch cosine | Validation hidden cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.138934 | 0.087241 | 0.962313 | 0.920846 | 0.960090 |
| 2 | 0.079669 | **0.077568** | **0.967033** | **0.928742** | **0.964112** |

第 2 epoch 的 validation objective 較低，因此被選為正式 best checkpoint。這些 epoch-level 數值顯示第二個 epoch 仍持續改善 Teacher/Student 表徵對齊，但不能單獨證明 native-resolution generalization；後者由 BRACS 下游評估檢驗。

## BRACS 評估設計

### Dataset 與 labels

評估使用 BRACS 官方 ROI split tree，共 4,539 ROIs、七個類別：ADH、DCIS、FEA、IC、N、PB、UDH。

| Split | 總數 | ADH | DCIS | FEA | IC | N | PB | UDH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 3,657 | 387 | 665 | 624 | 521 | 357 | 714 | 389 |
| Validation | 312 | 41 | 40 | 49 | 47 | 46 | 43 | 46 |
| Test | 570 | 79 | 85 | 83 | 81 | 81 | 79 | 82 |

所有 ROI 都先以 Lanczos 從 40×物理重採樣至 20×，再套用各協定的空間處理方式。

### 輸入協定

| Protocol | Teacher | Student | 目的 |
|---|---|---|---|
| `matched_512` | 中央裁切並對稱白色 padding 至 512 × 512，絕不 resize | 相同處理 | 在完全一致的輸入支援範圍下比較架構。 |
| `native_roi` | 仍以固定 512 × 512 評估，輸出 ID 為 `T_fixed512` | 保留完整 20× ROI，只在右側／下側補白至 16 的倍數 | 測試原生視野與序列長度外插。 |

`T_fixed512` 刻意與 `matched_512/T` 完全相同，只是在 native-ROI 結果命名空間中重複作為固定解析度基準。Teacher 因 APE 固定於 32 × 32，並沒有在原生可變尺寸 grid 上執行。

### Frozen linear probe

每一個 model/protocol 組合皆執行：

1. 凍結 image encoder，分別為 train、validation、test 抽取 768 維 `global_preproj` feature。
2. 使用官方 train split 訓練未加 class weight、`lbfgs` solver 的 multinomial logistic regression。
3. 依 validation Macro-F1 從 `{0.01, 0.1, 1.0, 10.0}` 選擇 `C`；完全同分時選較小的 `C`。
4. 最後只在官方 test split 評估一次。

六個評估均選到 `C = 1.0`。Primary metric 為 Macro-F1；結果也包含 balanced accuracy、accuracy、one-vs-rest Macro-AUROC、per-class F1、confusion matrix、coverage 與 OOM count。

### Paired bootstrap

不確定性使用 10,000 次 true-label-stratified paired percentile bootstrap、seed 42 與 95% 信賴區間估計。同一次重抽中，兩個模型使用完全相同的 test indices，並維持各類別原始樣本數。

同協定比較使用 T/S0/S1 三者共同成功的 test intersection：`matched_512` 有 570 筆，`native_roi` 因 S0 有五筆 test OOM 而剩 565 筆。Matched T 與 native S1 的直接跨協定比較則有完整 570 筆共同成功樣本。

## BRACS 實驗結果

### 整體表現

下表的 Macro-F1、balanced accuracy、accuracy 與 AUROC 皆以成功完成 feature extraction 的 test samples 計算；`Coverage-adjusted accuracy` 則額外將失敗樣本視為錯誤。

| Protocol | Model | Macro-F1 | Balanced accuracy | Accuracy | Macro-AUROC | Test coverage | Coverage-adjusted accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Matched 512 | S0 | 0.5107 | 0.5199 | 0.5228 | 0.8541 | 1.0000 | 0.5228 |
| Matched 512 | S1 | **0.6004** | **0.5993** | **0.6000** | **0.9021** | 1.0000 | **0.6000** |
| Matched 512 | T | 0.5917 | 0.5963 | 0.5982 | 0.8936 | 1.0000 | 0.5982 |
| Native ROI | S0 native | 0.5236 | 0.5323 | 0.5310 | 0.8583 | 0.9912 | 0.5263 |
| Native ROI | S1 native | **0.6030** | **0.6062** | **0.6070** | **0.9083** | 1.0000 | **0.6070** |
| Native ROI | T fixed 512 | 0.5917 | 0.5963 | 0.5982 | 0.8936 | 1.0000 | 0.5982 |

Matched input 下，S1 相對 S0 提升 0.0896 Macro-F1；若直接比較兩個 native 評估各自完整可用 cohort 的點估計，提升為 0.0794。正式推論比較則使用下一節明確定義的共同 cohort。

### Paired-bootstrap 比較

所有 delta 均定義為 `Macro-F1(model B) - Macro-F1(model A)`。

| 範圍 | 比較 | 共同 test n | Delta Macro-F1 | 95% CI | CI 是否排除 0？ |
|---|---|---:|---:|---:|---:|
| Matched 512 | S0 − T | 570 | −0.0810 | [−0.1249, −0.0382] | 是 |
| Matched 512 | S1 − S0 | 570 | **+0.0896** | **[+0.0470, +0.1326]** | **是** |
| Matched 512 | S1 − T | 570 | +0.0086 | [−0.0275, +0.0448] | 否 |
| Native ROI | S0 − T | 565 | −0.0747 | [−0.1177, −0.0323] | 是 |
| Native ROI | S1 − S0 | 565 | **+0.0860** | **[+0.0481, +0.1243]** | **是** |
| Native ROI | S1 − T | 565 | +0.0112 | [−0.0301, +0.0529] | 否 |
| 跨協定 | S1 native − T matched | 570 | +0.0112 | [−0.0295, +0.0528] | 否 |

S1 相對 S0 的 paired evidence 十分明確：KD 修復了 positional-encoding surgery 所造成的大量表徵損失。S1 的點估計雖略高於 Teacher，但信賴區間同時容許小幅退步與小幅改善。

### Per-class F1

類別順序為 evaluator 保存的排序後 label order。

| Protocol / model | ADH | DCIS | FEA | IC | N | PB | UDH |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matched T | 0.3824 | **0.6162** | 0.7251 | 0.8718 | 0.6628 | 0.4678 | 0.4161 |
| Matched S0 | 0.2114 | 0.5464 | 0.7225 | 0.7660 | 0.5926 | 0.3892 | 0.3472 |
| Matched S1 | **0.4416** | 0.5422 | **0.7602** | **0.8758** | **0.6909** | 0.4607 | **0.4314** |
| Native T fixed 512 | 0.3824 | **0.6162** | 0.7251 | 0.8718 | 0.6628 | 0.4678 | **0.4161** |
| Native S0 | 0.2270 | 0.5283 | 0.6771 | 0.8811 | 0.6047 | 0.4693 | 0.2778 |
| Native S1 | **0.4027** | 0.5476 | **0.7470** | **0.8987** | **0.7263** | **0.4908** | 0.4076 |

Matched S1 相對 S0 最大的恢復出現在 ADH，其次為 UDH、N 與 IC。S1 並未在所有類別都高於 T：兩種 S1 協定的 DCIS 都較低；native S1 則在 FEA、IC、N 與 PB 最高。這些是描述性 class-level 結果，實驗沒有計算逐類別信賴區間或 multiple-comparison correction。

### Native-ROI 外插

KD 訓練視野限制為任一軸最多 32 patches、序列最多 1,024 spatial tokens；Native BRACS test ROI 明顯超出此範圍：

- 401/570 個 test ROI 至少一個 grid 軸大於 32，屬於 `coordinate_extrapolation`。
- 362/570 個 test ROI 超過 1,024 spatial tokens，屬於 `sequence_extrapolation`。
- Test sequence length 範圍為 36–55,685 tokens，中位數為 1,630。

蒸餾後 Student 的 subgroup 結果如下：

| 分組 | Group | n | Coverage | Macro-F1 | Accuracy |
|---|---|---:|---:|---:|---:|
| Coordinate | 範圍內 | 169 | 1.0000 | 0.5304 | 0.5680 |
| Coordinate | 外插 | 401 | 1.0000 | **0.6067** | **0.6234** |
| Sequence | 範圍內 | 208 | 1.0000 | 0.5524 | 0.5817 |
| Sequence | 外插 | 362 | 1.0000 | **0.5991** | **0.6215** |

不能把 subgroup 差異解讀為「較長序列本身提高準確率」；不同 group 的 ROI 尺寸、組織內容、診斷組成與難度皆可能不同。合理結論較窄：在此次執行中，S1 對大量超出 KD 座標與序列範圍的 grid 仍能完成預測，且維持具實用性的分類表現。

### Feature extraction coverage 與 OOM

Native feature extraction 在不同模型執行中並沒有完全相同的成功率：

| Model | Train 成功 | Validation 成功 | Test 成功 |
|---|---:|---:|---:|
| S0 native | 3,497 / 3,657（95.63%） | 304 / 312（97.44%） | 565 / 570（99.12%） |
| S1 native | 3,651 / 3,657（99.84%） | 312 / 312（100%） | 570 / 570（100%） |
| T fixed 512 | 3,657 / 3,657（100%） | 312 / 312（100%） | 570 / 570（100%） |

結果 JSON 只摘要 test coverage；上表 train/validation 數量由保存的 prediction records 統計。由於 S0 與 S1 架構相同，單純的權重訓練並不是改變 attention 理論記憶體需求的合理機制。兩者 OOM 數量不同應視為 execution-level observation，可能受到 allocator state、可用 GPU memory、kernel 或執行條件影響，不能作為 KD 改善架構記憶體效率的證據。

模型各自缺失的 train/validation features 也表示 native S0 與 S1 probe 是在略有不同的成功 cohort 上訓練及選擇。Paired bootstrap 能對齊成功的 test samples，但無法消除上游 probe-training cohort 的差異。

## 結果解讀與限制

### 實驗支持的結論

1. **未經調適地移除 APE 會明顯破壞表徵。** 在 matched 512 input 下，S0 顯著低於 T，隔離出 positional-encoding conversion 的代價。
2. **Representation KD 能修復此破壞。** S1 在 matched 與 native 協定都顯著高於 S0，且訓練期間 validation cosine similarities 持續上升。
3. **Student 保留 Teacher-level linear-probe performance。** S1 的 Macro-F1 點估計略高於 T，但 paired CI 包含零。
4. **模型可執行遠超 KD token range 的輸入。** 保存結果中，native S1 完成全部 570 個 test ROI，最長序列達 55,685 patches。
5. **Native context 是可行的，但尚未隔離其因果效益。** S1-native 相對 T-matched 的跨協定估計為正但不確定，且同時改變模型狀態與可見視野。

### 重要限制

- 本報告只有一次 KD run 與一個 BRACS split，未量測跨 seed、跨 run 或跨 split 變異。
- Bootstrap 只估計固定 probe 下的 test-sample uncertainty，沒有重做 feature extraction、probe fitting、hyperparameter selection 或 KD training。
- 沒有 native T；`T_fixed512` 是固定 grid 基準，因此 native S1 與 T 的比較同時改變 encoder state 與可見 context。
- Native S0/S1 因 OOM 使用了不同的成功 train/validation cohorts。
- OOM 具有硬體與執行條件依賴性，本實驗沒有受控的記憶體 benchmark。
- 外插 subgroup 結果為描述性分析，並受 subgroup composition 混淆。
- 沒有提供完整 validation set 的正式 representation-fidelity 結果；現有單樣本 smoke outputs 不被當作正式證據。
- 關聯訓練 run 的 `git_commit` 為 `null`，BRACS artifacts 也沒有保存 S1 checkpoint hash。
- 尚無外部測試集、calibration、臨床效益研究或病理醫師比較。本專案是 representation-learning 實驗，不是臨床醫療器材。

## 重現方式

### 環境

```bash
conda activate MVLM_NRKD
cd /home/yasaisen/Desktop/27_research/research_main_v4/lab_01/NativeResKD
pip install -r requirements.txt
```

CONCH v1.5 權重必須存在於設定檔指定位置。本專案不散布權重；權重仍受原始授權與 gated-access 條款約束。

### Preflight

Preflight 會檢查 checkpoint architecture、APE→RoPE surgery、Teacher/Student gradients、單一 batch KD loss、metadata constraints，以及 512 × 512、384 × 512、512 × 384、384 × 384 Student inputs。

```bash
python R27_MVLM_preflightNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json
```

小型 data-path smoke test：

```bash
python R27_MVLM_preflightNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json \
  --set train_size_expected=128 \
  --set valid_size_expected=64 \
  --set num_workers=0
```

### 訓練

本機單 GPU：

```bash
python R27_MVLM_trainNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json
```

HPC 四 GPU：

```bash
sbatch NRKD_train_4GPU.slurm
```

精確續接同一個 run（`keepTrain`）：

```bash
python R27_MVLM_trainNRKD_v0.0.py \
  --latest-checkpoint-path /path/to/checkpoints/YYMMDDHHMMSS \
  --use-this-dir
```

只載入 Student，重建 optimizer/scheduler 並開始新 run（`reTrain`）：

```bash
python R27_MVLM_trainNRKD_v0.0.py \
  --best-checkpoint-path /path/to/checkpoints/YYMMDDHHMMSS \
  --retrain
```

每一個 checkpoint 是原子化三件式組合：`[model]*.pth`、`[trainer]*.pth`、`[checkpoint]*.json`。Manifest 保存 checkpoint ID、檔案大小與 SHA-256；精確續訓還會核對 metadata/model/loss/DDP/step signatures，並恢復各 rank 的 Python、NumPy、PyTorch、CUDA、DataLoader generator 與 sampler states。

### Representation fidelity

```bash
python R27_MVLM_evalRepresentationNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json \
  --model-id S0 \
  --output-dir representation_eval/S0

python R27_MVLM_evalRepresentationNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json \
  --model-id S1 \
  --checkpoint '/path/to/run/[model]best_model.pth' \
  --output-dir representation_eval/S1
```

Evaluator 會報告 full、background-trimmed，以及從 24 × 24 至 32 × 32 的九種 deterministic lossy grids，包含 global、projected、hidden-global 與 mapped-patch cosine statistics。

### BRACS metadata

```bash
python build_bracs_metadata.py \
  --root /path/to/BRACS/BRACS_RoI/latest_version \
  --output projects/bracs_metadata.csv
```

Builder 要求官方 `train`、`val`、`test` 目錄，並記錄每個 ROI 的尺寸、類別、split 與來源倍率。

### BRACS 評估

Matched Teacher 與 Students：

```bash
python R27_MVLM_evalBRACSNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json \
  --model-id T --protocol matched_512 --output-dir bracs_eval

python R27_MVLM_evalBRACSNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json \
  --model-id S0 --protocol matched_512 --output-dir bracs_eval

python R27_MVLM_evalBRACSNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json \
  --model-id S1 --protocol matched_512 \
  --checkpoint '/path/to/run/[model]best_model.pth' \
  --output-dir bracs_eval
```

Native-ROI 評估：

```bash
python R27_MVLM_evalBRACSNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json \
  --model-id T --protocol native_roi --output-dir bracs_eval

python R27_MVLM_evalBRACSNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json \
  --model-id S0 --protocol native_roi --output-dir bracs_eval

python R27_MVLM_evalBRACSNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json \
  --model-id S1 --protocol native_roi \
  --checkpoint '/path/to/run/[model]best_model.pth' \
  --output-dir bracs_eval
```

彙整 metrics 並重新計算 paired bootstrap：

```bash
python aggregate_bracs_results.py \
  --root bracs_eval \
  --bootstrap-iterations 10000 \
  --bootstrap-seed 42 \
  --bootstrap-confidence-level 0.95
```

若 predictions 已存在，可不重新抽取 feature，直接重跑 bootstrap：

```bash
python R27_MVLM_evalBRACSPairedBootstrapNRKD_v0.0.py \
  --root bracs_eval \
  --iterations 10000 \
  --seed 42 \
  --confidence-level 0.95
```

## Repository 結構

```text
NativeResKD/
├── NativeResKD/
│   ├── configs/                 # Flat 4.01 configuration 與驗證
│   ├── criterions/              # Global、patch、hidden representation KD
│   ├── datasets/                # KD/BRACS datasets 與 shape bucketing
│   ├── evaluator/               # BRACS、fidelity、paired-bootstrap 評估
│   ├── models/                  # CONCH loader、APE→RoPE surgery、Mixed RoPE
│   ├── processors/              # Tissue masks 與 native-view construction
│   ├── projects/                # 本機與 HPC resolved configs
│   └── trainer/                 # DDP training、scheduler、checkpoint policy
├── R27_MVLM_trainNRKD_v0.0.py
├── R27_MVLM_validNRKD_v0.0.py
├── R27_MVLM_preflightNRKD_v0.0.py
├── R27_MVLM_evalRepresentationNRKD_v0.0.py
├── R27_MVLM_evalBRACSNRKD_v0.0.py
├── R27_MVLM_evalBRACSPairedBootstrapNRKD_v0.0.py
├── aggregate_bracs_results.py
├── build_bracs_metadata.py
├── NRKD_train_4GPU.slurm
└── requirements.txt
```

每一個 BRACS evaluation directory 包含 `metrics.json`、`predictions.csv` 與 `extrapolation_analysis.csv`。Aggregation 會產生 `summary.csv`、`per_class_metrics.csv`、`native_coverage.csv`、`paired_bootstrap_macro_f1.csv` 及其 JSON 版本。

## 授權與權重

Repository code 的授權請見 [LICENSE](LICENSE)。CONCH v1.5 權重不包含在本專案內，仍受其原始授權及存取條款約束；BRACS 資料亦不由本專案重新散布。使用者須自行依適用條款取得資料與模型權重。
