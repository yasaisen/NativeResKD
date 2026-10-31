# NativeResKD

[繁體中文](README.zh-TW.md) | **English**

NativeResKD (NRKD, R27 MVLM experiment 4.01) converts the fixed-resolution CONCH v1.5 vision encoder into a genuinely variable-length encoder. It removes the student's absolute positional embedding (APE), introduces learnable two-dimensional Mixed RoPE on attention queries and keys, and then restores the representation with knowledge distillation (KD) from a frozen 512 × 512 teacher.

The central question is whether a RoPE student can retain CONCH's pathology representation while accepting rectangular native-resolution regions of interest (ROIs), including sequences far longer than the 1,024 spatial tokens seen during KD. On the BRACS frozen linear-probe benchmark, the trained student recovered the large loss caused by the untrained APE-to-RoPE conversion and reached teacher-level performance:

- Matched 512 × 512 Macro-F1: teacher **0.5917**, untrained student **0.5107**, trained student **0.6004**.
- Native-ROI Macro-F1: fixed-512 teacher **0.5917**, untrained native student **0.5236**, trained native student **0.6030**.
- The trained-vs-untrained improvements were positive under both protocols, with paired-bootstrap 95% confidence intervals excluding zero.
- The trained-vs-teacher confidence intervals included zero. The evidence therefore supports recovery to teacher-level performance, not a statistically established improvement over the teacher.

## Contents

- [Method overview](#method-overview)
- [Model variants](#model-variants)
- [KD data and native-view construction](#kd-data-and-native-view-construction)
- [Distillation objective](#distillation-objective)
- [Training configuration and observed convergence](#training-configuration-and-observed-convergence)
- [BRACS evaluation design](#bracs-evaluation-design)
- [BRACS results](#bracs-results)
- [Interpretation and limitations](#interpretation-and-limitations)
- [Reproduction](#reproduction)
- [Repository layout](#repository-layout)

## Method overview

CONCH v1.5 uses a ViT-L/16 image tower with a learned APE and is therefore tied to a fixed positional grid. NativeResKD keeps a frozen fixed-resolution teacher and constructs a student by copying the teacher, unregistering the student's APE, and replacing positional handling inside every self-attention block with learnable 2D Mixed RoPE.

```text
shared 512 × 512 Breast H&E ROI
                |
        shared geometric augmentation
                |
        +-------+-----------------------+
        |                               |
frozen teacher                    trainable student
APE, 32 × 32 grid                 no APE, 24–32 × 24–32 grid
fixed 1,024 patches               true rectangular sequence
        |                               |
        +---- global / patch / hidden --+
                    cosine KD
```

The implementation has the following invariants:

- The teacher is always in `eval()` mode, has no trainable parameters, and runs under `no_grad()`.
- The student is initialized from all teacher weights except APE; initialization verification fails if any other key is missing, unexpected, shape-mismatched, or value-mismatched.
- Student attention rejects attention masks. Sequence reduction must come from an actual smaller image field of view, not padded tokens hidden behind a mask.
- Patch correspondence is explicit: every student patch stores its `(y, x)` location in the teacher's 32 × 32 grid.
- Prefix tokens are left unchanged by RoPE. RoPE is applied to Q and K, but not V.
- The code validates a 16 × 16 patch size and a 24-layer transformer before training.

### CONCH backbone

The local CONCH v1.5 construction is a ViT-L/16 with:

| Component | Setting |
|---|---:|
| Input anchor | 512 × 512 |
| Patch size | 16 × 16 |
| Teacher grid | 32 × 32 |
| Transformer depth | 24 |
| Embedding width | 1,024 |
| Attention heads | 16 |
| Public image feature | 768-dimensional attentional-pool output |

The checkpoint is constructed at its public 448 × 448 size and the teacher APE is bicubically anchored to 512 × 512 (28 × 28 to 32 × 32). The resulting teacher remains fixed at that grid.

### Learnable 2D Mixed RoPE

For every transformer layer and attention head, NativeResKD initializes mixed x/y rotary frequencies with seed 42 and `theta = 10`. Coordinates are row-major and local to the student's current rectangular field of view. Frequencies are learnable and are optimized separately from the rest of the backbone.

For a patch coordinate `(x, y)`, each layer/head forms an angle from a learned mixture of the two axes. The resulting cosine and sine rotate Q and K spatial-token pairs. Because coordinates are generated at runtime, the student does not require interpolation of a stored positional table and can process rectangular grids.

## Model variants

The result report uses three model states:

| ID | Definition | Positional encoding | Input used in evaluation |
|---|---|---|---|
| **T** | Frozen CONCH v1.5 teacher | 512-anchored APE | Fixed 512 × 512 only |
| **S0** | Student immediately after teacher-to-RoPE surgery, before KD | Learnable 2D Mixed RoPE at deterministic initialization | Matched 512 or native ROI |
| **S1** | S0 after the two-epoch representation-KD run | Trained 2D Mixed RoPE; full student fine-tuned | Matched 512 or native ROI |

S0 is an important control. Its poor downstream result measures the immediate representation disruption caused by removing APE and introducing untrained rotary frequencies. S1 then measures how much of that disruption the KD procedure repairs.

## KD data and native-view construction

### Source data and selection

The formal configuration draws Breast H&E ROIs from the specified NAS6 RepGenVLM metadata:

| Split | Candidate ROIs | Selected ROIs | Cases represented |
|---|---:|---:|---:|
| Train | 47,460 | 40,000 | 911 |
| Validation | 12,234 | 4,000 | 226 |

Selection uses `case_aware_deterministic_round_robin` with seed 42. The saved manifests report no skipped samples. Every accepted ROI must be:

- Breast tissue with H&E staining;
- 20×, MPP 0.5;
- exactly 512 × 512 pixels, with resizing forbidden;
- inside the configured ROI root;
- outside the forbidden BRACS root; and
- non-empty under the HSV-saturation Otsu tissue mask.

Training and validation are checked for both sample-level and case-level overlap. BRACS is explicitly excluded from KD metadata, keeping downstream evaluation separate from representation training.

### Shared augmentation

Teacher and student first receive the same deterministic per-epoch augmentation:

- horizontal flip;
- vertical flip; and
- a random multiple of 90° rotation.

Only after this shared transformation is the student view selected. This preserves spatial correspondence between the teacher target and the student's retained patches.

### Student views

Each training sample chooses one of three view families:

| View | Sampling probability | Behavior |
|---|---:|---|
| Full | 0.30 | Keep the complete 32 × 32 grid. |
| Background trim | 0.50 | Iteratively remove border rows/columns that are at least 95% background, without going below 24 patches on either axis; fall back to full if nothing can be removed. |
| Lossy trim | 0.20 | Sample a rectangular 24–32 × 24–32 grid and choose the crop that retains the greatest tissue mass. |

A patch is considered background when its tissue fraction is below 0.05. The resulting training sequence has 576–1,024 spatial tokens and can be rectangular. No token/image padding is used to pretend that a full image has a shorter sequence.

## Distillation objective

The model compares teacher and student representations at the final public image feature and at transformer layers 6, 12, 18, and 24. For cosine distance `d(a,b) = 1 - cos(a,b)`, the total objective is:

```text
L = 1.00 L_global + 0.50 L_patch + 0.25 L_hidden
```

- `L_global` aligns the 768-dimensional post-final-normalization, pre-projection CONCH image feature.
- `L_patch` aligns each student patch with its explicitly mapped teacher patch and averages over the four selected layers.
- `L_hidden` aligns the pooled hidden representation at the same four layers.

For lossy crops only, each sample's global semantic distance is weighted by the square of its retained-tissue ratio. Patch and hidden targets remain unweighted. The loss implementation performs finite-value checks at every selected layer and on the final objective.

## Training configuration and observed convergence

### Formal optimization setup

| Setting | Value |
|---|---:|
| Epochs | 2 |
| Train / validation samples | 40,000 / 4,000 |
| Effective batch size | 32 |
| Local run | 1 GPU × batch 1 × accumulation 32 |
| HPC alternative | 4 GPUs × batch 1 × accumulation 8 |
| Optimizer | AdamW |
| Backbone learning rate | 1 × 10⁻⁵ |
| RoPE learning rate | 1 × 10⁻⁴ |
| Weight decay | 0.05 |
| Adam betas | (0.9, 0.95) |
| Precision | bfloat16 autocast |
| Gradient clipping | global norm 1.0 |
| Scheduler | linear warm-up followed by cosine annealing |
| Warm-up | 125 of 2,500 optimizer steps (5%) |
| Seed | 42, deterministic mode |

There are 1,250 optimizer steps per epoch and 2,500 total optimizer steps. At sample level, the run processes 80,000 training examples over two epochs.

### Associated formal run

The supplied BRACS results were analyzed alongside run `260830105807`:

| Provenance item | Value |
|---|---|
| Host / GPU | `MISAKA-gpu24`, one NVIDIA GeForce RTX 4090 |
| Python / PyTorch | 3.10.21 / 2.6.0+cu124 |
| CUDA / cuDNN | 12.4 / 90100 |
| Best checkpoint | epoch 2 (`epoch_idx=1`), optimizer step 2,500 |
| Best checkpoint ID | `d888a8b34b0649f18a5e52b53649d522` |
| Best model SHA-256 | `360c1edf8e43d47dbe6f97fa7c76649f4e0fa8bfb0d05487bf200219a7affbac` |
| Recorded Git commit | unavailable (`null`) |

The BRACS result files do not embed the checkpoint path or hash. The run above is therefore the supplied associated training provenance, but the exact result-to-checkpoint link cannot be independently verified from `bracs_eval` alone.

### Training and validation trajectory

Validation in the standard training loop uses the deterministic full 32 × 32 student view. It is a representation-KD validation result, not a BRACS classification result.

| Epoch | Train total loss | Validation total loss | Validation global cosine | Validation patch cosine | Validation hidden cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.138934 | 0.087241 | 0.962313 | 0.920846 | 0.960090 |
| 2 | 0.079669 | **0.077568** | **0.967033** | **0.928742** | **0.964112** |

The lower validation objective at epoch 2 selected the formal best checkpoint. These epoch-aggregated values show continued representation alignment over the second epoch. They do not, by themselves, establish native-resolution generalization; that question is evaluated downstream on BRACS.

## BRACS evaluation design

### Dataset and labels

Evaluation uses the official BRACS ROI split tree with 4,539 ROIs and seven labels: ADH, DCIS, FEA, IC, N, PB, and UDH.

| Split | Total | ADH | DCIS | FEA | IC | N | PB | UDH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 3,657 | 387 | 665 | 624 | 521 | 357 | 714 | 389 |
| Validation | 312 | 41 | 40 | 49 | 47 | 46 | 43 | 46 |
| Test | 570 | 79 | 85 | 83 | 81 | 81 | 79 | 82 |

All ROIs are physically resampled from 40× to 20× with Lanczos before applying a protocol-specific spatial policy.

### Input protocols

| Protocol | Teacher | Student | Purpose |
|---|---|---|---|
| `matched_512` | Center crop and symmetric white pad to 512 × 512; never resize | Same | Architecture comparison under identical input support. |
| `native_roi` | Still evaluated at fixed 512 × 512 and reported as `T_fixed512` | Preserve the complete 20× ROI; white-pad only the bottom/right edge to a multiple of 16 | Test native field of view and sequence-length extrapolation. |

`T_fixed512` is intentionally identical to `matched_512/T`; it is repeated under the native-ROI result namespace as the fixed-resolution reference. The teacher is not run on a native variable-size grid because its APE is anchored to 32 × 32.

### Frozen linear probe

For every model/protocol combination:

1. Freeze the image encoder and extract its 768-dimensional `global_preproj` feature independently for train, validation, and test.
2. Train an unweighted multinomial logistic-regression classifier with the `lbfgs` solver on the official train split.
3. Select `C` from `{0.01, 0.1, 1.0, 10.0}` by validation Macro-F1, choosing the smaller `C` on an exact tie.
4. Evaluate once on the official test split.

All six evaluations selected `C = 1.0`. The primary metric is Macro-F1; the artifacts also report balanced accuracy, accuracy, one-vs-rest Macro-AUROC, per-class F1, confusion matrices, coverage, and OOM counts.

### Paired bootstrap

Uncertainty is estimated with 10,000 true-label-stratified paired percentile bootstrap iterations, seed 42, and 95% confidence intervals. Both models in a comparison use the same resampled test indices, while each class retains its original sample count.

Within a protocol, the code uses the common successful T/S0/S1 test intersection. This gives 570 samples for `matched_512` and 565 for `native_roi`, because S0 had five test-time OOMs. The direct cross-protocol comparison of matched T and native S1 uses all 570 common successful samples.

## BRACS results

### Overall performance

Macro-F1, balanced accuracy, accuracy, and AUROC below are computed on successfully encoded test samples. `Coverage-adjusted accuracy` additionally treats unsuccessful test samples as incorrect.

| Protocol | Model | Macro-F1 | Balanced accuracy | Accuracy | Macro-AUROC | Test coverage | Coverage-adjusted accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Matched 512 | S0 | 0.5107 | 0.5199 | 0.5228 | 0.8541 | 1.0000 | 0.5228 |
| Matched 512 | S1 | **0.6004** | **0.5993** | **0.6000** | **0.9021** | 1.0000 | **0.6000** |
| Matched 512 | T | 0.5917 | 0.5963 | 0.5982 | 0.8936 | 1.0000 | 0.5982 |
| Native ROI | S0 native | 0.5236 | 0.5323 | 0.5310 | 0.8583 | 0.9912 | 0.5263 |
| Native ROI | S1 native | **0.6030** | **0.6062** | **0.6070** | **0.9083** | 1.0000 | **0.6070** |
| Native ROI | T fixed 512 | 0.5917 | 0.5963 | 0.5982 | 0.8936 | 1.0000 | 0.5982 |

S1 improves over S0 by 0.0896 Macro-F1 under matched inputs and by 0.0794 when comparing the full-cohort point estimates of the two native evaluations. For inferential comparisons, the next table uses the explicitly defined common cohorts.

### Paired-bootstrap comparisons

Every delta is `Macro-F1(model B) - Macro-F1(model A)`.

| Scope | Comparison | Common test n | Delta Macro-F1 | 95% CI | CI excludes 0? |
|---|---|---:|---:|---:|---:|
| Matched 512 | S0 − T | 570 | −0.0810 | [−0.1249, −0.0382] | Yes |
| Matched 512 | S1 − S0 | 570 | **+0.0896** | **[+0.0470, +0.1326]** | **Yes** |
| Matched 512 | S1 − T | 570 | +0.0086 | [−0.0275, +0.0448] | No |
| Native ROI | S0 − T | 565 | −0.0747 | [−0.1177, −0.0323] | Yes |
| Native ROI | S1 − S0 | 565 | **+0.0860** | **[+0.0481, +0.1243]** | **Yes** |
| Native ROI | S1 − T | 565 | +0.0112 | [−0.0301, +0.0529] | No |
| Cross protocol | S1 native − T matched | 570 | +0.0112 | [−0.0295, +0.0528] | No |

The paired evidence is unambiguous for S1 versus S0: KD recovered a substantial part of the representation lost during positional-encoding surgery. The trained student has a slightly higher point estimate than the teacher, but the uncertainty interval is compatible with both a small loss and a small gain.

### Per-class F1

The class order is the sorted label order stored by the evaluator.

| Protocol / model | ADH | DCIS | FEA | IC | N | PB | UDH |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matched T | 0.3824 | **0.6162** | 0.7251 | 0.8718 | 0.6628 | 0.4678 | 0.4161 |
| Matched S0 | 0.2114 | 0.5464 | 0.7225 | 0.7660 | 0.5926 | 0.3892 | 0.3472 |
| Matched S1 | **0.4416** | 0.5422 | **0.7602** | **0.8758** | **0.6909** | 0.4607 | **0.4314** |
| Native T fixed 512 | 0.3824 | **0.6162** | 0.7251 | 0.8718 | 0.6628 | 0.4678 | **0.4161** |
| Native S0 | 0.2270 | 0.5283 | 0.6771 | 0.8811 | 0.6047 | 0.4693 | 0.2778 |
| Native S1 | **0.4027** | 0.5476 | **0.7470** | **0.8987** | **0.7263** | **0.4908** | 0.4076 |

The largest matched S1 recovery over S0 is for ADH, followed by UDH, N, and IC. S1 does not uniformly dominate T: DCIS remains lower under both S1 protocols, while native S1 is strongest on FEA, IC, N, and PB. These are descriptive class-level results; no class-wise confidence intervals or multiple-comparison correction were computed.

### Native-ROI extrapolation

Training views are restricted to at most 32 patches on either axis and at most 1,024 spatial tokens. Native BRACS test ROIs extend well beyond this support:

- 401/570 test ROIs have at least one grid axis greater than 32 (`coordinate_extrapolation`).
- 362/570 test ROIs have more than 1,024 spatial tokens (`sequence_extrapolation`).
- Test sequence length ranges from 36 to 55,685 tokens, with a median of 1,630.

Trained-student results by subgroup are:

| Grouping | Group | n | Coverage | Macro-F1 | Accuracy |
|---|---|---:|---:|---:|---:|
| Coordinate | In range | 169 | 1.0000 | 0.5304 | 0.5680 |
| Coordinate | Extrapolation | 401 | 1.0000 | **0.6067** | **0.6234** |
| Sequence | In range | 208 | 1.0000 | 0.5524 | 0.5817 |
| Sequence | Extrapolation | 362 | 1.0000 | **0.5991** | **0.6215** |

These subgroup differences must not be interpreted as evidence that longer sequences intrinsically improve accuracy. ROI size, tissue content, diagnosis mix, and difficulty differ between groups. The supported conclusion is narrower: in this run, S1 produced predictions with full test coverage and useful performance on many grids well outside its KD coordinate and sequence ranges.

### Feature-extraction coverage and OOM observations

Native feature extraction did not have identical success rates across model runs:

| Model | Train successful | Validation successful | Test successful |
|---|---:|---:|---:|
| S0 native | 3,497 / 3,657 (95.63%) | 304 / 312 (97.44%) | 565 / 570 (99.12%) |
| S1 native | 3,651 / 3,657 (99.84%) | 312 / 312 (100%) | 570 / 570 (100%) |
| T fixed 512 | 3,657 / 3,657 (100%) | 312 / 312 (100%) | 570 / 570 (100%) |

The result JSON summarizes test coverage only; the train and validation counts above come from the saved prediction records. Because S0 and S1 have the same architecture, weight training is not a credible mechanism for changing the theoretical attention-memory requirement. Their different OOM counts should be treated as execution-level observations that may depend on allocator state, free GPU memory, kernels, or run conditions—not as evidence that KD made the architecture more memory-efficient.

The model-specific missing train/validation features also mean the native S0 and S1 probes were fit and selected on slightly different successful cohorts. The paired bootstrap aligns successful test samples, but does not remove this upstream probe-training difference.

## Interpretation and limitations

### What the experiment supports

1. **Removing APE without adaptation is damaging.** S0 is significantly worse than T under matched 512 inputs, isolating the cost of the positional-encoding conversion.
2. **Representation KD repairs that disruption.** S1 is significantly better than S0 under both matched and native protocols, and its validation cosine similarities improve over training.
3. **Teacher-level linear-probe performance is retained.** S1's Macro-F1 point estimates are slightly above T, while paired confidence intervals include zero.
4. **The trained model can execute far beyond the KD token range.** Native S1 covers all 570 test ROIs, including sequences as long as 55,685 patches in the saved results.
5. **Native context is feasible, but its causal benefit is not isolated.** The cross-protocol S1-native versus T-matched estimate is positive but uncertain, and it changes both model and field of view.

### Important limitations

- This is one KD run and one BRACS split; run-to-run and split-level uncertainty were not measured.
- The bootstrap quantifies test-sample uncertainty for fixed fitted probes. It does not repeat feature extraction, probe fitting, hyperparameter selection, or KD training.
- Native T is not evaluated. `T_fixed512` is a fixed-grid reference, so native S1 versus T changes both encoder state and visible context.
- Native S0/S1 feature extraction used different successful train/validation cohorts because of OOMs.
- OOM behavior is hardware- and execution-dependent. No controlled memory benchmark was performed.
- Subgroup extrapolation results are descriptive and confounded by subgroup composition.
- Formal full-validation representation-fidelity files were not supplied. The available one-sample smoke outputs are intentionally not reported as experimental evidence.
- The associated training run records `git_commit: null`, and the BRACS artifacts do not embed the S1 checkpoint hash.
- No external test set, calibration analysis, clinical utility study, or pathology-reader comparison is included. This repository is a representation-learning experiment, not a clinical device.

## Reproduction

### Environment

```bash
conda activate MVLM_NRKD
cd /home/yasaisen/Desktop/27_research/research_main_v4/lab_01/NativeResKD
pip install -r requirements.txt
```

CONCH v1.5 weights are expected at the configured checkpoint location. They are not distributed by this repository and remain subject to their original license and gated-access terms.

### Preflight

The preflight validates the checkpoint architecture, APE-to-RoPE surgery, teacher/student gradients, one-batch KD loss, metadata constraints, and 512 × 512, 384 × 512, 512 × 384, and 384 × 384 student inputs.

```bash
python R27_MVLM_preflightNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json
```

For a small data-path smoke test:

```bash
python R27_MVLM_preflightNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json \
  --set train_size_expected=128 \
  --set valid_size_expected=64 \
  --set num_workers=0
```

### Training

Local single-GPU training:

```bash
python R27_MVLM_trainNRKD_v0.0.py \
  --cfg-path NativeResKD/projects/NativeResKD_config.json
```

Four-GPU HPC training:

```bash
sbatch NRKD_train_4GPU.slurm
```

Continue the same run exactly (`keepTrain`):

```bash
python R27_MVLM_trainNRKD_v0.0.py \
  --latest-checkpoint-path /path/to/checkpoints/YYMMDDHHMMSS \
  --use-this-dir
```

Load only the student and start a new optimizer/scheduler/run (`reTrain`):

```bash
python R27_MVLM_trainNRKD_v0.0.py \
  --best-checkpoint-path /path/to/checkpoints/YYMMDDHHMMSS \
  --retrain
```

Each checkpoint is an atomic three-part set: `[model]*.pth`, `[trainer]*.pth`, and `[checkpoint]*.json`. The manifest stores checkpoint IDs, sizes, and SHA-256 hashes. Exact continuation additionally validates metadata/model/loss/DDP/step signatures and restores rank-local Python, NumPy, PyTorch, CUDA, DataLoader-generator, and sampler states.

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

The evaluator reports full, background-trimmed, and nine deterministic lossy grids from 24 × 24 through 32 × 32, including global, projected, hidden-global, and mapped-patch cosine statistics.

### BRACS metadata

```bash
python build_bracs_metadata.py \
  --root /path/to/BRACS/BRACS_RoI/latest_version \
  --output projects/bracs_metadata.csv
```

The builder requires the official `train`, `val`, and `test` directories and records every ROI's dimensions, class, split, and source magnification.

### BRACS evaluation

Matched teacher and students:

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

Native-ROI evaluation:

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

Aggregate metrics and recompute the paired bootstrap:

```bash
python aggregate_bracs_results.py \
  --root bracs_eval \
  --bootstrap-iterations 10000 \
  --bootstrap-seed 42 \
  --bootstrap-confidence-level 0.95
```

If predictions already exist, bootstrap can be rerun without feature extraction:

```bash
python R27_MVLM_evalBRACSPairedBootstrapNRKD_v0.0.py \
  --root bracs_eval \
  --iterations 10000 \
  --seed 42 \
  --confidence-level 0.95
```

## Repository layout

```text
NativeResKD/
├── NativeResKD/
│   ├── configs/                 # Flat 4.01 configuration and validation
│   ├── criterions/              # Global, patch, and hidden representation KD
│   ├── datasets/                # KD/BRACS datasets and shape bucketing
│   ├── evaluator/               # BRACS, fidelity, and paired-bootstrap evaluation
│   ├── models/                  # CONCH loader, APE-to-RoPE surgery, Mixed RoPE
│   ├── processors/              # Tissue masks and native-view construction
│   ├── projects/                # Local and HPC resolved configurations
│   └── trainer/                 # DDP training, scheduler, and checkpoint policy
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

Per-evaluation BRACS directories contain `metrics.json`, `predictions.csv`, and `extrapolation_analysis.csv`. Aggregation produces `summary.csv`, `per_class_metrics.csv`, `native_coverage.csv`, `paired_bootstrap_macro_f1.csv`, and its JSON equivalent.

## License and weights

Repository code is covered by [LICENSE](LICENSE). CONCH v1.5 weights are not included and remain governed by their original license and access conditions. BRACS data are likewise not redistributed by this project. Users are responsible for obtaining all data and model weights under the applicable terms.
