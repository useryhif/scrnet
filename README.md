# SCRNet — Self-Constrained Hierarchical Crystal Symmetry Recognition from Powder XRD

Deep learning models that classify crystallographic properties of materials directly from
1D powder X-ray diffraction (XRD) spectra, across the crystallographic symmetry hierarchy.

| Level | Classes | Description |
|-------|---------|-------------|
| Crystal system | 7 | Triclinic, Monoclinic, Orthorhombic, Tetragonal, Trigonal, Hexagonal, Cubic |
| Lattice type (Bravais) | 14 | e.g. `cubic_F`, `orthorhombic_I`, `trigonal_R` |
| Point group | 32 | e.g. `4/mmm`, `-43m`, `6/m` |
| Space group | 230 | `spacegroup1` … `spacegroup230` — the finest and hardest level |

Input is a 1D XRD spectrum of length 1500. The repository holds one model per level of the
hierarchy, the code that trains them, and the figures from the paper.

---

## Repository layout

```
SCRNet/
├── src/
│   ├── labels.py                Symmetry hierarchies and the expert partition
│   ├── dataset.py               Datasets (plain, per-expert train, per-expert test)
│   ├── resnet.py                RCNet — baseline architecture for the three lower levels
│   ├── scrnet.py                SCRNet — the space-group expert
│   ├── focal_loss.py            Focal loss
│   ├── selective_loss.py        Selective loss for expert training
│   ├── train_baseline.py        Train crystal system / lattice type / point group
│   ├── train_spacegroup.py      Train the seven space-group experts
│   ├── evaluate_spacegroup.py   Route the experts into a final space-group prediction
│   ├── fetch_xrd.py             Download structures from the Materials Project, build XRD
│   └── preprocess.py            Label, resample and split into train/val/test
├── results/
│   ├── README.md                Figure captions and provenance
│   ├── figures/                 Figures 1–5, extracted from the manuscript
│   └── confusion_matrices/      Row-normalized matrices at all four levels
├── requirements.txt
└── README.md
```

Run everything from inside `src/`, which is the working directory the scripts expect:

```bash
cd src
```

---

## The four models

### Crystal system / lattice type / point group — `resnet.py`

The three lower levels are each predicted directly from the XRD spectrum by **`RCNet`**: a
1D ResNet with multi-head self-attention after the stem convolution, "same"-padded
convolutions, and a confidence head alongside the classification logits.

### Space group — `scrnet.py`

The 230-class problem is the hard one. Space groups are partitioned by crystal system and
**one expert is trained per partition** (seven in total). Each expert takes the XRD spectrum
plus the predicted crystal-system, Bravais-lattice and point-group probability vectors, fuses
them through cross-modal attention, and predicts three things at once:

* space-group logits over all 230 classes,
* a binary confidence score — "does this input belong to my crystal system?",
* the unit-cell parameters (a, b, c, α, β, γ) as an auxiliary regression task.

The confidence head is what makes routing work: at inference each expert scores every sample,
the routing step picks the highest-scoring expert, and that expert's space group is the final
answer.

| Expert | Crystal system | Space group range | Count |
|--------|---------------|-------------------|-------|
| 0 | Triclinic | 0 | 2 |
| 1 | Monoclinic | 2 | 13 |
| 2 | Orthorhombic | 15 | 59 |
| 3 | Tetragonal | 74 | 68 |
| 4 | Trigonal | 142 | 25 |
| 5 | Hexagonal | 167 | 27 |
| 6 | Cubic | 194 | 36 |

---

## Data preparation

Set `MP_API_KEY` first — get a key at https://materialsproject.org/api.

```bash
export MP_API_KEY=...

# 1. Download structures and compute Gaussian-broadened XRD patterns
python fetch_xrd.py --by space_group --output-dir data/raw

# 2. Label, resample onto the training grid, and split
python preprocess.py --input data/raw --output-dir data --lattice-params
```

**Labels are derived, not copied.** `preprocess.py` reads each space group number with
pymatgen and derives the crystal system and point group directly, then forms the Bravais
lattice from the crystal system plus the centring letter of the space group symbol. Two
normalisations bring that into the standard 14-lattice notation: `orthorhombic_A` →
`orthorhombic_C` (A-centring is equivalent to C-centring) and `trigonal_P` → `hexagonal_P`
(the trigonal primitive lattice is hexagonal). The derivation has been checked to produce
exactly the 14 labels in `labels.py` for all 230 space groups.

**The three probability columns (−5, −4, −3) are model outputs.** They hold the upstream
crystal-system, Bravais-lattice and point-group distributions, so they cannot be computed
from the raw data. `preprocess.py` writes zeros there and warns unless you pass
`--embeddings` with vectors produced by upstream inference. Training on the zeros is
meaningless.

---

## Training

### The three lower levels

```bash
cd src
python train_baseline.py --task crystal_system \
    --train data/train.npy --val data/val.npy --test data/test.npy \
    --epochs 800 --output-dir runs
```

`--task` is one of `crystal_system`, `lattice`, `point_group`; the number of output classes
follows from the task. The best-validation checkpoint and a per-epoch CSV are written to
`--output-dir`, and the test confusion matrix is saved as `.npy` at the end.

### The space-group experts

```bash
python train_spacegroup.py --train data/train.npy --val data/val.npy \
    --epochs 100 --output-dir runs
```

This trains all seven experts in sequence; `--model-index 3` trains only one. Each expert
keeps the checkpoint with the best mean of its two accuracies (space group and confidence)
and stops early after `--patience` epochs without improvement. Existing checkpoints are
resumed unless `--scratch` is passed.

Each expert's loss is the sum of three terms:

1. **`SelectiveLoss`** over the space-group head — samples inside the expert's label range
   get `CrossEntropyLoss`; samples outside it have their logits pulled toward 0.5, so the
   expert stays undecided on inputs belonging to another crystal system.
2. **Cross-entropy** over the binary confidence head.
3. **MSE** over the unit-cell parameters, scaled by 1/1000.

### Inference

```bash
python evaluate_spacegroup.py --test data/test.npy --weights runs
```

Every expert is run over the full test set, then routed. The routing score interpolates
between an expert's peak class probability and its confidence:

```
score = gamma * max_prob + (1 - gamma) * confidence
```

`--gamma 0` (the default) routes purely on the confidence head. The script prints the
space-group accuracy and per-expert usage, and writes the 230 × 230 confusion matrix.

---

## Data format

Data is stored as `.npy` and loaded with `np.load(..., allow_pickle=True)`. Columns are
**positional, not named**; the indices live in `labels.py`.

| Index | Content |
|-------|---------|
| 0 | Material ID |
| 1 | Element list |
| 2 | Crystal system label |
| 3 | Space group label |
| 4 | Lattice type label |
| 5 | Point group label |
| 6 | XRD spectrum (float array, length 1500) |
| −5 | Crystal system probabilities (7) |
| −4 | Bravais lattice probabilities (14) |
| −3 | Point group probabilities (32) |
| −2 | Unit cell a, b, c |
| −1 | Unit cell α, β, γ |

---

## Results

Baseline validation accuracy for the three lower levels, from the training logs of the final
runs (single train/val split, ~70/17.5/12.5):

| Task | Best val acc | Epoch |
|------|-------------|-------|
| Crystal system | 0.862 | 694 |
| Lattice type | 0.799 | 785 |
| Point group | 0.630 | 793 |

**Final space-group result** — the seven experts, evaluated on the full 15,360-sample frozen
test set (test set used for evaluation only, no tuning):

| Metric | Value |
|--------|-------|
| Space-group accuracy | 89.59% |
| Top-3 accuracy | 96.85% |
| Top-5 accuracy | 98.20% |
| Macro precision (206 classes present in test) | 70.05% |
| Macro recall (206 classes present in test) | 66.30% |
| Macro F1 (206 classes present in test) | 66.85% |

Macro metrics are sensitive to the class set and zero-division convention, so the denominator
is stated explicitly. Parameters: 164,495,410 per expert, 1,151,467,870 across the seven;
~5.62 ms per sample for the full seven-expert inference pass on a Tesla V100 32 GB.

The confidence head output is a **domain-routing score** (mean ≈ 0.9994), not a calibrated
space-group probability; selected-class ECE-15 is 7.84%.

---

## What is not in this repository

- **Datasets** (`.npy`, 1.8–2.9 GB each) — the merged labelled dataset, class-balanced
  variants, and pre-split train/val/test tensors.
- **Trained weights** (`.pth`) — the seven experts (~4.6 GB in FP32) and the per-task
  checkpoints.

The scripts will therefore not run as-is until the data paths are populated. The dataset
derives from the Materials Project.

---

## Environment

No virtual environment or package manager is configured; this was developed against system
Python 3.10 with CUDA 12.6. See `requirements.txt`.

Models detect CUDA automatically and are wrapped in `nn.DataParallel` **only when a GPU is
present**. Checkpoints are saved from `model.module` where applicable, so the `module.`
prefix is stripped — load them into an unwrapped model.

---

## Citation

If you use this code, please cite the accompanying paper. *(Citation details to be added.)*

## License

No license has been chosen yet, so this code is currently **all rights reserved**. If you
intend others to reuse it, add a `LICENSE` file — MIT and Apache-2.0 are common choices for
research code.
