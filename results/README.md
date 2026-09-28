# Results

The five figures below are the figures used in the paper. They were extracted directly from
the manuscript (`SCRNet_IUCrJ.docx`), so these files are the authoritative versions that
appear in the text.

All results are on the same frozen 15,360-sample test set, used for evaluation only.

---

## Figure 1 — `figures/figure1_hierarchical_performance.png`

> **Figure 1** Comparison of model performance across hierarchical symmetry levels. (a) Overall
> test accuracy, (b) macro-averaged precision, and (c) macro-averaged recall at the
> crystal-system, Bravais-lattice, point-group, and space-group levels for the baseline ResNet,
> the hierarchy ResNet, SCRNet, and SCRNet+. (d-f) Class-wise recall for (d) the seven crystal
> systems, (e) the fourteen Bravais lattices, and (f) the thirty-two point groups. For all
> models, the accuracies at the upper symmetry levels are obtained by mapping the predicted
> space groups back to the corresponding crystal systems, Bravais lattices, and point groups.

---

## Figure 2 — `figures/figure2_architecture.png`

> **Figure 2** Architecture of the proposed self-constrained framework and its geometric
> enhancement. (a) Overview of the Self-Constrained Residual Network (SCRNet) for hierarchical
> crystal symmetry inference. The one-dimensional powder XRD pattern, together with the
> predicted probability distributions over crystal systems, Bravais lattices, and point groups,
> is provided to multiple specialized experts. Each expert fuses these inputs to generate a
> probability distribution over its assigned space groups and a confidence score. A routing
> mechanism then selects the final prediction based on the model selection score. (b) The base
> architecture is shared between hierarchy model, SCRNet, and SCRNet+. It consists of an
> attention layer that fuses the inputs, a ResNet, and parallel fully connected layers that
> produce the space-group distribution. The confidence head that outputs the score c and the
> routing that selects the final prediction are specific to the SCRNet experts and are not used
> in the hierarchy model. The yellow elements represent the geometric enhancement branch
> exclusive to SCRNet+. This branch performs an auxiliary regression task on lattice parameters
> to guide representation learning during training and is ignored during inference.

---

## Figure 3 — `figures/figure3_expert_specialization.png`

> **Figure 3** Specialization behavior of the SCRNet experts. (a) Sample-number distribution
> over the 230 space groups. (b) Average predicted space-group distributions of the SCRNet
> experts. Each color corresponds to one expert and shows the mean output probability as a
> function of space-group index. The localized probability concentration demonstrates that
> different experts specialize in restricted regions of the full label space, providing direct
> evidence that the self-constrained framework partitions the symmetry inference task into more
> manageable subsets. (c) Mean per-class F1 score of space group classification as a function
> of the number of samples per space group. Space groups are binned into five intervals, with
> the number of classes in each bin indicated in parentheses.

---

## Figure 4 — `figures/figure4_bravais_confusion_matrices.png`

> **Figure 4** Bravais-lattice confusion matrices. Row-normalized confusion matrices at the
> Bravais-lattice level for (a) the baseline ResNet and (b) SCRNet+, where each row denotes the
> true label and each column the predicted label. The baseline matrix shows substantial
> off-diagonal weight, indicating frequent confusion across related lattice categories, whereas
> the SCRNet+ matrix is markedly more concentrated along the diagonal. This shows that
> hierarchical conditioning, expert specialization, and geometric supervision make the
> predictions more crystallographically localized and reduce broad misclassification across
> symmetry categories.

---

## Figure 5 — `figures/figure5_spacegroup_precision.png`

> **Figure 5** Class-wise performance and sample imbalance at the space-group level. Per-class
> precision versus the logarithm of the predicted count for each space group. Marker size is
> also proportional to the number of samples in each class, and colors distinguish crystal
> systems. The star markers indicate the enantiomorphic pairs. The results show that sample
> imbalance and crystallographic distinguishability jointly affect fine-grained space-group
> inference.

The manuscript's discussion of this figure:

> Figure 5 further clarifies the reliability and remaining limitations of the space-group
> predictions. In practical use, precision is especially important because it answers a
> different question from recall. When the model predicts a given space group, precision
> measures how trustworthy that predicted label is. The figure shows that most space groups
> with more than 100 predicted test samples have precision above 0.8. This suggests that
> high-frequency predicted labels are generally reliable, even though rare classes remain more
> difficult. Therefore, the model can be useful as a high-throughput screening tool in which
> confident predictions for well-populated groups are treated as reliable candidates for
> subsequent crystallographic verification.
>
> The low-precision outliers in Figure 5 also have a clear physical interpretation. The starred
> points mark enantiomorphic space groups, corresponding to left- and right-handed structures.
> These pairs are indistinguishable [...]

---

## `confusion_matrices/`

These are **not** the paper's figures — they are the row-normalized confusion matrices
underlying Figures 1 and 4, exported at all four symmetry levels. Each file is a row of four
matrices (baseline ResNet, hierarchy ResNet, SCRNet, SCRNet+) sharing a single colour scale,
so the panels are directly comparable.

| File | Level | Matrix size |
|------|-------|-------------|
| `crystal_system_4models.png` | Crystal system | 7 × 7 |
| `bravais_lattice_4models.png` | Bravais lattice | 14 × 14 |
| `point_group_4models.png` | Point group | 32 × 32 |
| `space_group_4models.png` | Space group | 230 × 230 |

Rows are normalized, so each row sums to 1 and the colour encodes the *proportion* of true
class *i* predicted as class *j* — not raw counts. This matters most for the 230-class case,
where raw counts are dominated by a handful of well-populated space groups.

The paper uses only the Bravais-lattice pair as Figure 4, in a two-panel layout; these
four-level versions are the full set.

---

## Notes

- These figures come from the paper's evaluation pipeline, not from the training scripts in
  `src/` — those scripts produce checkpoints and training logs, not these plots.
- Space-group accuracy figures quoted in the text: baseline ResNet 62.3%, hierarchy ResNet
  81.5%, SCRNet 85.7%, SCRNet+ 89.6%.
