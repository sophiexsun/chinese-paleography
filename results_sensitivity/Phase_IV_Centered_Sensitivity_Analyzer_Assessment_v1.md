# Phase IV Centered-Sensitivity Analyzer Assessment v1

## Status and scope

**Sensitivity verdict: PASS.** The centered analysis preserves and generally strengthens the primary raw result. It does not produce a material reversal of the semantic-channel conclusion or of the empirical S-side > P-side asymmetry.

The uncentered raw analysis remains primary. Centering is interpreted only as a sensitivity analysis testing whether the results depend on a model/channel-wide mean embedding direction.

One earlier secondary interpretation requires qualification: the raw `zh_char` P-side architecture split is not stable under centering. The negative raw median effects for BGE-M3 and GTE become positive after centering. Therefore, the raw character-only model dependence should not be treated as an intrinsic architectural difference.

## Integrity and comparability

- Embedding-cache integrity: **16/16 PASS**.
- Centered C2 completion markers: **16/16**.
- Centered D completion markers: **16/16**.
- Centering rule: within each model × channel, subtract the mean over every cached character, then use the centered representation for all entities and candidates.
- Eligibility counts match between raw and centered summaries.
- Primary support threshold: at least 3 other training families/contexts. Thresholds 5 and 8 are secondary sensitivity checks.

Because centering changes both the baseline and offset searches, robustness is assessed by comparing the **within-mode gain** (`offset − baseline`) and paired raw/centered family effects—not by comparing raw and centered absolute ranks in isolation.

## 1. S-side forward retrieval

The principal S-side result is highly robust.

| Indicator across 16 model × channel cells | Raw | Centered |
|---|---:|---:|
| Majority of edges improve | 16/16 | 16/16 |
| Positive median rank gain | 14/16 | 16/16 |
| Positive Recall@10 gain | 15/16 | 16/16 |
| Mean share of edges improving | 84.45% | 84.16% |
| Median of cell-level median rank gains | +1,905 | +2,073.5 |
| Mean Recall@10 gain | +1.390 pp | +1.612 pp |

At the family level in the three semantic channels:

- mean positive-family share is essentially unchanged: **94.36% raw → 94.15% centered**;
- mean raw/centered sign concordance is **98.43%** among nonzero pairs;
- mean Spearman family-effect correlation is **0.830**;
- average cell median family effect increases from **+1,420 → +1,632 rank positions**.

This is strong directional replication with no material attenuation. Centering slightly strengthens the magnitude while preserving which semantic families tend to benefit.

## 2. P-side forward retrieval

The P-side result also survives, but remains smaller and more representation-sensitive than the S-side result.

| Indicator across 16 model × channel cells | Raw | Centered |
|---|---:|---:|
| Majority of cases improve | 15/16 | 16/16 |
| Positive median rank gain | 12/16 | 16/16 |
| Positive Recall@10 gain | 9/16 | 11/16 |
| Mean share of cases improving | 58.46% | 60.57% |
| Median of cell-level median rank gains | +43.75 | +226.5 |
| Mean Recall@10 gain | +0.921 pp | +0.958 pp |

At the family level in the three semantic channels:

- mean positive-family share rises from **55.83% → 62.51%**;
- mean raw/centered sign concordance is **84.81%** among nonzero pairs;
- mean Spearman family-effect correlation is **0.866**;
- average cell median family effect increases from **+54.7 → +127.7 rank positions**.

This is not attenuation. Centering strengthens the broad rank evidence for a transferable P-side contribution. The lower sign concordance than on the S-side confirms that P effects are more heterogeneous and more sensitive to representation geometry.

## 3. P-side semantic-channel dependence

Matched S-versus-P comparisons on identical cases give the following P-side results, averaged equally across the four models:

| Channel | Positive-case share raw → centered | Median rank gain raw → centered | Recall@10 gain raw → centered |
|---|---:|---:|---:|
| `cn_def` | 53.76% → 55.26% | +47.75 → +124.5 | +0.155 → +0.129 pp |
| `en_def_translated_cn_def` | 52.77% → 53.83% | +6.75 → +67.5 | −0.078 → −0.129 pp |
| `en_gloss_unihan` | 60.16% → 65.26% | +122.13 → +339.88 | +2.620 → +2.351 pp |
| `zh_char` | 68.25% → 69.16% | −346.63 → +380.5 | +0.948 → +1.320 pp |

Interpretation:

1. **Unihan English gloss remains the clearest P-side semantic channel.** Its Recall@10 gain attenuates modestly (about 0.27 percentage points), but remains large, positive, and replicated across models. Its rank breadth strengthens substantially.
2. **Chinese definitions remain small but directionally positive.** Rank evidence strengthens; the already small Recall@10 gain attenuates slightly.
3. **Translated definitions remain weak/mixed.** Centering improves the proportion and rank magnitude, but the average Recall@10 change stays slightly negative. This channel cannot independently carry the strongest form of the P-side claim.
4. **`zh_char` changes materially.** Centering reverses the aggregate median from negative to positive and specifically removes the negative BGE/GTE median pattern. The raw architecture split is therefore sensitivity-dependent.

## 4. Matched S-side versus P-side interpretation

The empirical asymmetry survives centering.

| Channel | Centered S positive share | Centered P positive share | Centered S median rank gain | Centered P median rank gain |
|---|---:|---:|---:|---:|
| `cn_def` | 85.46% | 55.26% | +1,990.25 | +124.5 |
| `en_def_translated_cn_def` | 83.72% | 53.83% | +1,952.5 | +67.5 |
| `en_gloss_unihan` | 87.22% | 65.26% | +3,250.5 | +339.88 |
| `zh_char` | 79.66% | 69.16% | +1,694.88 | +380.5 |

S-side improvement remains much broader and larger in rank magnitude in every channel. P-side improvement nevertheless remains detectable, especially in Unihan glosses. On matched Unihan cases, P has the larger Recall@10 gain (**+2.351 pp P versus +1.244 pp S**) even though S improves far more cases and produces much larger median rank movement. This distinction should be retained:

- S-side: broad, strong, highly stable compositional contribution;
- P-side: smaller and heterogeneous, but capable of meaningful high-rank retrieval gains in the strongest semantic channel.

The results continue to support **partially symmetric composition in existence but asymmetric composition in empirical strength and robustness**.

## 5. Inverse S-side anchor recovery

Inverse anchor recovery remains secondary and modest.

| Indicator across 16 cells | Raw | Centered |
|---|---:|---:|
| Majority of ranks improve | 16/16 | 15/16 |
| Positive median rank gain | 15/16 | 13/16 |
| Positive Recall@10 gain | 7/16 | 10/16 |
| Mean share improving | 56.05% | 54.52% |
| Mean Recall@10 gain | +0.123 pp | +0.320 pp |

Centering improves sparse Top-10 performance but slightly reduces the breadth of rank improvement. This does not overturn the earlier characterization: inverse recovery is suggestive and useful for human-readable identification, but materially weaker than forward S-side retrieval.

## 6. Support-threshold sensitivity

The direction is stable at thresholds 3, 5, and 8:

- Centered S-side forward: positive Recall@10 gain in **16/16 cells at every threshold**; positive median rank gain in **16/16**.
- Centered P-side: positive Recall@10 gain in **11/16 cells at every threshold**; majority rank improvement and positive median rank gain in **16/16**.
- Centered inverse recovery: positive Recall@10 gain increases from **10/16 cells at thresholds 3 and 5 to 11/16 at threshold 8**.

There is no evidence that the principal conclusions depend on the frozen minimum-support cutoff.

## 7. What remains supported—and what changes

### Supported robustly

> Conventionally classified phonophores can carry transferable semantic information across distinct character contexts, although the effect is heterogeneous and empirically weaker or less robust than S-side composition.

The centered analysis strengthens rather than weakens this conditional wording. The result is not explained solely by a common model/channel mean direction.

### Retain as qualifications

- P-side evidence is strongest in Unihan glosses, modest in Chinese definitions, and weak/mixed in translated definitions.
- Retrieval improvement is primarily a distributional rank effect; Top-1 remains sparse.
- P-side effects are more family-specific and representation-sensitive than S-side effects.
- Centering is a sensitivity analysis, not a replacement primary specification.

### Revise the earlier secondary interpretation

Do not state that the `zh_char` P-side effect is inherently positive only in Qwen architectures and negative in BGE/GTE. That pattern reverses under centering. Instead state:

> Character-only P-side magnitude is sensitive to global embedding geometry; after mean-centering, all four models show positive median rank improvement.

## Analyzer conclusion

**PASS for centered sensitivity.** There is no material reversal of the core semantic-channel finding or the S-side > P-side interpretation. Centering generally strengthens rank-based effects, modestly attenuates a few Top-k gains, and exposes one sensitivity-dependent secondary result in `zh_char`. The raw analysis should remain primary, accompanied by the centered results as evidence of robustness and as a qualification on character-only architecture dependence.

