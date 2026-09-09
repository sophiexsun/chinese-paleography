# Phase IV C2 + D 4×4 Complete Results — Analyzer Report

## Executive conclusion

The completed 4-model × 4-channel experiment supports **transferable composition on both the S side and the P side**, but it does **not** support equal or fully symmetric composition.

The most defensible summary is:

> `r_S` is a strong, broadly replicating compositional relation across direct-character and semantic channels. `r_P` also transfers beyond `zh_char`, most clearly in the Unihan English-gloss channel, but its effect is weaker and less consistent in full definition channels. The system is therefore partially symmetric in existence, but asymmetric in strength, robustness, and channel dependence.

This result is not primarily a `zh_char` artifact: important effects survive when the written target character is absent. However, the P-side evidence is **graded rather than universal**. It supports a transferable P-associated semantic signal in some channels—especially Unihan glosses—not the claim that every phonophore family carries a stable semantic contribution in every model and representation.

## The two symmetric tests

### S-side composition

For a held-out root family `F*`, learn the recurring contribution of `S` from other phonophore families:

`r_S^(-F*) = mean[E(S+P) - E(P)] from other root families`

Then test:

`E(P*) + r_S^(-F*) -> C*`

### P-side composition

For a held-out canonical `S*` context within one P family, learn the recurring contribution associated with `P` from the other distinct `S` contexts:

`r_P^(-S*) = mean[E(S+P) - E(S_independent)] from other S contexts`

Then test:

`E(S_independent*) + r_P^(-S*) -> C*`

Both tests use raw, uncentered embeddings; exact cosine `argmax`; identical baseline/offset candidate universes; deterministic tie handling; primary support of at least 3 other independent units; and predeclared `>=5` and `>=8` sensitivities.

## Run validation

- All **16 C2 model-channel checkpoints** completed.
- All **16 D model-channel checkpoints** completed.
- Models: Qwen3-0.6B, BGE-M3, GTE multilingual base, Qwen3-4B INT8.
- Primary channels: `zh_char`, `en_gloss_unihan`, `cn_def`, `en_def_translated_cn_def`.
- C2 held-out cases range from 2,960 in definition channels to 5,240 in `zh_char` at the primary threshold.
- D held-out cases range from 2,398 in definition channels to 4,763 in `zh_char`.
- D includes 320 eligible P root families in definition channels, 448 in Unihan glosses, and 474 in `zh_char`.
- Frozen A4 mapping validation, leakage checks, equal-context weighting, candidate-universe checks, ranking bounds, and exact Top-1 invariants passed.
- The P-versus-S matched summary is populated for all 16 conditions and all three thresholds, despite the final D QC text retaining a stale “matched comparison pending” line.

No redesign, threshold change, PCA/CCA/SVCCA, or learned `W` was introduced.

## 1. Does P-side improvement survive in semantic channels?

### Answer

**Yes, but unevenly.** The clearest cross-model P-side semantic survival occurs in `en_gloss_unihan`. Chinese definitions show a smaller positive tendency. Translated English definitions are weak and mixed.

### Unihan English gloss: consistent P-side success

All four models show:

- more cases improving than worsening;
- improved median and mean rank;
- positive Recall@1, Recall@10, and MRR gains.

| Model | Cases with better rank | Median rank | Recall@10 | R@10 gain |
|---|---:|---:|---:|---:|
| Qwen 0.6B | 59.1% | 1,280 -> 1,217 | 1.55% -> 4.90% | +3.34 pp |
| BGE-M3 | 59.8% | 1,845 -> 1,609 | 1.62% -> 4.30% | +2.68 pp |
| GTE | 62.4% | 1,299 -> 1,071 | 1.86% -> 4.52% | +2.65 pp |
| Qwen 4B | 59.2% | 970 -> 939 | 2.87% -> 5.40% | +2.53 pp |

At the root-family level, positive mean effects appear in 54.5%–65.2% of families across the four models. That is directionally replicated, but less overwhelming than the edge-level percentages.

Because Unihan gloss embeddings do not contain the written target character itself, direct graphic containment cannot explain this result. This is the strongest current evidence that `r_P` includes transferable semantic or lexical-semantic information.

### Chinese definitions: weak-to-moderate P-side survival

All four models have positive mean-rank and MRR changes, and 53.0%–55.0% of edges improve rank. Results at the top of the list are mixed:

- Recall@1 increases in all four models.
- Recall@10 increases for GTE and Qwen 4B, but declines slightly for BGE-M3 and Qwen 0.6B.
- Median rank improves for GTE, Qwen 0.6B, and Qwen 4B, but worsens for BGE-M3.
- Root-family positive shares range from 48.1% to 62.2%.

This is suggestive semantic evidence, not strong convergence.

### Translated English definitions: near-neutral and inconsistent

- Only three models exceed 50% edge-level rank improvement, and only slightly.
- Mean rank improves in all four, but median changes are tiny.
- Recall@10 decreases in three of four models.
- MRR increases in only two models.
- Root-family positive shares are 49.1%–56.9%.

Therefore, P-side transfer does not robustly survive this channel under all metrics.

![P-side Recall@10 gains](Phase_IV_C2_D_4x4_Analyzer_Charts_v1/01_p_side_recall10_gain.png)

![P-side rank improvement share](Phase_IV_C2_D_4x4_Analyzer_Charts_v1/02_p_side_rank_improvement_share.png)

## 2. Matched S-side versus P-side strength

The matched table restricts both equations to identical model, channel, and held-out edges. This is the valid comparison; unmatched headline rates should not be used to rank the two sides.

### How broadly does each offset improve rank?

Average across four models on matched cases:

| Channel | P-side better-ranked cases | S-side better-ranked cases |
|---|---:|---:|
| `cn_def` | 53.8% | **85.4%** |
| `en_def_translated_cn_def` | 52.8% | **84.0%** |
| `en_gloss_unihan` | 60.2% | **87.3%** |
| `zh_char` | 68.3% | **81.0%** |

S-side is much broader in every channel.

![Matched S versus P rank improvement](Phase_IV_C2_D_4x4_Analyzer_Charts_v1/03_matched_s_vs_p_rank_improvement.png)

### How much does Top-10 identification improve?

Average Recall@10 gain across four models on matched cases:

| Channel | P-side gain | S-side gain | Stronger side |
|---|---:|---:|---|
| `cn_def` | +0.16 pp | **+1.16 pp** | S |
| `en_def_translated_cn_def` | -0.08 pp | **+1.05 pp** | S |
| `en_gloss_unihan` | **+2.62 pp** | +1.40 pp | P |
| `zh_char` | +0.95 pp | **+2.20 pp** | S |

The important exception is Unihan gloss: P-side helps fewer cases than S-side, but when it helps, it pushes more true characters into the Top 10.

![Matched Recall@10 gains](Phase_IV_C2_D_4x4_Analyzer_Charts_v1/04_matched_s_vs_p_recall10_gain.png)

### Median-rank scale

Across semantic channels, S-side median-rank improvements are extremely large and consistent:

- `cn_def`: average improvement of about 1,831 ranks.
- translated definition: about 1,704 ranks.
- Unihan gloss: about 2,834 ranks.

P-side median improvements are much smaller:

- `cn_def`: about 48 ranks.
- translated definition: about 7 ranks.
- Unihan gloss: about 122 ranks.

Thus, the two equations are symmetric in form but not in empirical strength.

## 3. Model × channel consistency

### S-side

S-side forward composition is the most stable result in the experiment:

- In all three semantic channels, every model shows very broad rank improvement on matched cases—approximately 83%–88%.
- Recall@10 gains are positive across every model in every semantic channel.
- This directly rejects a “primarily `zh_char` structure” interpretation for `r_S`.

### P-side

P-side is channel- and architecture-dependent:

- **Unihan gloss:** positive across all four models and all primary aggregate retrieval metrics.
- **Chinese definitions:** small positive direction overall, strongest for GTE and Qwen 0.6B at the family level; BGE-M3 is near neutral.
- **Translated definitions:** weak and inconsistent.
- **`zh_char`:** Qwen 0.6B and Qwen 4B improve mean and median rank; BGE-M3 and GTE worsen them substantially despite a majority of edges moving upward.

The `zh_char` split aligns with earlier Phase III evidence that Qwen 0.6B and Qwen 4B share especially strong linear alignment. It also warns against treating one model’s direct-character result as a property of Chinese orthography rather than of that model family.

![P-side model-channel heatmap](Phase_IV_C2_D_4x4_Analyzer_Charts_v1/05_p_side_model_channel_heatmap.png)

![P-side family-level replication](Phase_IV_C2_D_4x4_Analyzer_Charts_v1/06_p_side_family_positive_share.png)

## 4. Symmetric, asymmetric, or primarily `zh_char`?

### Not purely `zh_char`

This interpretation is rejected because:

- S-side composition replicates strongly across all semantic channels and models.
- P-side composition replicates most consistently in Unihan English glosses.
- These channels do not embed the target written character itself.

### Not fully symmetric

This interpretation is also rejected because:

- S-side rank improvement is much broader than P-side improvement on identical cases.
- S-side is strong in full Chinese and translated definitions; P-side is weak or mixed there.
- P-side’s clearest strength is concentrated in short Unihan glosses.

### Best description: partially symmetric, empirically asymmetric composition

Both components make transferable contributions:

- `S` contributes a general relation learned across independent P/root families.
- `P` contributes a family-specific relation learned across distinct S contexts.

But the strength and generalizability differ. `r_S` behaves like a robust compositional relation across representational channels. `r_P` behaves like a weaker, channel-sensitive family signal that is semantically recoverable most clearly in concise lexical glosses.

## 5. Does this support P as a semantic carrier?

The full result permits a stronger statement than the earlier one-condition run:

> P carries transferable information that survives removal of direct written-character identity and improves derived-character retrieval in a semantic channel across four different embedding models.

That supports **a P-associated semantic contribution**, especially in Unihan glosses.

The justified qualification is equally important:

> The semantic contribution is not uniformly recoverable across full Chinese and translated definitions, and it is not equally strong across P families.

Therefore the evidence supports “P can act as a semantic carrier across contexts” more strongly than “P universally acts as the semantic carrier.” It remains synchronic evidence from language encoders, not proof of historical character-formation causation.

## 6. Human-readable retrieval cases

### Replicated success: `E(艸) + r_扁 -> 萹`

In the Unihan English-gloss channel, adding the P-side relation learned from other 扁-family S contexts retrieves 萹 at Top 1 in **all four models**:

| Model | Baseline true rank | Offset true rank |
|---|---:|---:|
| Qwen 0.6B | 118 | **1** |
| BGE-M3 | 105 | **1** |
| GTE | 82 | **1** |
| Qwen 4B | 21 | **1** |

This is a clear, cross-model semantic-channel success.

### Chinese-definition improvement: `E(口) + r_共 -> 哄`

All four models improve the true rank in `cn_def`:

| Model | Baseline rank | Offset rank |
|---|---:|---:|
| Qwen 0.6B | 136 | 20 |
| BGE-M3 | 54 | 2 |
| GTE | 228 | 31 |
| Qwen 4B | 47 | **1** |

This is a useful example of P-side transfer surviving in a full definition channel, although only Qwen 4B reaches Top 1.

### Diagnostic mixed case: `E(火) + r_堯 -> 燒`

- In `cn_def`, all four models improve the true rank modestly.
- In translated definitions, all four also improve rank.
- In Unihan gloss, all four worsen rank.
- In `zh_char`, Qwen 0.6B improves 6 -> 5, while BGE-M3 and GTE suffer very large losses.

Thus, 堯 is not a universal win. Its outcome depends strongly on what semantic representation is supplied and which model encodes it.

### Replicated failure: `E(虫) + r_扁 -> 蝙`

All four models worsen severely in both full-definition channels:

- `cn_def` losses range from about 3,660 to 4,904 ranks.
- translated-definition losses range from about 1,420 to 4,784 ranks.

Yet two models improve it in Unihan gloss. The same P family can therefore transfer well for one member such as 萹 and fail for another such as 蝙. This argues against universal within-family semantic uniformity.

### Top-1 loss: `E(斤) + r_父 -> 斧`

In translated English definitions, 斧 begins at Top 1 for multiple models. Adding `r_父` knocks it to rank 2 or 3 in BGE-M3, Qwen 0.6B, and Qwen 4B. This is a transparent case where the learned P-family relation damages an already-correct semantic identification.

### 占-family diagnostics

For the P-side construction:

- `E(手) + r_店 -> 掂` is mixed: some models/channels improve, others worsen; no condition identifies 掂 at Top 1.
- `E(心) + r_店 -> 惦` improves strongly in `cn_def` for all four models, but still remains far from Top 1; `zh_char` is positive for both Qwen models and sharply negative for GTE.

These cases remain conceptually illuminating but do not determine the population conclusion.

## 7. Statistical reading and cautions

- The root family, not the individual edge, is the safer inferential unit for P-side generalization.
- Unihan P-side effects remain directionally positive at the family level in all four models, but positive-family shares are only 54.5%–65.2%; some large families or effects contribute disproportionately to edge-level gains.
- Chinese-definition P-side family effects range from near-neutral BGE-M3 (48.1% positive) to stronger GTE (62.2%).
- Translated-definition P-side family results cluster near 50%, so this channel should not be described as successful replication.
- Top-1 rates remain low in absolute terms because candidate universes contain thousands of characters. Rank distributions, MRR, Top-5/10, rescues/losses, and family effects are necessary alongside Recall@1.
- The semantic channels remove direct target-character identity, but they do not prove historical causation or establish the particular historical sense transmitted by P.
- C2’s earlier matched-control row-key concern should still be resolved before final paired-control claims; it does not affect the baseline-versus-offset or matched S/P retrieval tables used here.

## 8. Answers to the five requested questions

1. **Does P-side improvement survive in semantic channels?** Yes—clearly in Unihan gloss across all models, modestly in Chinese definitions, and not robustly in translated definitions.
2. **S-side versus P-side on identical cases?** S-side is much broader and usually stronger. P-side exceeds S-side only for matched Unihan Recall@10 gain.
3. **Model × channel consistency?** S-side semantic replication is highly consistent. P-side is consistent in Unihan gloss, weak/mixed in definitions, and model-family-dependent in `zh_char`.
4. **Symmetric, asymmetric, or `zh_char`?** Partially symmetric in existence, strongly asymmetric in magnitude and robustness, and not primarily `zh_char`.
5. **Human-readable cases?** Cross-model successes such as `艸 + r_扁 -> 萹` coexist with replicated failures such as `虫 + r_扁 -> 蝙`; this heterogeneity is scientifically central.

## What I would bring to Planner next

1. Treat S-side semantic composition as the strongest replicated Phase IV result.
2. Treat P-side Unihan replication as genuine evidence that P-associated information can include semantics beyond written-character identity.
3. Phrase the P conclusion as conditional and heterogeneous: “P can carry transferable semantic information,” not “P universally carries semantic information.”
4. Characterize the overall system as **partially symmetric but empirically asymmetric**.
5. Investigate why P-side transfer is strong in concise Unihan glosses but weak in full translated definitions before designing a learned subspace.
6. Preserve family-level reporting so a few large or strongly responding families do not dominate the narrative.
7. Use replicated successes and failures to define predeclared heterogeneity analyses, not to tune the existing result.

## Chart-data files

The chart directory contains one CSV per major visualization plus a complete replicated-case table:

- `01_p_side_condition_summary.csv`
- `03_matched_s_vs_p_rank_improvement.csv`
- `04_matched_s_vs_p_recall10_gain.csv`
- `05_p_side_mean_rank_gain_heatmap.csv`
- `06_p_side_family_effects.csv`
- `07_p_side_replicated_cases.csv`
