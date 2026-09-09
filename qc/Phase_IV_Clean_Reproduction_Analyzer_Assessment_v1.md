# Phase IV Clean-Reproduction Analyzer Assessment v1

## Verdict

**PASS — the frozen raw C2 and D numerical results reproduced exactly from the validated embedding caches.**

This clears the clean-reproduction portion of Step 10. It does not by itself complete the entire validation-and-freeze round; family-level inference, semantic-channel audit, structural audit, and other predeclared checks retain their own gates.

## Reproduction conditions

- Embedding-cache integrity: **16/16 PASS**.
- C2 model × channel conditions completed: **16/16**.
- D model × channel conditions completed: **16/16**.
- Numerical comparison tolerance: **1 × 10⁻⁶**.
- Environment recorded, including Python, platform, NumPy, pandas, installed packages, and SHA-256 hashes for the runner and C2/D analysis scripts.

## Independently verified exact matches

Every compared file below has identical row count, byte size, and SHA-256 hash between the frozen reference and clean reproduction. Consequently, each has zero differing cells and zero numerical error.

### C2 / S-side

| File | Rows | Result |
|---|---:|---|
| `edge_offset_scores_v5.csv` | 66,724 | Exact match |
| `semantic_relation_summary_v5.csv` | 4,076 | Exact match |
| `forward_analogy_retrieval_v5.csv` | 63,232 | Exact match |
| `inverse_anchor_recovery_v5.csv` | 63,232 | Exact match |
| `retrieval_identification_summary_v5.csv` | 48 | Exact match |
| `null_model_results_v5.csv` | 494,476 | Exact match |
| `residual_family_coherence_v5.csv` | 8,984 | Exact match |

### D / P-side

| File | Rows | Result |
|---|---:|---|
| `p_side_forward_retrieval_v1.csv` | 54,980 | Exact match |
| `p_side_retrieval_identification_summary_v1.csv` | 48 | Exact match |
| `p_side_channel_coverage_v1.csv` | 16 | Exact match |
| `p_vs_s_side_matched_retrieval_summary_v1.csv` | 48 | Exact match |

These matches cover the primary row-level retrieval outcomes, S/P matched comparison, summary statistics, relation evidence, residual-family evidence, and frozen null results.

## Resolution of the apparent failed comparison

An earlier runner summary showed one failed C2 comparison because its manifest requested:

`residual_family_comparison_v5.csv`

That filename does not exist in either the frozen reference or the reproduced output. This was a comparison-manifest naming mistake, not a missing scientific output or numerical mismatch.

The actual residual artifact is:

`residual_family_coherence_v5.csv`

It was discovered and compared separately:

- reference rows: 8,984;
- reproduced rows: 8,984;
- differing cells: 0;
- maximum absolute numerical error: 0;
- SHA-256 hashes: identical.

The corrected QC therefore properly marks the nonexistent requested filename as `NOT_APPLICABLE` and the actual residual comparison as `PASS`. This is a bounded clerical defect in the original comparison manifest and has no effect on C2 estimates, D estimates, matched S-versus-P results, or scientific interpretation.

## What this establishes

The clean run establishes that, given the same frozen inputs and validated embedding caches:

1. the C2 and D analysis code regenerates the same eligible cases;
2. the learned relation vectors lead to the same retrieval ranks;
3. the primary and diagnostic summaries are deterministic;
4. the S-side versus P-side interpretation is not an artifact of an unreproducible prior execution;
5. the null and residual-family outputs are exactly recoverable.

## Boundary of the claim

This is **analysis-pipeline reproducibility from frozen embedding caches**. It is not a claim that regenerating embeddings from newly downloaded model weights, a different GPU stack, or future library versions must be bit-for-bit identical. The environment lock and cache hashes are therefore essential parts of the frozen release.

For the paper and repository, use wording such as:

> A clean execution from integrity-verified frozen embedding caches reproduced all compared primary C2 and D outputs exactly, including row counts, cell values, and SHA-256 hashes.

Do not broaden this to unrestricted cross-hardware or future-version bitwise reproducibility unless separately tested.

## Freeze consequence

**Step 10 clean-reproduction gate: PASS.**

Required housekeeping correction before release: retain the corrected comparison table and document that `residual_family_comparison_v5.csv` was an erroneous manifest label replaced by comparison of the actual `residual_family_coherence_v5.csv`. No analytical rerun or change to the scientific findings is required.

