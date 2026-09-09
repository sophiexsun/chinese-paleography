# Sensitivity Results

This folder tests whether the primary conclusions depend on the shared mean direction of each embedding space.

The raw, uncentered analysis remains primary. The sensitivity analysis subtracts the model-by-representation mean vector before recomputing the relational-offset tests. It evaluates direction, magnitude, family-level concordance, and semantic-part versus phonetic-part interpretation.

## Files

| File | Contents |
|---|---|
| `raw_vs_centered_C2_summary_v1.csv` | Raw versus centered comparison for the semantic-part analysis. |
| `raw_vs_centered_D_summary_v1.csv` | Raw versus centered comparison for the phonetic-part analysis. |
| `raw_vs_centered_S_family_effects_v1.csv` | Family-level semantic-part effects under both representations. |
| `raw_vs_centered_P_family_effects_v1.csv` | Family-level phonetic-part effects under both representations. |
| `centered_sensitivity_QC_v1.json` | Completion and consistency checks for all centered conditions. |
| `centered_sensitivity_output_SHA256_v1.csv` | Checksums for centered outputs. |
| `embedding_cache_integrity_manifest_v1.csv` | Cache identifiers, dimensions, and hashes. |
| `embedding_cache_integrity_QC_v1.json` | Integrity verdict for all 16 caches. |
| `Phase_IV_Centered_Sensitivity_Analyzer_Assessment_v1.md` | Plain-language robustness assessment and interpretation boundaries. |

## Conclusion

Centering preserves and generally strengthens the core findings:

- semantic-part improvement remains strong;
- phonetic-part majority improvement rises from 15 of 16 raw model-by-representation cells to 16 of 16 centered cells;
- semantic-part transfer remains substantially stronger;
- the raw written-character split between Qwen and BGE/GTE disappears.

The last result means the earlier model split was sensitive to global embedding geometry and should not be described as an intrinsic architecture difference.

Complete centered row-level outputs and checkpoints may be distributed as `Phase_IV_Validation_Centered_Results_v1.zip` in a GitHub Release.
