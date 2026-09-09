# Publication Tables and Figure Data

This folder contains compact, human-readable summaries and the exact values used in the public figures.

The large row-level scientific outputs remain in `../results_primary/` and `../results_sensitivity/`. This folder is intended for readers who want to inspect the reported aggregates without loading the complete analysis tables.

## Files

| File | Contents |
|---|---|
| `109_Eligible_Meaning_Component_Patterns_v1.csv` | Complete 109-row population underlying the semantic-relation recurrence figure. |
| `109_Eligible_Meaning_Component_Patterns_Guide_v1.md` | Method and column guide for the 109 eligible component patterns. |
| `01_p_side_condition_summary.csv` | Phonetic-part performance summary by model and representation. |
| `03_matched_s_vs_p_rank_improvement.csv` | Matched semantic-part versus phonetic-part rank-improvement comparison. |
| `04_matched_s_vs_p_recall10_gain.csv` | Matched Recall@10 gains for both component directions. |
| `05_p_side_mean_rank_gain_heatmap.csv` | Values behind the phonetic-part model-by-representation heatmap. |
| `06_p_side_family_effects.csv` | Family-level phonetic-part effect summaries. |
| `07_p_side_replicated_cases.csv` | Cases that replicate across models or representations. |
| `visual_metrics_v1.json` | Exact plotted values for the main public-facing figures. |
| `qing_family_visual_metrics_v1.json` | Exact plotted values for the focused 青-family figures. |
| `Phase_IV_C2_D_4x4_Analyzer_Report_v1.md` | Full interpretation of the primary four-model by four-representation results. |
| `Qing_Family_Focused_Analyzer_Report_v1.md` | Focused analysis of 青-family outcomes, including the 晴 result and the missing 清 edge. |

## Naming note

Frozen computational filenames retain `s_side` and `p_side` for reproducibility. In public prose, these correspond to **semantic part** and **phonetic part**.

The JSON files are not narrative documents. They exist so the figures can be audited and regenerated from exact recorded values.
