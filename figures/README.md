# Figures

This folder contains a compact visual summary of the project for GitHub, LinkedIn, presentations, and other public explanations.

The figures use the public terms **semantic part** and **phonetic part**. These are conventional labels. The experiment tests whether the phonetic part also contributes semantic information.

## Main figures

| File | Message |
|---|---|
| `01_semantic_relations_recur_above_null.png` | Semantic-component patterns recur more strongly than a shuffled comparison. This validates the relational method; it is not direct evidence about phonetic parts. |
| `02_both_components_improve_retrieval.png` | Both learned component patterns generally move the correct character upward. |
| `03_asymmetric_composition_strength.png` | Semantic-part improvement is much larger than phonetic-part improvement. Composition is asymmetric, not one-sided. |
| `04_p_side_replication_across_models.png` | Adding the phonetic-part pattern improves Top-10 retrieval in short English glosses across all four models. |
| `05_exact_clean_reproduction.png` | The primary outputs reproduced exactly from integrity-verified caches. |
| `06_qing_sunny_component_tests.png` | Focused results for 晴 = 日 + 青 in the two test directions. |
| `07_qing_family_phonetic_transfer.png` | Transfer of the learned 青 pattern across eligible 青-family characters. |

## Technical overview figures

| File | Purpose |
|---|---|
| `Phase_III_ToolStudy_Findings_Dashboard_v1.png` | Summary of embedding anisotropy, effective dimensionality, and cross-model alignment. |
| `Phase_IV_C2_D_4x4_Analyzer_Charts_v1_contact_sheet.png` | Compact overview of the full model-by-representation relational-offset results. |

## Interpretation notes

- Figure 01 concerns semantic-part relations.
- Figure 03 compares the size of the added improvement from each component pattern.
- Figure 04 isolates the phonetic-part experiment. Its grey bar is the baseline from the semantic component alone, not the incremental semantic-part effect.
- 清 was not a tested success in the frozen v2 analysis. Use 晴 for the measured 青 example.

The plotted values and supporting tables are available in `../tables/`.
