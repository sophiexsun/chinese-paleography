# Semantic-Part Results

This folder contains the primary relational-offset analysis for conventionally labeled semantic parts.

For a held-out phonetic family, the analysis learns a recurring semantic-part pattern from other families and tests whether adding it to the held-out phonetic anchor moves the correct compound character upward in the candidate list.

This direction also serves as a positive methodological check: well-established semantic components should produce recurring meaning-related patterns if the method is functioning sensibly.

## Files

| File | Contents |
|---|---|
| `edge_offset_scores_v5.csv` | Edge-level offset vectors and similarity statistics. |
| `semantic_relation_summary_v5.csv` | Aggregated recurrence statistics for semantic-component relations. |
| `forward_analogy_retrieval_v5.csv` | Row-level held-out forward-retrieval results. |
| `inverse_anchor_recovery_v5.csv` | Secondary inverse test recovering the component anchor from the compound relation. |
| `retrieval_identification_summary_v5.csv` | Top-k, rank, and identification summaries by model, representation, and threshold. |
| `relation_learning_curves_v5.csv` | Effect of relation-support size on learned-pattern stability. |
| `residual_family_coherence_v5.csv` | Remaining family coherence after accounting for learned semantic-part patterns. |
| `matched_control_results_v5.csv` | Matched-control comparison results. |
| `matched_subset_relation_summary_v5.csv` | Relation summaries on matched eligible subsets. |
| `cross_model_relation_summary_v5.csv` | Replication and agreement across embedding models. |
| `cross_channel_relation_summary_v5.csv` | Comparison across written-character, Chinese-definition, translated-definition, and Unihan-gloss representations. |
| `cross_language_relation_summary_v5.csv` | Cross-language summary of learned relations. |
| `matched_cross_channel_inverse_recall_v5.csv` | Matched inverse-retrieval comparison across representations. |
| `diagnostic_identification_cases_v5.csv` | Human-readable successes, failures, predicted characters, and ranks. |
| `channel_coverage_and_eligibility_v5.csv` | Eligible-case counts and exclusions by model and representation. |
| `null_model_results_v5.csv` | Frozen shuffled-label comparison results. |
| `raw_offset_method_manifest_v5.json` | Machine-readable analysis rules and parameters. |
| `input_fingerprints_v5.json` | Hashes and identifiers for the exact analysis inputs. |
| `raw_offset_QC_v5.md` | Quality-control checks and completion status. |

## Optional binary artifact

`semantic_relation_prototypes_v5.npz` stores learned prototype arrays. Because it is binary and potentially large, it may be placed in a versioned GitHub Release rather than tracked in the main repository.

## Headline result

In Chinese definitions, 104 of 109 eligible semantic-component relation classes, or 95.4%, showed stronger recurrence than the shuffled comparison across the four models.
