# Phonetic-Part Results

This folder contains the primary relational-offset analysis for conventionally labeled phonetic parts and the matched comparison with semantic-part performance.

Within a phonetic family, the analysis learns a recurring pattern from other semantic contexts. It then adds that pattern to the independent semantic component of a held-out case and tests whether the correct derived character moves upward in the candidate list.

This is the experiment that directly tests whether a phonetic family contains transferable semantic information beyond visible shared character structure.

## Files

| File | Contents |
|---|---|
| `p_side_forward_retrieval_v1.csv` | Row-level baseline and offset retrieval outcomes for held-out phonetic-family cases. |
| `p_side_retrieval_identification_summary_v1.csv` | Top-k, rank, and mean-reciprocal-rank summaries by model, representation, and threshold. |
| `p_side_diagnostic_cases_v1.csv` | Human-readable successes, failures, predicted characters, and position changes. |
| `p_side_channel_coverage_v1.csv` | Eligible cases and family coverage by representation. |
| `p_vs_s_side_matched_retrieval_summary_v1.csv` | Direct semantic-part versus phonetic-part comparison on identical held-out cases. |
| `p_side_method_manifest_v1.json` | Machine-readable analysis rules and parameters. |
| `p_side_input_fingerprints_v1.json` | Hashes and identifiers for the exact inputs. |
| `p_side_QC_v1.md` | Validation checks and run-completion status. |

## Headline result

The phonetic-part pattern improved a majority of held-out cases across all four models in the Unihan English-gloss representation. On matched cases, an average of 60.2% moved upward, and Recall@10 improved in every model.

The result supports transferable phonetic-part information, but it is weaker and more representation-dependent than semantic-part transfer. It does not imply that every phonetic family carries one stable meaning.
