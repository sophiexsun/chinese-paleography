# Quality Control and Reproduction

This folder contains the environment record, cache-integrity checks, and clean-reproduction comparisons for the frozen Phase IV analysis.

The clean run started from the frozen inputs and integrity-verified embedding caches. It regenerated the primary semantic-part and phonetic-part results and compared them with the reference outputs.

## Files

| File | Contents |
|---|---|
| `environment_lock_v1.json` | Python, platform, package, script, and environment information recorded for the clean run. |
| `embedding_cache_integrity_manifest_v1.csv` | Model, representation, dimension, row-count, and hash record for all embedding caches. |
| `embedding_cache_integrity_QC_v1.json` | Pass/fail integrity summary for the 16 caches. |
| `clean_reproduction_C2_comparison_v1.csv` | File-level comparison of reproduced semantic-part outputs with the frozen reference. |
| `clean_reproduction_C2_residual_comparison_v1.csv` | Corrected comparison for `residual_family_coherence_v5.csv`. |
| `clean_reproduction_D_comparison_v1.csv` | File-level comparison of reproduced phonetic-part outputs with the frozen reference. |
| `clean_reproduction_QC_v1.json` | Overall clean-run completion and comparison verdict. |
| `clean_reproduction_output_SHA256_v1.csv` | SHA-256 hashes of reproduced outputs. |
| `Phase_IV_Clean_Reproduction_Analyzer_Assessment_v1.md` | Independent explanation of the reproduction result and its boundary. |

## Verdict

All 16 embedding caches passed integrity checking. All 11 scientifically relevant compared outputs reproduced exactly, with identical row counts, zero differing cells, zero numerical error, and identical SHA-256 hashes.

An earlier apparent failure was a comparison-manifest filename error: it requested nonexistent `residual_family_comparison_v5.csv`. The actual file, `residual_family_coherence_v5.csv`, reproduced exactly and is documented in the corrected comparison.

This establishes exact analysis-pipeline reproduction from the frozen embedding caches. It does not promise bit-for-bit embedding regeneration under future model downloads, hardware, drivers, or library versions.
