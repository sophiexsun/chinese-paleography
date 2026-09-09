# Phase IV-D P-Side Relational Composition QC v1

Updated UTC: `2026-09-07T20:03:46.544985+00:00`

- Current condition: `qwen3_4b_2560d_int8__en_def_translated_cn_def`
- Frozen A4 mapping validation: PASS (306 S classes; 236 primary-eligible high-confidence A/B mappings)
- Channel-eligible characters: 6,040
- Literal P-side offset-eligible edges: 3,259
- Held-out cases with >=3 / >=5 / >=8 OTHER distinct S contexts: 2,398 / 1,804 / 1,112
- Mapped independent-S characters requested / available / unavailable: 218 / 199 / 19
- Holdout leakage check: PASS; the held-out canonical S context is wholly excluded from `r_P`.
- Weighting check: PASS; edges are averaged within canonical S, then distinct S contexts are equally weighted.
- Baseline/offset candidate universe and held-out cases: identical.
- Exact cosine argmax, rank, Top-1 identity, bounds, and deterministic ties: PASS.
- Matched S-side comparison: available.
- Primary vectors: raw, uncentered `E`; no learned W, PCA, CCA, or SVCCA.

Human-facing lead result: Across 2,398 held-out members of P families, the correct derived character was rank 1 in 0.46% using S alone and 0.54% after adding r_P.
