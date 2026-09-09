# Phase IV-C Raw Offset QC v5

Updated UTC: `2026-09-07T19:59:57.332975+00:00`

- Completed model/channel checkpoints: 16
- Current combination: `qwen3_4b_2560d_int8__en_def_translated_cn_def`
- Channel-eligible characters: 6,040
- Documented direct structural edges: 6,195
- Offset-eligible direct edges (both C and P embedded): 3,160
- Structurally documented but offset-ineligible here: 3,035
- Immediate families used: 900
- Relation classes: 238
- Root anchor families: 700
- Held-out edges with >= 3 other training root families: 2,960
- Held-out edges with >= 5 other training root families: 2,883
- Held-out edges with >= 8 other training root families: 2,790
- Outside-population anchors in frozen v2: 714; available in this channel: 0
- Normalization-corrected edges used: 30
- Disputed edges used: 59
- Duplicate canonical edges rejected: 0
- Graph/lineage validation: PASS (no self-edges, duplicate derived endpoints, or cycles)
- Corrected `青` family: PASS (18 canonical edges; no canonical `靑` anchors)
- Frozen edge SHA-256: `1e7c972c120bec39d59998b97f45a8dd3c584c37cf888a5f85f43d06087b9f39`
- Frozen census SHA-256: `19192ccdda875f755abfb83f2deba9d5884c3163691c733d49b4f72aae712cb0`
- Primary vectors: raw, uncentered `E(x)`; no PCA, CCA, SVCCA, or learned W.
- Prototype weighting: edges averaged within root family, then family means averaged.
- Direct relations come exclusively from `relational_direct_edges_v2.csv`; recursive lineage depth is retained but edges are never collapsed.
- Google direct-character gloss is blocked unless explicitly enabled as exploratory.
- Identification headline: `argmax_P cos(E(P), E(C*) - r_S) = P*`, evaluated on unseen root families.
- Deterministic ties: frozen `character_index.csv` order; first maximum wins; ranks use the identical ordering.
- Baseline and offset candidate universes are identical; exact argmax/rank/Top-1 invariants: PASS.
