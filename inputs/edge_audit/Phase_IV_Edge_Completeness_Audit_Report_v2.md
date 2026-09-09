# Phase IV Edge Completeness Audit Report v2

## Outcome

The frozen v2 inputs passed exact hash verification. The audit preserves v2 unchanged and produces a conservative v3 with **6,196 active direct edges**: 6,195 inherited rows plus one independently confirmed addition, `青 → 清` with semantic component `氵`.

No edge was added from modern pronunciation alone. The remaining generated candidates are review queues, not ground truth.

## Required input verification

| File | SHA-256 | Status |
|---|---|---|
| `relational_direct_edges_v2.csv` | `1e7c972c120bec39d59998b97f45a8dd3c584c37cf888a5f85f43d06087b9f39` | PASS |
| `relational_character_master_5channel_translated_v2.csv` | `dbaa165b6843bb687659fce81dedc139775a6f9b98f3cbfda952e0ba14434fc5` | PASS |
| `Traditional_Chinese_Component_Family_Universe_11151_v3.xlsx` | `19192ccdda875f755abfb83f2deba9d5884c3163691c733d49b4f72aae712cb0` | recovered frozen copy |
| `Phase_IV_A3_Normalization_Correction_QC_v1.md` | `5f113e71e02d11576b83f7579f8f40aeef092efb2d439f31525100465179fc38` | recovered frozen copy |

## Candidate enumeration

- All 11,151 master rows received exactly one character-level disposition.
- IDS expressions were parsed recursively using their top-level operator arity.
- Both semantic/phonetic orientations of top-level components were enumerated.
- Candidate generation did not require a `聲` annotation.
- Indexing-radical allograph agreement was retained only as prioritization evidence.
- 22,396 candidate orientations were recorded; unvalidated cases remain excluded.

## V3 decision

### Added: `青 → 清`

- Structure: `清 = ⿰氵青`
- Semantic component: `氵` (positional form of `水`)
- Phonetic anchor: `青`
- Taiwan MOE Dictionary of Chinese Character Variants: `从水，青聲`
- Historical corroboration: *Shuowen Jiezi* entry records `从水青聲`
- Modern Mandarin identity (`qīng/qīng`) was used only as corroborative screening evidence.

Sources:

- https://dict.variants.moe.edu.tw/dictView.jsp?ID=24025
- https://ctext.org/dictionary.pl?char=%E6%B8%85&if=gb

## Existing v2 audit

All 6,195 inherited edges passed mechanical endpoint, canonical-key, self-edge, cycle, and normalization checks. Clean rows retain their explicit CJKVI `聲/亦聲` source status. The 118 previously disputed rows remain visibly disputed rather than being promoted to confirmed primary evidence.

This pass did **not** claim a second independent scholarly validation for every inherited edge. The source-audit file makes that boundary explicit. A publication requiring two independent sources per legacy edge would need a separate multi-source lexicographic campaign.

## Character dispositions

| Disposition | Count |
|---|---:|
| `confirmed_edge` | 6,078 |
| `confirmed_non_edge` | 142 |
| `ambiguous_or_disputed` | 118 |
| `component_not_in_population` | 2,549 |
| `structurally_unresolved` | 31 |
| `insufficient_external_evidence` | 2,233 |


Total: **11,151**.

## QC

- Active canonical duplicate keys: 0
- Self-edges: 0
- Direct graph cycles: 0
- Family census edges: 6,196 (reconciles to v3)
- `清 = 氵 + 青`: explicit confirmed record present
- Output decisions use no embeddings, cosine scores, or retrieval outcomes

## Modeling handoff

Use `relational_direct_edges_v3.csv` as the frozen active table and retain v2 as the immutable baseline. The change log isolates the added `清` edge and inherited disputed rows. Unconfirmed candidates remain outside the primary test and may be used only in a separately labeled sensitivity analysis after independent linguistic review.
