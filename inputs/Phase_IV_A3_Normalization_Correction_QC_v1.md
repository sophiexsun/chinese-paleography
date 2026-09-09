# Phase IV-A3 — Relational Edge Normalization Correction QC v1

## Outcome

- Frozen population: **11,151** (unchanged).
- Structural edge rows audited: **6,195**.
- Unique source phonetic labels/components and endpoints recorded in normalization map: **7,927** rows.
- Accepted unique mappings: **10**; unresolved map rows: **200**; rejected map rows: **212**.
- Edge rows changed only by normalized structural labels: **41**.

## Before versus after

| Metric | v1 | v2 | Change |
|---|---:|---:|---:|
| edges | 6,195 | 6,195 | +0 |
| immediate_families | 1,427 | 1,421 | -6 |
| root_lineages | 1,114 | 1,001 | -113 |
| S_classes | 307 | 306 | -1 |
| anchors_outside | 751 | 714 | -37 |
| cycles | 0 | 0 | +0 |

- False outside-population anchor rows recovered: **37**.
- Canonical duplicates introduced: **0** rows in collapsed groups. No rows were silently deleted.
- Family membership records changed: **20** anchors (full audit in `family_membership_changes_v1.csv`).

## Accepted mappings

- `兹 → 茲`
- `册 → 冊`
- `呉 → 吳`
- `壮 → 壯`
- `奥 → 奧`
- `戸 → 戶`
- `郞 → 郎`
- `鄕 → 鄉`
- `靑 → 青`
- `黒 → 黑`

## 青 diagnostic

- Before: canonical `青` edge/root records = **0**; raw `靑` anchor records = **18**.
- After: canonical `青` edge/root records = **18**; canonical `靑` anchor records = **0**.
- Explicit path: `靑聲` (raw source label) → `靑` (parsed source component) → `青` (frozen-population component).
- Raw source labels, raw endpoints, and raw IDS patterns remain in dedicated v2 columns.

## Interpretation safeguards

- Unihan semantic or specialized-semantic variants were not treated as structural identity.
- Simplified/traditional mappings that may name a subcomponent rather than the whole population character remain unresolved unless CJKVI and strong Unicode orthographic evidence jointly support the exact mapping.
- Genuine outside-population components remain outside the sampling frame.
- No embeddings, offsets, retrieval, W-learning, or significance tests were run.

## Phase IV-C handoff hashes

- `relational_direct_edges_v2.csv`: `1e7c972c120bec39d59998b97f45a8dd3c584c37cf888a5f85f43d06087b9f39`
- `Traditional_Chinese_Component_Family_Universe_11151_v3.xlsx`: `19192ccdda875f755abfb83f2deba9d5884c3163691c733d49b4f72aae712cb0`

Phase IV-C must consume the frozen `relational_direct_edges_v2.csv` directly and must not reconstruct edges independently from the character master.
