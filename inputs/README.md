# Inputs

This folder contains the frozen datasets used by the published Phase IV relational-offset analyses.

The files here define the analyzed character population, meaning representations, and component relationships. They should be treated as immutable inputs. Corrections belong in a new version, not as silent replacements.

## Files

| File | Purpose |
|---|---|
| `relational_character_master_5channel_translated_v2.csv` | Master table for 11,151 Traditional Chinese characters, including the representation channels used to create embeddings. |
| `relational_direct_edges_v2.csv` | Frozen direct semantic-part, phonetic-part, and derived-character relationships used by the reproduced Phase IV analysis. |
| `Traditional_Chinese_Component_Family_Universe_11151_v3.xlsx` | Auditable workbook containing the expanded character and component-family universe. |
| `Phase_IV_A3_Normalization_Correction_QC_v1.md` | Quality-control record for component normalization and edge correction before the analyzed v2 edge freeze. |

## Version boundary

The primary and centered Phase IV results were calculated from `relational_direct_edges_v2.csv`. A later audit found that this high-precision edge inventory is not exhaustive. In particular, `清 = 氵 + 青` was missing because its source decomposition did not explicitly mark 青 as phonetic.

The analyzed v2 file remains here for reproducibility. Subsequent edge-audit files belong in `edge_audit/` until a complete rerun is released.

## Data-source note

The datasets draw on Taiwan Ministry of Education character inventories and definitions, Unicode Unihan glosses, CJKVI decompositions, and Google Cloud Translation. Review upstream licenses and redistribution terms before publishing the source-derived tables.
