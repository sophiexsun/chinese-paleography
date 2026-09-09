# Relational Edge Audit

This folder contains the post-analysis audit of the relational edge inventory and its proposed v3 correction.

These files document improvements to dataset coverage. They are not inputs to the currently published v2 Phase IV results unless a corresponding v3 model rerun is also released.

## Files

| File | Purpose |
|---|---|
| `relational_direct_edges_v3.csv` | Audited active edge table containing confirmed v3 relationships. |
| `relational_direct_edges_v3_change_log.csv` | Row-level record of additions, corrections, removals, disputes, and unchanged v2 edges. |
| `Phase_IV_Edge_Completeness_Audit_Report_v2.md` | Explanation of the audit method, evidence, coverage, limitations, and disposition counts. |
| `Phase_IV_Edge_v3_QC.json` | Machine-readable validation results for the v3 edge release. |
| `Phase_IV_Edge_v3_SHA256.txt` | SHA-256 checksums for the v3 audit files. |

## Interpretation rule

Do not mix v3 edges with v2 results. Until v3 has been run through the same model and representation matrix, report it as a subsequent data-quality audit rather than as the basis of the published numerical findings.

Candidates generated from structure or modern pronunciation are not automatically valid phonetic relationships. Confirmed additions require independent linguistic evidence.
