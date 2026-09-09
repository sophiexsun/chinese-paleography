# Scripts

This folder contains the final executable scripts used across the pilot, semantic-coherence, embedding-diagnostic, translation, and relational-offset stages.

Only the final retained version of each script is included. Earlier development versions are omitted to keep the public workflow understandable.

## Files

| File | Phase and purpose |
|---|---|
| `pilot14_fsc_colab_test_v5.py` | Phase I pilot test of Family Semantic Coherence for the 青 family. |
| `Phase_II_D_10Family_FSC_Colab_v1.py` | Phase II semantic-coherence analysis for ten phonetic families with matched controls. |
| `Phase_III_ToolStudy_4808_4Model_v2.py` | Phase III anisotropy and effective-dimensionality diagnostics across four embedding models. |
| `Phase_III_ToolStudy_C1_RSA_v3.py` | Phase III cross-model representational similarity analysis. |
| `Phase_III_ToolStudy_C2_SVCCA_v3.py` | Phase III cross-model SVCCA analysis. |
| `Phase_IV_B_Google_Translation_5Channel_v2.py` | Builds and validates the translated meaning-representation channel. |
| `Phase_IV_C1_Build_Raw_Embedding_Caches_v5.py` | Generates the frozen raw embedding caches for four models and four representations. |
| `Phase_IV_C2_Raw_E_Relational_Offset_Analysis_v5.py` | Runs the semantic-part relational-offset analysis, controls, null comparisons, and retrieval tests. |
| `Phase_IV_D_Raw_E_P_Side_Relational_Composition_v1.py` | Runs the phonetic-part relational-offset analysis and matched comparison with the semantic-part test. |
| `analyze_centered_sensitivity_v1.py` | Compares raw and mean-centered outcomes and quantifies robustness. |

## Running the code

Use the frozen inputs and package versions recorded in `../qc/environment_lock_v1.json`. Verify input fingerprints and cache hashes before interpreting results.

The raw, uncentered analysis is primary. Centering is a sensitivity analysis. Do not change eligibility thresholds, candidate universes, or held-out rules when attempting to reproduce the reported results.

API credentials are not included. Supply Google Cloud credentials through a secure environment configuration and never commit keys to the repository.
