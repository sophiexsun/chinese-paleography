# The 109 Eligible Meaning-Component Patterns

This file documents the population summarized in `01_semantic_relations_recur_above_null.png`.

## What the 109 items are

Each item is a semantic component whose learned change in meaning could be compared across at least three distinct phonetic-part families in the Chinese dictionary-definition representation. “Semantic” and “phonetic” are the conventional labels; they do not predetermine the experiment’s conclusion. These are component-pattern classes, not 109 individual characters and not 109 phonetic-part families.

For each eligible component, the analysis:

1. measured how similarly its meaning-change pattern recurred across different sound-part families;
2. repeated the calculation after shuffling the family labels; and
3. subtracted the shuffled result from the observed result.

The figure averages each component’s result across the four embedding models. A positive value means the observed cross-family recurrence was stronger than the shuffled comparison.

## Column guide

| Column | Meaning |
|---|---|
| `rank_by_similarity_above_shuffled` | Rank from strongest to weakest result above the shuffled comparison |
| `meaning_labeled_component` | The conventionally classified semantic component being tested |
| `observed_cross_family_similarity_mean` | Average observed similarity of its learned pattern across sound-part families |
| `shuffled_label_similarity_mean` | Average similarity after family labels were shuffled |
| `similarity_above_shuffled` | Observed value minus shuffled value; this is the figure’s vertical axis |
| `phonophore_family_count` | Number of distinct sound-part families contributing to the test |
| `models_evaluated` | Number of embedding models included |
| `above_shuffled` | Whether the difference was positive |

The complete 109-row table is provided in `109_Eligible_Meaning_Component_Patterns_v1.csv`.
