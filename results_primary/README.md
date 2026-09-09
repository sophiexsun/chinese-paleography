# Primary Results

This folder contains the frozen raw, uncentered Phase IV relational-offset results.

The analysis has two complementary directions:

- `semantic_part/` tests whether a semantic-part pattern learned across other phonetic families helps retrieve a held-out derived character.
- `phonetic_part/` tests whether a phonetic-part pattern learned across other semantic contexts helps retrieve a held-out derived character.

The mathematical forms are parallel, but the effects are not equal. The semantic-part contribution is broader and much larger. The phonetic-part contribution is smaller, heterogeneous, and clearest in short English glosses.

## Subfolders

| Folder | Contents |
|---|---|
| `semantic_part/` | Edge-level relations, held-out retrieval, null comparisons, control results, summaries, fingerprints, and QC. |
| `phonetic_part/` | Held-out phonetic-family retrieval, model and representation summaries, diagnostics, matched semantic-versus-phonetic comparison, fingerprints, and QC. |

## Naming note

Some frozen filenames use the internal terms `s_side` and `p_side`. In the public explanation:

- `S` or `s_side` means the conventionally labeled semantic part;
- `P` or `p_side` means the conventionally labeled phonetic part.

The conventional labels describe the experimental roles and do not predetermine whether the phonetic part also carries semantic information.
