# Do Phonetic Components in Chinese Characters Also Carry Meaning?

An AI-assisted computational paleography study of semantic structure in Traditional Chinese characters.

## Project overview

Chinese characters classified as **semantic-phonetic compounds** (形聲字) are conventionally described as combining:

- a **semantic part**, which suggests a meaning category; and
- a **phonetic part**, which suggests pronunciation.

For example:

- 清, “clear,” combines 氵 (water) with 青;
- 晴, “clear weather,” combines 日 (sun) with 青.

The conventional labels are useful, but they may describe an asymmetry rather than an exclusive division of labor. This project tests whether a phonetic part such as 青 can also contribute transferable semantic information across the characters that contain it.

The central question is:

> When the semantic part changes, does a recurring meaning-related signal associated with the phonetic part remain detectable?

Rather than relying only on selected examples, the project represents thousands of Traditional Chinese characters and their dictionary meanings with multilingual embedding models, then tests whether component-associated relationships recur and transfer to held-out characters.

## Why this matters

The study brings a long-running idea in Chinese philology into a testable computational framework. It is related to 右文說 and 聲符示源, traditions that treat many phonetic components as preserving traces of semantic origin rather than functioning as arbitrary sound markers.

The project does **not** assume that every phonetic component carries one fixed meaning, or that modern embedding structure proves how a character was historically created. It asks a narrower synchronic question: whether semantic information associated with a component is recoverable across present-day character meanings.

## Research design

### Character population

The project began with the 4,808-character Taiwan Ministry of Education Common Standard inventory and later expanded to an audited working universe of **11,151 Traditional Chinese characters**.

The current relational dataset contains **6,195 direct component edges**. For the primary Chinese-definition relational-offset analysis, **2,960 derived characters** met the held-out eligibility rule. The matched semantic-part versus phonetic-part comparison contained **1,933 characters**.

Eligibility differs by representation because not every character has every form of definition or gloss.

| Representation | Semantic-part test | Phonetic-part test | Matched cases |
|---|---:|---:|---:|
| Chinese definitions | 2,960 | 2,398 | 1,933 |
| English translations of Chinese definitions | 2,960 | 2,398 | 1,933 |
| Unihan English glosses | 4,648 | 4,186 | 3,636 |
| Written character alone | 5,240 | 4,763 | 4,166 |

### Meaning representations

Each eligible character was tested through four channels:

1. the written Chinese character;
2. a Chinese dictionary definition;
3. an English translation of that Chinese definition;
4. a short English gloss from Unihan.

The three definition or gloss channels do not contain the target written character as the input. They therefore help distinguish semantic transfer from visual recognition of shared character structure.

### Embedding models

The full experiment was run across four multilingual embedding models:

- Qwen3-Embedding-0.6B;
- Qwen3-Embedding-4B, INT8;
- BGE-M3;
- GTE Multilingual Base.

This produces a 4-model × 4-representation design rather than relying on a single model or a single definition source.

## From family coherence to relational offsets

The project developed in four stages.

### Phase I: pilot study

The 青 family was used to test whether character meanings within one phonetic family are more coherent than matched random controls.

### Phase II: multi-family semantic coherence

The family-coherence test was expanded to ten phonetic families with family-size-matched controls. This phase also produced the first Semantic Atlas, a visual map of character meanings in embedding space.

### Phase III: embedding-space diagnostics

The four model spaces were examined for:

- anisotropy, or clustering around shared global directions;
- effective dimensionality through PCA;
- cross-model agreement through RSA and SVCCA.

These diagnostics helped separate linguistic findings from properties of individual embedding geometries.

### Phase IV: held-out relational-offset tests

Semantic similarity within a family is suggestive, but it cannot show which component contributes what. Phase IV therefore tests learned component relationships directly.

Let:

- \(C\) be a compound character;
- \(S\) be its conventionally labeled semantic part;
- \(P\) be its conventionally labeled phonetic part;
- \(E(X)\) be the embedding of the character or its meaning representation.

#### Semantic-part test

A recurring relation for a semantic part is learned from other phonetic families and applied to a held-out family:

$\[
r_S^{(-F^*)}=\operatorname{mean}\left[E(S+P)-E(P)\right]
\]$

$\[
\operatorname*{arg\,max}_{C}\cos\left(E(C), E(P^*)+r_S^{(-F^*)}\right)=C^*
\]$

#### Phonetic-part test

A recurring relation for a phonetic part is learned from other semantic contexts and applied to a held-out semantic part:

$\[
r_P^{(-S^*)}=\operatorname{mean}\left[E(S+P)-E(S_{independent})\right]
\]$

$\[
\operatorname*{arg\,max}_{C}\cos\left(E(C), E(S_{independent}^*)+r_P^{(-S^*)}\right)=C^*
\]$

In plain language, the test asks whether adding a learned component pattern moves the correct held-out character upward in a candidate list. The baseline and offset queries use the same candidates, and the held-out case is excluded from learning its own relation.

## Main findings

### 1. Component relationships recur

In Chinese definitions, **104 of 109 eligible semantic-component relation classes, or 95.4%, showed stronger recurrence than a shuffled-label comparison** across the four models.

This establishes that the method can recover a well-attested compositional signal before asking the more controversial question about phonetic components.

### 2. Both components contribute transferable information

On identical held-out cases, adding either learned component pattern usually improved the position of the correct character. The semantic-part effect was much broader and larger.

Average share of matched cases that moved upward across four models:

| Representation | Semantic-part pattern | Phonetic-part pattern |
|---|---:|---:|
| Chinese definitions | 85.4% | 53.8% |
| English translated definitions | 84.0% | 52.8% |
| Unihan English glosses | 87.3% | 60.2% |
| Written character alone | 81.0% | 68.3% |

The evidence therefore supports **partially symmetric but empirically asymmetric composition**: both parts can contribute, but not equally.

### 3. Phonetic-part transfer survives without the written character

The strongest phonetic-part evidence appears in short Unihan English glosses. Across all four models, the phonetic-part pattern:

- improved more cases than it worsened;
- improved median and mean candidate position;
- increased Recall@1, Recall@10, and mean reciprocal rank;
- produced positive family-level effects in a majority of tested families.

Because the target character itself is absent from these gloss inputs, direct visual containment cannot explain the result.

Chinese definitions show a smaller positive tendency. English translations of those definitions are weak and mixed. The phonetic-part effect is therefore real but heterogeneous and representation-dependent.

### 4. 青 provides a human-readable example

For 晴 = 日 + 青, across four models and three meaning-based representations:

- the learned 日 semantic-part pattern improved retrieval in 12 of 12 tests;
- the learned 青 phonetic-part pattern improved retrieval in 11 of 12 tests.

Across the eligible 青 family, the 青 pattern improved retrieval in **112 of 160 tests, or 70%**.

This supports a transferable contribution from 青. It does not imply that 青 supplies one fixed meaning to every family member.

Important data note: 清, although central to the explanatory example, was omitted from the current edge dataset because the source decomposition did not explicitly label 青 as phonetic in that entry. 清 must not be presented as a measured Phase IV success. It is a priority case in the expanded edge audit.

### 5. The primary results survive sensitivity testing

The uncentered embedding analysis remains primary. Mean-centering was run as a predeclared sensitivity analysis.

- Semantic-part improvement remained strong in all model and representation combinations.
- Phonetic-part majority improvement increased from 15 of 16 raw cells to 16 of 16 centered cells.
- The conclusion that semantic-part transfer is stronger than phonetic-part transfer remained intact.
- A raw written-character difference between Qwen and BGE/GTE disappeared after centering. That model split is therefore attributed cautiously to embedding geometry, not to an intrinsic architecture difference.

### 6. The analysis is exactly reproducible from frozen caches

A clean execution from 16 integrity-verified embedding caches reproduced all 11 compared primary output files exactly:

- identical row counts;
- zero differing cells;
- zero numerical error;
- identical SHA-256 hashes.

This establishes deterministic reproduction from the frozen caches. It does not guarantee bit-for-bit regeneration of embeddings under future model, library, GPU, or driver versions.

## What the findings support

The strongest defensible conclusion is:

> Conventionally classified phonetic components can carry transferable semantic information across distinct character contexts. This contribution is heterogeneous and weaker or less robust than the contribution of semantic components.

The findings do not establish that:

- every phonetic component carries meaning;
- a phonetic component contributes the same meaning to every derived character;
- the two components contribute equally;
- modern embedding relations prove historical causation;
- the current relational edge inventory is exhaustive.

## Current limitation and next phase

The completed Phase IV computations are valid and reproducible for the tested edges, but the frozen relational dataset is incomplete.

Most edges were derived from CJKVI decomposition records explicitly marked with 聲. This high-precision rule missed usable decompositions without that explicit annotation, including 清. A first-pass completeness audit found:

| Audit population | Count |
|---|---:|
| Characters in the master inventory | 11,151 |
| Frozen direct edges | 6,195 |
| Usable decompositions without an assigned phonetic role | 4,956 |
| Preliminary missing-edge candidates | 1,324 |
| Exact modern-pronunciation candidates | 514 |
| Active-use priority candidates | 61 |

These are review candidates, not automatically confirmed edges. The next paper-oriented release will independently validate them, publish a corrected edge dataset, and rerun the affected and full analyses from the existing embedding caches.

## Repository guide

The release is organized around frozen inputs, executable scripts, primary results, sensitivity analyses, quality control, figures, and publication-facing summaries.

```text
.
├── README.md
├── MANIFEST.json
├── environment.txt
├── inputs/                 # Frozen character, definition, and edge tables
├── scripts/                # Data preparation and analysis code
├── results_primary/        # Uncentered primary analyses
├── results_sensitivity/    # Centered and threshold sensitivities
├── qc/                     # Integrity, leakage, and reproduction checks
├── figures/                # Paper and public-facing figures
├── tables/                 # Compact reported results
└── SHA256SUMS.txt           # Release checksums
```

Files should be interpreted together with `MANIFEST.json`, which records model versions, channels, eligibility rules, random seeds, candidate universes, pooling methods, thresholds, and input and script hashes.

## Reproducing the analysis

The frozen release is designed for a clean Python environment or Google Colab session.

1. Install the versions recorded in `environment.txt`.
2. Verify release files against `SHA256SUMS.txt`.
3. Confirm all 16 embedding-cache hashes and integrity checks.
4. Run the primary semantic-part and phonetic-part scripts without editing cells during execution.
5. Run the centered sensitivity analysis separately.
6. Compare regenerated outputs with the expected hashes in the clean-reproduction tables.

The embedding caches are part of the reproducibility boundary. If they cannot be distributed because of repository size or model licensing constraints, the release manifest should provide their hashes and document how to regenerate them.

## Data sources

The project draws on:

- Taiwan Ministry of Education Traditional Chinese character inventories and dictionary definitions;
- Unicode Unihan English definitions;
- CJKVI character decomposition and component annotations;
- Google Cloud Translation for the English translation channel.

Exact source files, versions, URLs, licenses, transformations, and hashes belong in the release manifest and data documentation. Users of the dataset should review the upstream terms before redistribution.

## Project status

- Pilot and multi-family semantic-coherence studies: complete.
- Four-model embedding-space diagnostics: complete.
- Phase IV 4 × 4 relational-offset analysis: complete.
- Centered sensitivity analysis: complete.
- Clean reproduction from frozen embedding caches: passed.
- Relational-edge completeness audit and v3 expansion: in progress for the paper-oriented release.
- Public article, interactive Semantic Atlas, and manuscript: in preparation.

## About the project

This is an independent research project combining Chinese paleography, linguistics, statistics, and multilingual language models. It grew from years of close study and public explanation of Chinese character families, including more than 1,700 educational videos for English-speaking audiences.

AI was used as a research instrument for data curation, embedding generation, computational testing, code development, diagnostics, and reproducibility checks. The linguistic hypothesis, dataset decisions, interpretation, and claims remain human-directed.

## Citation and license

Citation metadata and repository licenses will be added with the frozen public release. Code and data may require separate licenses because several source datasets retain their own terms.

Until those files are present, please cite the repository URL and release tag, and do not assume that every included data file is licensed for unrestricted redistribution.

