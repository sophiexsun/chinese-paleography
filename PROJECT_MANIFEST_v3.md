# Project Manifest: Computational Re-evaluation of 形声字
## Paper 1 — Quantifying Semantic Inheritance in Conventional “Phonetic” Components

## 0. Research Position and Scope

This project tests, rather than presupposes, the proposition that a substantial share of Chinese characters conventionally classified as 形声字 (phono-semantic compounds) preserve recoverable semantic information in the component conventionally labeled 声符 (“phonetic component”).

The historical point of departure is 右文说. Later scholarship framed a closely related mechanism as 声符示源: the so-called phonetic component can preserve or indicate a lexical/etymological source. For this paper, these are treated as historically related formulations addressing the same broad empirical phenomenon, while differences in individual scholars’ claims will be documented in the literature review rather than assumed away.

The paper does **not** begin by trying to prove that most Chinese characters are semantic. It asks how much of the conventional 形声 inventory is better explained by **semantic inheritance** than by **phonetic appropriation**, and how much remains indeterminate.

The larger long-term program is a data-driven reconsideration of 六书 and of Chinese writing as a semiotic system. That meta-level argument is outside Paper 1 except as motivation and discussion.

---

## 1. Working Terminology

These terms are operational definitions for the project and can be refined during the literature review.

- **字 / character:** a unit of the writing system, e.g. 德, 波, 人.
- **词 / word:** a linguistic lexical unit; it may contain one or multiple characters.
- **语素 / morpheme:** the smallest linguistic unit carrying lexical or grammatical meaning. A Chinese character often writes a morpheme, but character and morpheme are not identical concepts.
- **Semantic / 语义:** pertaining to meaning in general.
- **Lexical-semantic / 词汇语义:** pertaining specifically to the meanings and meaning-relations of words or morphemes.
- **Synchronic / 共时:** analysis of a system at a particular period or state.
- **Diachronic / 历时:** analysis of historical development through time.
- **Semantic inheritance:** a character/component relationship in which historical evidence supports continuity, derivation, specialization, metaphorical extension, cognation, or another non-accidental semantic pathway from an earlier conceptual/lexical source.
- **Phonetic appropriation:** use or reuse of an existing graphic form primarily because of sound, without sufficient evidence that its original semantic content motivates the target meaning. 假借 is an important historical case to examine here.
- **Indeterminate:** available paleographic, phonological, textual, or semantic evidence is insufficient or conflicting; no forced classification is made.
- **Polysemy / 多义:** multiple related senses of one lexical item. It is one possible mechanism of semantic inheritance, not a synonym for all semantic inheritance.
- **Phonophore / 声符:** the component conventionally analyzed as supplying phonetic information in a 形声字. The term describes its conventional analytical role and does not prejudge whether it also preserves semantic/source information.

---

## 2. Central Research Question

Among Chinese characters conventionally classified as 形声字, **how much semantic information is carried by the component conventionally labeled 声符 after controlling for the semantic classifier, phonological similarity, graphic structure, chronology, and chance?**

A stronger historical question follows:

> When such semantic information is detected, how often is it supported by diachronic evidence as genuine semantic/etymological inheritance rather than synchronic association, coincidence, analogy, or sound-based appropriation?

The intended empirical output is not a predetermined binary answer but an estimated distribution:

1. **Semantic inheritance**
2. **Phonetic appropriation**
3. **Indeterminate / disputed**

---

## 3. Core Hypotheses

### H1 — Semantic Information in Conventional Phonophores

For a substantial proportion of characters conventionally classified as 形声, the conventional 声符 contributes statistically detectable information about meaning beyond that supplied by the conventional semantic classifier.

A useful information-theoretic formulation is:

$$I(M;P\mid S) > 0$$

where:

- $M$ = character/morpheme meaning,
- $P$ = conventional phonophore,
- $S$ = conventional semantic classifier,
- $I$ = conditional mutual information.

This tests the phenomenon associated historically with 右文说 without assuming the result.

### H2 — Historical Semantic Inheritance

For a subset of families showing H1 effects, the semantic signal corresponds to historically reconstructable semantic or lexical ancestry supported by early forms, early textual usage, historical phonology, and/or paleographic and etymological scholarship.

This is the stronger 声符示源/右文 mechanism: the shared component is not merely correlated with present-day meanings but preserves evidence of a historical source relationship.

### H3 — Residual Phonetic Appropriation

After semantic inheritance is accounted for, a residual set of characters will be better explained primarily by phonetic appropriation: historical phonological similarity is present while a defensible semantic inheritance pathway is absent.

### H4 — Indeterminacy Is a Real Category

Some characters cannot be responsibly assigned to either mechanism because early forms are missing, decompositions are disputed, historical meanings are uncertain, or multiple explanations remain viable. The research will quantify rather than suppress this uncertainty.

---

## 4. Longer-Term Classification Proposal — Not a Presupposition of Paper 1

The broader research program asks whether 六书 can eventually be reframed using a smaller set of generative relationships between graphic form and lexical meaning:

1. **Direct semantic representation (logographic/iconic):** a graphic form represents a concept through recognizable visual motivation.
2. **Semantic composition/inheritance:** multiple graphic components, potentially nested hierarchically, participate in constructing or preserving a conceptual/lexical pathway.
3. **Phonetic appropriation:** a graphic form is recruited primarily through sound despite semantic discontinuity.

Paper 1 does **not** assume that these three classes are sufficient. It tests the disputed 形声 inventory to determine whether the evidence warrants such a later reclassification.

---

## 5. Semiotic Framing — Background, Not the Primary Test

The project is relevant to 符号学 (semiotics), especially the relationship among iconic, indexical, and symbolic sign functions.

A preliminary analogy is useful but must not be treated as a one-to-one equivalence with 六书:

- **Iconic:** a sign bears some resemblance to what it represents; this is strongly relevant to 象形.
- **Indexical:** a sign points to a concept through a motivated relation; some 指事 constructions such as 本 and 末 provide intuitive examples of graphically indicated relations.
- **Symbolic/conventional:** the sign-meaning relation depends more heavily on learned convention; phonetic appropriation is a strong candidate, but conventionality also exists throughout writing, including historically motivated characters.

Therefore, “symbolic” must not become a residual synonym for “unknown.” An indeterminate character may simply lack surviving evidence of its earlier motivation.

A future meta-level project may ask how Chinese orthography distributes information across visual, semantic, morphological, and phonological channels relative to other writing systems. That question is deliberately outside the main inferential burden of Paper 1.

---

## 6. Character Representation: Hierarchical, Not Flat

Characters must not be reduced to left/right radical pairs. The structural layer should preserve recursive/nested composition.

A character is represented as a tree or graph:

```text
Character
├── Component A
│   ├── Subcomponent A1
│   └── Subcomponent A2
└── Component B
    ├── Subcomponent B1
    └── Subcomponent B2
```

This permits hypotheses in which semantic information exists at multiple levels of composition. A modern decomposition must, however, be distinguished from a historically demonstrated decomposition.

For each proposed component relation, store:

- synchronic graphic decomposition,
- earliest attested form,
- proposed diachronic derivation,
- source/citation,
- scholarly disagreement,
- confidence level.

---

## 7. Evidence Architecture

Each character/family should combine four evidence layers.

### 7.1 Visual/Paleographic Layer
- Oracle-bone forms where attested
- Bronze forms
- Warring States / Seal forms where available
- Clerical/Regular forms as needed
- Modern Traditional and Simplified forms
- Provenance and dating of forms

### 7.2 Structural Layer
- Recursive IDS/component tree
- Conventional semantic classifier
- Conventional phonophore
- Alternative historical decompositions
- Component substitutions and graphic transformations

### 7.3 Semantic/Etymological Layer
- Earliest attested textual senses where available
- 《说文解字》 analysis as Han-period scholarly evidence, not unquestioned ground truth
- 段玉裁 and other historical commentaries
- Modern paleographic/etymological scholarship
- Modern senses and documented semantic extensions

Represent semantic claims as provenance-tagged assertions:

$$A=(character, sense, source, date, evidence\ type, confidence)$$

### 7.4 Phonological Layer
- Reconstructed Old Chinese readings from explicitly named reconstruction systems
- Middle Chinese where relevant
- Modern Mandarin
- Selected Sinitic varieties where useful

Phonology is required both as evidence and as a control variable. Similar sound cannot itself be taken as proof of phonetic appropriation, nor can semantic similarity itself prove common historical origin.

---

## 8. Dataset Strategy

### Stage A — Gold-Standard Toy Dataset

Construct a manually reviewed pilot dataset of approximately **50–100 phonetic families**, intentionally sampling:

- strong traditional 右文/声符示源 examples,
- apparently strong phonetic-appropriation cases,
- disputed families,
- families with major graphic transformation,
- families with good early attestations,
- families with sparse evidence.

The pilot is for ontology design, annotation rules, statistical proof-of-concept, and failure discovery — not for estimating the final population percentage.

### Stage B — Expanded Digitized Inventory

After the pilot methodology stabilizes, scale to the largest defensible intersection of:

- digitized Chinese character inventories,
- structural decomposition data,
- historical forms,
- semantic/etymological sources,
- historical phonological reconstruction.

Do not equate “all Unicode CJK characters” with “all analyzable Chinese characters.” The inferential population must be explicitly defined by inclusion criteria and evidence availability.

---

## 9. Quantitative Framework

### 9.1 Family Semantic Coherence

For a conventional phonophore family $F_P=\{C_1,...,C_n\}$:

$$FSC(P)=\frac{2}{n(n-1)}\sum_{i<j}sim(S_i,S_j)$$

where $S_i$ represents the semantic description of family member $i$.

### 9.2 Conditional Semantic Contribution

Estimate whether knowledge of the conventional phonophore improves prediction of meaning after accounting for the conventional semantic classifier:

$$I(M;P\mid S)$$

Alternative predictive implementations can compare models with and without phonophore information and measure out-of-sample improvement.

### 9.3 Matched Null Models

Target families must be compared against increasingly strict controls, for example:

1. matched graphic layout,
2. matched semantic classifier,
3. matched historical period/frequency,
4. matched historical phonological neighborhood,
5. combinations of the above.

A key quantity is:

$$\Delta = Coherence_{same\ phonophore}-Coherence_{phonologically\ matched\ control}$$

This helps distinguish a graphic-family semantic effect from the broader historical relationship between sound and meaning.

### 9.4 Historical Evidence Score

Semantic clustering is discovery evidence, not automatic proof of etymology. Families showing significant clustering receive a separate historical evaluation using paleographic forms, attested senses, chronology, and phonological reconstruction.

The final estimates should distinguish:

$$R_s=\frac{characters\ with\ detectable\ phonophore\ semantic\ signal}{analyzable\ conventional\ 形声\ characters}$$

from the stronger:

$$R_h=\frac{characters\ with\ historically\ supported\ semantic\ inheritance}{analyzable\ conventional\ 形声\ characters}$$

with $R_h\leq R_s$ expected by definition.

---

## 10. Bias and Falsifiability Protocol

The researcher has substantial prior exposure to Chinese etymology and a strong working intuition that conventional 形声 analysis often under-describes semantic structure. This expertise is valuable for generating hypotheses but creates a risk of motivated decomposition or post-hoc semantic explanation.

Mitigations:

- define annotation rules before full-scale labeling,
- record competing explanations rather than only preferred explanations,
- distinguish observed evidence from researcher interpretation,
- use explicit confidence levels,
- retain an indeterminate class,
- blind or independent annotation of a sample where feasible,
- perform inter-annotator agreement analysis where feasible,
- preregister primary hypotheses/statistical tests before the full dataset is analyzed,
- report negative cases and families that contradict the theory.

The study is successful if it produces a credible estimate, even if the estimated semantic-inheritance proportion is much smaller than expected.

---

## 11. Paper 1 Workflow

### Phase 1 — Literature Review and Theory
- History and formulations of 右文说
- 声符示源 literature
- Modern theories of 形声字 and 六书
- Historical Chinese morphology/etymology relevant to phonetic series
- Computational studies of Chinese character components and semantic transparency
- Define terminology and competing causal explanations

**Output:** literature matrix + final operational definitions + research questions.

### Phase 2 — Pilot Annotation Ontology and Gold Dataset
- Select 50–100 representative phonetic families
- Build recursive component structures
- Attach paleographic, semantic, phonological, and bibliographic evidence
- Create classification and confidence rubric
- Test annotation consistency

**Output:** versioned pilot dataset + annotation manual.

### Phase 3 — Algorithmic Proof of Concept
- Establish semantic representations
- Compute family coherence
- Build matched controls/permutation tests
- Test conditional semantic contribution
- Run sensitivity/ablation analyses

**Output:** reproducible analysis notebook/pipeline + pilot results.

### Phase 4 — Scale Dataset
- Automate source parsing where legally/technically possible
- Cross-reference character identifiers and variants
- Expand to the defined analyzable inventory
- Audit uncertain mappings and sampling bias

**Output:** research dataset with provenance and confidence fields.

### Phase 5 — Confirmatory Analysis
- Freeze/preregister primary tests
- Run full statistical analysis
- Estimate semantic-inheritance, phonetic-appropriation, and indeterminate proportions
- Conduct robustness tests and error analysis

**Output:** final tables, figures, statistical results, and reproducibility package.

### Phase 6 — Paper
- Introduction and research question
- Intellectual history/literature review
- Data and annotation methodology
- Statistical methodology
- Results
- Case studies explaining successes/failures
- Discussion: implications for 形声, 右文说, and possible reconsideration of 六书
- Limitations and future semiotic research

---

## 12. Explicitly Deferred to Future Research

The following are valuable but should not be required for Paper 1:

- a Transformer that learns cross-era graphic transformation grammar,
- a universal model of radial polysemy,
- full reclassification of 六书,
- quantitative comparison of Chinese with alphabetic writing systems,
- a general semiotic theory of Chinese writing,
- human ontology and cross-linguistic conceptual categorization.

These can become follow-on papers once the empirical foundation is established.

---

## 13. Success Criterion

The goal is not to prove a predetermined “semantic-majority” conclusion. The goal is to produce the first defensible estimate this project can achieve of how much of the conventional 形声 inventory shows:

$$\boxed{Semantic\ inheritance\;|\;Phonetic\ appropriation\;|\;Indeterminate}$$

and to determine whether the conventional 声符 carries statistically significant semantic information after serious historical and phonological controls.

If supported, the result supplies quantitative evidence for reconsidering 形声 as a heterogeneous category and provides an empirical foundation for a later, broader re-evaluation of 六书 and Chinese writing within semiotics.

---

## 14. August 2026 Methodological Revision: Traditional-First Component-Family Test

This section supersedes conflicting details in Sections 7–11 for Paper 1. It records the methodological decisions reached after stress-testing the quantitative framework.

### 14.1 Paper-1 Starting Universe: Traditional Chinese, Not Pre-Labeled 形聲字

Paper 1 begins with a defensible inventory of **living Traditional Chinese characters with machine-readable recursive component structures**, rather than selecting characters because a traditional source already labels them 形聲.

Let:

$$U=\{C: C\text{ is in the defined Traditional inventory and has an analyzable component tree}\}$$

For every recurring component $P$, construct the empirical component family:

$$F(P)=\{C\in U:P\text{ occurs anywhere in the recursive decomposition tree of }C\}$$

This is deliberately broader than a right-side 声符 lookup. A component may occur on the left, right, top, bottom, inside another component, or at another nested level. Conventional labels such as 部首, 意符, 声符, 形聲, 會意, etc. are joined **after** structural family construction so that 六書 classifications remain variables to be evaluated rather than inclusion rules that predetermine the dataset.

The first technical feasibility test is therefore:

> Can the pipeline recover every analyzable Traditional character containing a test component such as 青, together with its recursive structure, dictionary radical (部首), conventional phonophore label where available, pronunciation, definitions, and source provenance?

### 14.2 部首 Is an Indexing Field, Not the Structural Ontology

部首 is retained because it is useful dictionary metadata and because conventional 形聲 analysis frequently contrasts a semantic classifier with a 声符. It must not be confused with the full component structure of a character.

Store separately:

- dictionary 部首,
- complete recursive IDS/component tree,
- conventional 意符 assignment where available,
- conventional 声符 assignment where available,
- alternative/historical decompositions,
- confidence and source.

### 14.3 Primary Diachronic Lens: Earliest Attestation → Traditional

For Paper 1, the primary historical chain is:

$$Earliest\ attested\ form \rightarrow Oracle/Bronze\ (where\ available) \rightarrow Warring\ States/Seal \rightarrow Clerical/Regular \rightarrow Traditional$$

Not every character is attested at every stage. Missing stages remain missing rather than inferred.

**Simplified Chinese is excluded from the primary formation/decomposition endpoint.** Twentieth-century simplification can replace or merge historically informative components and therefore introduce noise when testing semantic motivation. For example, a study of 鑽 should not infer its historical formation from the modern simplified decomposition 钻 = 金 + 占.

Simplified Chinese is retained as a separate transformation layer for future research:

$$Traditional \xrightarrow{modern\ simplification} Simplified$$

A later project may quantify semantic/etymological transparency lost, preserved, or altered by simplification.

### 14.4 Diachronic Semantic Mechanisms Are Time-Indexed

Do not treat only formation-era inheritance as legitimate semantic motivation. Chinese characters and component families remain productive systems and may acquire later motivated analogies.

For each character-family relationship, classify the best-supported mechanism and, where possible, its period of emergence:

1. **Early semantic inheritance** — semantic relationship is supported at or near the earliest reconstructable formation stage.
2. **Later semantic analogy/productivity** — a later character or sense extends an established graphic-semantic family pattern. This is a legitimate adaptation, not statistical noise.
3. **Phonetic appropriation** — sound-based recruitment is the best-supported explanation and no defensible semantic motivation is established for the relevant formation.
4. **Indeterminate/disputed** — surviving evidence is insufficient or conflicting.

The final paper should estimate a distribution of mechanisms rather than force a synchronic-vs-historical binary.

---

## 15. Revised Semantic Measurement Engine

### 15.1 What Is Embedded

Semantic similarity is not calculated from the bare character glyph or character token. Doing so risks circularity because a language model may already encode shared orthography, phonophore membership, compounds, pronunciation, or known etymological commentary.

Instead, build semantic representations from **source-grounded sense definitions with the target character identity and shared component hidden from the embedding input wherever practical**.

For character $C$ with sense definition $D_C$:

$$S_C=E(D_C)$$

where $E$ is a sentence/text embedding model producing a high-dimensional semantic vector.

Embedding dimensions may number in the hundreds or thousands depending on the chosen model. Individual dimensions need not have human-interpretable labels; semantic information is distributed across the vector.

### 15.2 Cosine Semantic Similarity

For semantic vectors $S_i$ and $S_j$:

$$sim(S_i,S_j)=\frac{S_i\cdot S_j}{\|S_i\|\|S_j\|}$$

Cosine similarity remains mathematically valid in high-dimensional spaces. It measures directional alignment rather than raw vector magnitude.

### 15.3 Multiple Senses and Historical Senses

A character must not be reduced to one modern dictionary gloss when multiple relevant senses are documented. Store sense-level records with source and date/period where possible:

$$Sense=(character,definition,source,date/period,confidence)$$

Family similarity may initially be computed using a pre-specified sense-selection rule and later tested with alternative rules. Avoid unconstrained maximum-pair similarity across many senses because it can manufacture apparent relationships through multiple testing.

Where data permits, semantic coherence can be indexed by historical stage:

$$FSC(P,t)$$

This asks how coherent component family $P$ is in semantic space using evidence appropriate to period $t$.

### 15.4 Family Semantic Coherence (FSC)

For component family $F(P)=\{C_1,...,C_n\}$:

$$FSC(P)=\frac{2}{n(n-1)}\sum_{i<j}cos(S_i,S_j)$$

FSC is a **family-level descriptive statistic**. It asks whether definitions of characters sharing component $P$ occupy an unusually coherent region of semantic space.

There is no universal fixed FSC cutoff such as 0.5 or 0.7. Interpretation comes from matched empirical null distributions, effect sizes, and pre-specified statistical thresholds.

### 15.5 Bilingual / Cross-Lingual Robustness

English and Chinese embeddings from unrelated monolingual models cannot be directly compared as though they share coordinates. A multilingual embedding model may place languages in an aligned semantic space, but cross-language alignment must be empirically validated for this task.

A useful robustness design is to create two blinded semantic channels:

- **Chinese-definition channel:** source-grounded Chinese definitions embedded without exposing the target glyph/component where feasible.
- **English-definition channel:** controlled English renderings of the same definitions embedded independently.

Agreement across channels strengthens confidence that an observed family effect is not merely an artifact of Chinese orthographic co-occurrence. Translation can itself introduce semantic convergence or divergence (for example, one English word may translate several distinct Chinese expressions), so cross-language agreement is supporting evidence rather than ground truth.

### 15.6 LLM Reasoning as Secondary Annotation, Not the Primary Metric

A generative LLM may be used as a **blinded secondary semantic annotator**. It can receive anonymized definitions and infer whether they share a minimal conceptual nucleus. It must not see the target characters, shared component, conventional family label, or researcher-proposed etymology during primary scoring.

LLM narrative inference is kept separate from embedding cosine similarity because generative models are powerful pattern/story finders and can automate the same post-hoc explanatory bias the research is designed to control.

Researcher interpretations (for example, a proposed latent conceptual root for a family) are retained as **candidate hypotheses**, not as labels used to train or score the primary semantic test.

A small manually reviewed validation sample is still required to test whether the automated semantic metric behaves sensibly, but the project does not require manual annotation of the full inventory.

---

## 16. Revised Quantitative Logic

The quantitative framework is a sequence of progressively stronger questions rather than four independent formulas.

### 16.1 A — Detect Semantic Family Structure

Compute $FSC(P)$ from blinded definition embeddings.

Question:

> Are members of a recurring component family semantically more coherent than expected?

FSC measures the phenomenon; it does not establish etymology.

### 16.2 B — Test Incremental Semantic Information

The conceptual information-theoretic hypothesis remains:

$$I(M;P\mid S)>0$$

where $M$ is meaning, $P$ is the recurring/conventional phonophore component, and $S$ is the conventional semantic classifier.

For Paper 1, the preferred operational implementation may be predictive rather than direct sparse estimation of conditional mutual information:

$$Model_A: Meaning\sim S$$

$$Model_B: Meaning\sim S+P$$

Then evaluate out-of-sample improvement:

$$\Delta_{pred}=Performance(Model_B)-Performance(Model_A)$$

Question:

> After the conventional semantic classifier is known, does the component conventionally treated as phonetic still add reproducible information about meaning?

### 16.3 C — Construct Natural, Attested Phonological Controls

A weak control that merely samples modern Mandarin homophones is insufficient, especially given the high syllabic overlap of Chinese.

The preferred control holds as much structure constant as the historical inventory permits. For target $(S,P)$, search for **actually attested characters** of the form:

$$S+P_1,\ S+P_2,\ ...$$

where each $P_k$ is phonologically close to $P$ at the historically relevant period but does not share the target component identity. Controls should preferentially match:

- the same conventional semantic classifier/component $S$,
- comparable structural position/layout,
- attested character status (never invented combinations),
- relevant historical period,
- frequency/lexical status where possible,
- historical phonological neighborhood.

Modern Mandarin exact syllable/tone matching is secondary. Formation-era hypotheses should rely primarily on explicitly named Old Chinese reconstruction systems, with Middle Chinese and living varieties used as supplementary evidence.

The inventory can be viewed as a sparse natural matrix:

$$Meaning=f(S,P,S\times P)$$

Rows correspond approximately to semantic classifiers/components and columns to recurring component/phonophore families. The matrix is sparse because most theoretical combinations are unattested. Statistical methods must respect this missingness rather than fabricate cells.

A matched effect can be expressed as:

$$\Delta=Coherence_{same\ component}-Coherence_{historically\ phonology\ matched\ controls}$$

### 16.4 D — Permutation / Null Distribution

For each target family, construct many matched pseudo-families from eligible attested controls and compare their coherence with the observed FSC:

$$p=\frac{\#(FSC_{matched\ null}\ge FSC_{observed})}{N_{permutations}}$$

The exact matching algorithm and eligibility rules must be fixed during the pilot before full-scale inference. Multiple-comparison correction is required when testing many component families.

### 16.5 E — Diachronic Provenance Classification

Statistical semantic signal is discovery evidence. It does not automatically establish historical origin. Families with detectable signal receive provenance evaluation using the evidence layers defined above.

Final mechanism estimates should be time-aware and should report at minimum:

$$P(Early\ Semantic\ Inheritance)$$

$$P(Later\ Semantic\ Analogy/Productivity)$$

$$P(Phonetic\ Appropriation)$$

$$P(Indeterminate/Disputed)$$

with confidence intervals and explicit denominator/inclusion criteria.

This replaces the earlier $R_s$ versus $R_h$ hierarchy, which incorrectly implied that later analogical semantic development was less legitimate than formation-era inheritance.

---

## 17. Immediate Handoff: Data Collection Session

The next research session should focus only on **dataset feasibility and provenance**, not on proving the semantic hypothesis.

### First test family: 青

1. Define a Traditional Chinese inventory and document its source/licensing/version.
2. Obtain recursive component/IDS data.
3. Retrieve every analyzable Traditional character containing 青 **anywhere in its recursive component tree**, not only on the right side and not only when 青 is labeled 声符.
4. For each result, attach where available:
   - Traditional character,
   - Unicode code point / stable identifier,
   - dictionary 部首,
   - recursive IDS/component tree,
   - position/nesting of 青,
   - conventional 形聲/會意/etc. classification,
   - conventional 意符 and 声符 labels,
   - Mandarin pronunciation,
   - dictionary definition(s),
   - frequency or living-use indicator,
   - source and provenance for every field.
5. Audit false decompositions, variant forms, compatibility characters, rare/obsolete forms, and disagreements among structural databases.
6. Only after the 青 extraction is validated, generalize the pipeline to every recurring component family.

### Candidate Data Architecture

```text
Character
├── Identity
│   ├── Traditional glyph
│   ├── Unicode code point
│   └── variant relationships
├── Structure
│   ├── dictionary 部首
│   ├── recursive IDS tree
│   ├── recurring component membership(s)
│   ├── conventional 意符/声符 labels
│   └── source + confidence
├── Semantics
│   └── sense records [definition, source, period, confidence]
├── Phonology
│   ├── Mandarin
│   ├── Middle Chinese (later phase)
│   └── Old Chinese reconstruction(s) (later phase)
└── Paleography (later phase)
    └── attested forms [form/stage, source, date, confidence]
```

### Data-Collection Stop Rule

Do not scale to the full inventory until the 青 pilot demonstrates that:

- component-family retrieval is reproducible,
- Traditional/Simplified and variant relationships are not being conflated,
- recursive decomposition is adequate for nested structures,
- dictionary radical and component roles remain separate fields,
- semantic definitions can be obtained with provenance,
- missingness and disagreements can be represented explicitly.

The output of the next session should be a **versioned 青-family pilot table plus a data-source audit**, not a semantic conclusion.
