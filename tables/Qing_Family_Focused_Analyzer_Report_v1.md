# 青 Family: Focused Phase IV Result

## Terminology

The visuals use **semantic part** and **phonetic part**. These are the conventional labels for readability; the experiment explicitly tests whether the conventionally labeled phonetic part also carries transferable meaning-related information.

## 清 and 晴 status

- **晴 = 日 + 青 was tested in both directions.**
- **清 = 氵 + 青 was not tested.** The frozen Phase IV relational results contain no 青→清 edge. Occurrences of 清 elsewhere in output are predictions for unrelated cases, not a 清 test. This is a dataset-coverage gap that should be corrected and rerun before using 清 as a numerical example.

## 晴 result

The semantic-part test asks whether a pattern learned for 日 across other phonetic families helps recover 晴 from 青. The phonetic-part test asks whether a pattern learned for 青 across its other semantic contexts helps recover 晴 from the independent semantic component 日.

| Meaning representation | Semantic-part rank gain | Phonetic-part rank gain | Semantic models improved | Phonetic models improved |
|---|---:|---:|---:|---:|
| Chinese definitions | 677 (11.2%) | 64 (1.1%) | 4/4 | 4/4 |
| English translations | 151 (2.5%) | 272 (4.5%) | 4/4 | 4/4 |
| English glosses | 2111 (23.1%) | 240 (2.6%) | 4/4 | 3/4 |

Rank gain is reported as average candidate positions gained; the parenthetical value is the share of the full candidate list. Across the three meaning-based representations, the semantic-part pattern improved all **12/12** model-channel tests. The 青 phonetic-part pattern improved **11/12**.

This is strong case-level evidence that 青 contributed transferable information to 晴 in the embedding tests. It does not establish one fixed lexical meaning for 青, and it does not prove the historical derivation of 晴.

## 青-family phonetic-part result

Across all eligible 青-family cases in the three meaning-based representations, the phonetic-part pattern improved **112 of 160 tests (70.0%)**. The result is heterogeneous across individual characters:

| Character | Semantic component | Improved tests | Improvement rate | Median share of candidate list gained |
|---|---|---:|---:|---:|
| 婧 | 女 | 4/4 | 100.0% | 22.19% |
| 情 | 忄 | 12/12 | 100.0% | 7.32% |
| 靚 | 見 | 11/12 | 91.7% | 12.71% |
| 晴 | 日 | 11/12 | 91.7% | 0.56% |
| 倩 | 亻 | 11/12 | 91.7% | 5.59% |
| 靖 | 立 | 9/12 | 75.0% | 6.11% |
| 精 | 米 | 9/12 | 75.0% | 6.71% |
| 鯖 | 魚 | 9/12 | 75.0% | 0.71% |
| 猜 | 犭 | 8/12 | 66.7% | 2.33% |
| 靜 | 爭 | 7/12 | 58.3% | 2.91% |
| 錆 | 釒 | 2/4 | 50.0% | 0.34% |
| 睛 | 目 | 6/12 | 50.0% | 0.01% |
| 請 | 言 | 6/12 | 50.0% | 0.04% |
| 蜻 | 虫 | 5/12 | 41.7% | -2.62% |
| 綪 | 糸 | 1/4 | 25.0% | -1.09% |
| 菁 | 艹 | 1/4 | 25.0% | -2.76% |

## Interpretation boundary

The clean claim is: **within the tested 青 family, a pattern learned from 青 across other character contexts usually helped retrieve the correct derived character, including 晴.** The missing 清 edge prevents extending that numerical statement to 清 until a targeted correction-and-rerun is completed.
