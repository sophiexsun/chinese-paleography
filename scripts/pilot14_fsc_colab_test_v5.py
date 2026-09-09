"""Compute Pilot14 family semantic coherence (FSC) under five model conditions.

Designed for Google Colab. Install pinned dependencies first:
    %pip install -q "transformers==4.57.1" "sentence-transformers==5.1.2" accelerate bitsandbytes einops pandas openpyxl

Expected columns (legacy aliases are also accepted):
    blinded_character, EN_gloss, CN_definition, EN_definition

The five output conditions are Qwen3-Embedding-0.6B at 1024d,
8-bit Qwen3-Embedding-4B at 2560d and 1024d, BGE-M3 at 1024d, and
GTE-multilingual-base at 768d. Qwen-4B is encoded once; its first 1024
Matryoshka dimensions are sliced and separately L2-normalized.

Recommended two-stage run (reuse the same output directory):
    !python pilot14_fsc_colab_test_v5.py \
        --input /content/pilot14_blinded_embed_set_v4.csv \
        --output /content/pilot14_fsc_results_v5 \
        --models qwen3_0.6b bge_m3 gte_multilingual_base
    !python pilot14_fsc_colab_test_v5.py \
        --input /content/pilot14_blinded_embed_set_v4.csv \
        --output /content/pilot14_fsc_results_v5 \
        --models qwen3_4b --batch-size 1
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


QWEN_PREFIX = (
    "Represent this dictionary definition for cross-lingual semantic similarity: "
)

MODEL_FAMILIES = {
    "qwen3_0.6b": {
        "hf_id": "Qwen/Qwen3-Embedding-0.6B",
        "trust_remote_code": False,
        "prefix": QWEN_PREFIX,
        "native_dim": 1024,
        "output_dims": [1024],
    },
    "bge_m3": {
        "hf_id": "BAAI/bge-m3",
        "trust_remote_code": False,
        "prefix": "",
        "native_dim": 1024,
        "output_dims": [1024],
    },
    "gte_multilingual_base": {
        "hf_id": "Alibaba-NLP/gte-multilingual-base",
        "trust_remote_code": True,
        "prefix": "",
        "native_dim": 768,
        "output_dims": [768],
    },
    # Run the largest checkpoint last by default.
    "qwen3_4b": {
        "hf_id": "Qwen/Qwen3-Embedding-4B",
        "trust_remote_code": False,
        "prefix": QWEN_PREFIX,
        "native_dim": 2560,
        "output_dims": [2560, 1024],
        "int8_on_cuda": True,
    },
}

COLUMN_ALIASES = {
    "blinded_character": ["blinded_character"],
    "en_gloss": ["EN_gloss", "en_gloss", "definition_en"],
    "cn_definition": ["CN_definition", "cn_definition", "definition_cn"],
    "en_definition": [
        "EN_definition",
        "en_definition",
        "definition_en_from_cn_definition",
    ],
}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Input must be CSV, XLSX, or XLS.")


def load_pilot(path: Path) -> pd.DataFrame:
    raw = read_table(path)
    selected = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        matches = [name for name in aliases if name in raw.columns]
        if not matches:
            raise ValueError(
                f"Missing {canonical!r}. Accepted column names: {aliases}"
            )
        selected[canonical] = raw[matches[0]]
    df = pd.DataFrame(selected)
    if df.isna().any().any():
        raise ValueError("Blinded identifiers and all three definition fields are required.")
    for column in df.columns:
        df[column] = df[column].astype(str).str.strip()
        if (df[column] == "").any():
            raise ValueError(f"Column {column!r} contains blank values.")
    if df["blinded_character"].duplicated().any():
        raise ValueError("'blinded_character' identifiers must be unique.")
    if len(df) < 2:
        raise ValueError("At least two Pilot items are required.")
    return df.reset_index(drop=True)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("Encountered a zero-length embedding.")
    return (vectors / norms).astype(np.float32, copy=False)


class QuantizedQwenEncoder:
    """Minimal official last-token embedding path with 8-bit weight loading."""

    def __init__(self, hf_id: str) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-4B int8 loading requires a CUDA GPU.")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, padding_side="left")
        quantization = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModel.from_pretrained(
            hf_id,
            quantization_config=quantization,
            device_map={"": 0},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 1,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        vectors = []
        starts = range(0, len(texts), batch_size)
        if show_progress_bar:
            from tqdm.auto import tqdm
            starts = tqdm(starts, total=(len(texts) + batch_size - 1) // batch_size)
        with torch.inference_mode():
            for start in starts:
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt",
                ).to("cuda")
                hidden = self.model(**encoded).last_hidden_state
                # Left padding makes the final position the last valid token.
                pooled = hidden[:, -1]
                if normalize_embeddings:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.float().cpu().numpy())
        result = np.concatenate(vectors, axis=0)
        return result if convert_to_numpy else torch.from_numpy(result)


def encode_unique_texts(
    model: SentenceTransformer, df: pd.DataFrame, prefix: str, batch_size: int
) -> tuple[dict[str, np.ndarray], int]:
    text_groups = {
        "en_gloss": [prefix + text for text in df["en_gloss"]],
        "en_definition": [prefix + text for text in df["en_definition"]],
        "cn_definition": [prefix + text for text in df["cn_definition"]],
    }
    combined = [text for group in text_groups.values() for text in group]
    codes, unique_texts = pd.factorize(pd.Index(combined), sort=False)
    unique_vectors = model.encode(
        unique_texts.tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    restored = unique_vectors[codes]
    n = len(df)
    return (
        {
            name: restored[i * n : (i + 1) * n]
            for i, name in enumerate(text_groups)
        },
        len(unique_texts),
    )


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def cross_excluding_aligned_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Cross-representation matrices must be square.")
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def summarize_values(values: np.ndarray, metric: str) -> dict:
    return {
        f"fsc_{metric}": float(values.mean()),
        f"{metric}_n_pairs": int(values.size),
        f"{metric}_pair_cosine_median": float(np.median(values)),
        f"{metric}_pair_cosine_sd": float(values.std(ddof=1)),
        f"{metric}_pair_cosine_min": float(values.min()),
        f"{metric}_pair_cosine_max": float(values.max()),
    }


def describe_alignment(values: np.ndarray, metric: str) -> dict:
    return {
        f"{metric}_n_pairs": int(values.size),
        f"{metric}_mean": float(values.mean()),
        f"{metric}_median": float(np.median(values)),
        f"{metric}_sd": float(values.std(ddof=1)),
        f"{metric}_min": float(values.min()),
        f"{metric}_max": float(values.max()),
    }


def build_matrices(vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    eg, ed, cn = (
        vectors["en_gloss"],
        vectors["en_definition"],
        vectors["cn_definition"],
    )
    return {
        "en_gloss_en_gloss": eg @ eg.T,
        "en_definition_en_definition": ed @ ed.T,
        "cn_definition_cn_definition": cn @ cn.T,
        "en_gloss_cn_definition": eg @ cn.T,
        "en_definition_cn_definition": ed @ cn.T,
        "en_gloss_en_definition": eg @ ed.T,
    }


def make_summary(model_info: dict, matrices: dict[str, np.ndarray], n: int) -> dict:
    summary = {**model_info, "n_items": n}
    for name in (
        "en_gloss_en_gloss",
        "en_definition_en_definition",
        "cn_definition_cn_definition",
    ):
        summary.update(summarize_values(off_diagonal_values(matrices[name]), name))
    for name in (
        "en_gloss_cn_definition",
        "en_definition_cn_definition",
        "en_gloss_en_definition",
    ):
        summary.update(
            summarize_values(
                cross_excluding_aligned_values(matrices[name]),
                f"{name}_excluding_aligned",
            )
        )
        summary.update(
            describe_alignment(np.diag(matrices[name]), f"alignment_{name}_diagonal")
        )
    return summary


def save_matrix(matrix: np.ndarray, ids: list[str], path: Path) -> None:
    pd.DataFrame(matrix, index=ids, columns=ids).rename_axis("row_id").to_csv(path)


def make_pair_table(ids: list[str], matrices: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for i, left_id in enumerate(ids):
        for j, right_id in enumerate(ids):
            row = {"left_id": left_id, "right_id": right_id, "same_item": i == j}
            row.update({f"{name}_cosine": matrix[i, j] for name, matrix in matrices.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def condition_key(family_key: str, dimension: int) -> str:
    return f"qwen3_4b_{dimension}d" if family_key == "qwen3_4b" else family_key


def run_family(
    family_key: str,
    spec: dict,
    df: pd.DataFrame,
    output_root: Path,
    batch_size: int,
) -> tuple[list[dict], list[pd.DataFrame], dict[str, int]]:
    print(f"\nLoading {spec['hf_id']} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_kwargs = {}
    if device == "cuda" and spec.get("int8_on_cuda", False):
        model = QuantizedQwenEncoder(spec["hf_id"])
    else:
        model = SentenceTransformer(
            spec["hf_id"],
            trust_remote_code=spec["trust_remote_code"],
            device=device,
            **load_kwargs,
        )
    full_vectors, unique_count = encode_unique_texts(
        model, df, spec["prefix"], batch_size
    )
    actual_dim = next(iter(full_vectors.values())).shape[1]
    if actual_dim != spec["native_dim"]:
        raise RuntimeError(
            f"Unexpected native dimension for {family_key}: {actual_dim} "
            f"(expected {spec['native_dim']})."
        )

    summaries, pair_tables, counts = [], [], {}
    ids = df["blinded_character"].tolist()
    for dimension in spec["output_dims"]:
        vectors = {
            name: l2_normalize(values[:, :dimension])
            for name, values in full_vectors.items()
        }
        key = condition_key(family_key, dimension)
        matrices = build_matrices(vectors)
        model_dir = output_root / key
        model_dir.mkdir(parents=True, exist_ok=True)
        for name, matrix in matrices.items():
            save_matrix(matrix, ids, model_dir / f"cosine_{name}.csv")
        pairs = make_pair_table(ids, matrices).assign(
            model=key,
            model_family=family_key,
            hf_id=spec["hf_id"],
            embedding_dimension=dimension,
            weight_precision="int8" if spec.get("int8_on_cuda", False) else "native",
        )
        pairs.to_csv(model_dir / "all_pair_scores.csv", index=False)
        info = {
            "model": key,
            "model_family": family_key,
            "hf_id": spec["hf_id"],
            "embedding_dimension": dimension,
            "weight_precision": "int8" if spec.get("int8_on_cuda", False) else "native",
        }
        summaries.append(make_summary(info, matrices, len(df)))
        pair_tables.append(pairs)
        counts[key] = unique_count

    del model, full_vectors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summaries, pair_tables, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default=Path("pilot14_fsc_results_v5"), type=Path)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_FAMILIES),
        default=list(MODEL_FAMILIES),
        help="qwen3_4b automatically produces both 2560d and 1024d conditions.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    df = load_pilot(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "validated_pilot14_blinded_input.csv", index=False)

    summary_path = args.output / "pilot14_fsc_summary_by_model.csv"
    pairs_path = args.output / "all_models_pair_scores.csv"
    summaries = (
        pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []
    )
    pair_tables = [pd.read_csv(pairs_path)] if pairs_path.exists() else []
    completed = {row["model"] for row in summaries if "model" in row}
    unique_counts = {}
    for family_key in args.models:
        expected = {
            condition_key(family_key, dimension)
            for dimension in MODEL_FAMILIES[family_key]["output_dims"]
        }
        if expected.issubset(completed):
            print(f"Skipping completed family {family_key}: {sorted(expected)}")
            continue

        # Replace any incomplete prior rows for this family without duplicating
        # data when the same output directory is resumed.
        summaries = [row for row in summaries if row.get("model") not in expected]
        if pair_tables:
            existing = pd.concat(pair_tables, ignore_index=True)
            pair_tables = [existing[~existing["model"].isin(expected)]]

        family_summaries, family_pairs, family_counts = run_family(
            family_key, MODEL_FAMILIES[family_key], df, args.output, args.batch_size
        )
        summaries.extend(family_summaries)
        pair_tables.extend(family_pairs)
        unique_counts.update(family_counts)
        completed.update(expected)

        # Checkpoints retain completed families if a later Colab model fails.
        pd.DataFrame(summaries).to_csv(
            summary_path, index=False
        )
        pd.concat(pair_tables, ignore_index=True).to_csv(
            pairs_path, index=False
        )

    summary_df = pd.DataFrame(summaries)
    metadata = {
        "input_file": str(args.input),
        "n_items": int(len(df)),
        "models": {key: MODEL_FAMILIES[key] for key in args.models},
        "distinct_prefixed_texts_encoded_by_condition": unique_counts,
        "qwen_4b_1024_method": "first 1024 Matryoshka dimensions of the cached 2560d vectors, followed by separate L2 normalization",
        "fsc_definition": {
            "within_representation": "mean cosine over n(n-1)/2 unique item pairs",
            "cross_representation": "mean cosine over n(n-1) pairs excluding aligned i=j definitions",
            "alignment_QC": "diagonal summaries compare representations of the same item and are not FSC",
        },
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    zip_path = Path(
        shutil.make_archive(str(args.output.resolve()), "zip", root_dir=args.output.resolve())
    )
    print("\nPilot14 FSC summary")
    print(summary_df.to_string(index=False))
    print(f"\nSaved results to: {args.output.resolve()}")
    print(f"Created one-file download: {zip_path}")


if __name__ == "__main__":
    main()
