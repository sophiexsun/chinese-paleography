"""Phase II-D: ten frozen component families against size-matched random nulls.

Designed for Google Colab with a Tesla T4 GPU.  The expensive work is done once:
each model loads once, each distinct semantic text is embedded once, and all test
and control FSC values are calculated by indexing persistent normalized caches.

Colab dependencies:
  %pip install -q "transformers==4.57.1" "sentence-transformers==5.1.2" \
      accelerate bitsandbytes einops pandas numpy tqdm

Example (the instruction document's frozen 1,000-control design):
  !python /content/Phase_II_D_10Family_FSC_Colab_v1.py \
    --master /content/Phase_II_C_MOE4808_Semantic_Master_v1.csv \
    --members /content/Phase_II_C_Frozen10_Test_Members_v1.csv \
    --output /content/Phase_II_D_results_v1

For the subsequently approved 500-control design, add: --n-controls 500
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


VERSION = "1.0.0"
DEFAULT_SEED = 20260812  # Phase II-B frozen selection seed; controls are separate.
QWEN_INSTRUCTION = (
    "Represent this dictionary definition for cross-lingual semantic similarity: "
)

CHANNELS = {
    "cn_moe": "embed_text_cn_moe",
    "en_google_from_moe": "embed_text_en_google_from_moe",
    "en_unihan": "embed_text_en_unihan",
    "en_google_character": "embed_text_en_google_character",
}
PRIMARY_CHANNELS = ["cn_moe", "en_google_from_moe", "en_unihan"]
EXPECTED_FAMILIES = {
    "余": 8, "元": 6, "帝": 7, "闌": 6, "辰": 11,
    "曷": 11, "賓": 9, "圭": 13, "台": 15, "亥": 12,
}

MODELS = {
    "qwen3_0.6b_1024d": {
        "hf_id": "Qwen/Qwen3-Embedding-0.6B", "dimension": 1024,
        "pooling": "SentenceTransformers model-defined last-token pooling",
        "precision": "native", "trust_remote_code": False, "qwen": True,
    },
    "bge_m3_1024d": {
        "hf_id": "BAAI/bge-m3", "dimension": 1024,
        "pooling": "SentenceTransformers model-defined pooling",
        "precision": "native", "trust_remote_code": False, "qwen": False,
    },
    "gte_multilingual_base_768d": {
        "hf_id": "Alibaba-NLP/gte-multilingual-base", "dimension": 768,
        "pooling": "SentenceTransformers model-defined pooling",
        "precision": "native", "trust_remote_code": True, "qwen": False,
    },
    "qwen3_4b_2560d_int8": {
        "hf_id": "Qwen/Qwen3-Embedding-4B", "dimension": 2560,
        "pooling": "final-layer last non-padding token",
        "precision": "int8", "trust_remote_code": False, "qwen": True,
        "int8": True,
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def clean_text(value: Any) -> str:
    return str(value).strip()


def validate_inputs(master_path: Path, members_path: Path, channels: list[str]):
    master = read_csv(master_path)
    members = read_csv(members_path)
    required_master = {"blinded_character", "moe_id", "original_character"}
    required_master |= {CHANNELS[c] for c in channels}
    required_members = {"family_id", "family_order", "family_anchor",
                        "family_member", "member_order", "n_FSC", "moe_id"}
    missing_m = required_master - set(master.columns)
    missing_f = required_members - set(members.columns)
    if missing_m or missing_f:
        raise ValueError(f"Missing columns: master={sorted(missing_m)}, members={sorted(missing_f)}")
    if len(master) != 4808:
        raise ValueError(f"Semantic master must have 4,808 rows; found {len(master)}.")
    for col in ["blinded_character", "moe_id", "original_character"]:
        if master[col].duplicated().any() or (master[col].map(clean_text) == "").any():
            raise ValueError(f"Master column {col!r} must be complete and unique.")
    for c in channels:
        col = CHANNELS[c]
        master[col] = master[col].map(clean_text)
        if (master[col] == "").any():
            raise ValueError(f"Channel {c!r} has missing/blank inputs; frozen n cannot change.")

    members["member_order"] = pd.to_numeric(members["member_order"], errors="raise").astype(int)
    members["family_order"] = pd.to_numeric(members["family_order"], errors="raise").astype(int)
    members["n_FSC"] = pd.to_numeric(members["n_FSC"], errors="raise").astype(int)
    observed = members.groupby("family_anchor", sort=False).size().to_dict()
    if observed != EXPECTED_FAMILIES:
        raise ValueError(f"Frozen family sizes/order differ. Expected {EXPECTED_FAMILIES}; found {observed}.")
    if len(members) != 98 or members["moe_id"].duplicated().any():
        raise ValueError("Frozen membership must contain 98 non-overlapping MOE IDs.")
    lookup = pd.Series(np.arange(len(master), dtype=np.int64), index=master["moe_id"])
    if not set(members["moe_id"]).issubset(set(master["moe_id"])):
        raise ValueError("At least one frozen family member is absent from the semantic master.")
    members["master_row_index"] = members["moe_id"].map(lookup).astype(int)
    joined_char = members["master_row_index"].map(master["original_character"])
    if not np.array_equal(joined_char.to_numpy(), members["family_member"].to_numpy()):
        raise ValueError("MOE-ID join does not reproduce the frozen family characters exactly.")
    members = members.sort_values(["family_order", "member_order"], kind="stable")
    return master.reset_index(drop=True), members.reset_index(drop=True)


def derived_seed(master_seed: int, family_id: str) -> int:
    digest = hashlib.sha256(f"{master_seed}|{family_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def create_or_load_controls(master: pd.DataFrame, members: pd.DataFrame,
                            output: Path, n_controls: int, seed: int):
    npz_path = output / "Phase_II_D_Control_Indices_v1.npz"
    long_path = output / "Phase_II_D_Control_Indices_v1.csv"
    manifest_path = output / "Phase_II_D_Control_Indices_v1.json"
    family_manifest = []
    arrays: dict[str, np.ndarray] = {}
    if npz_path.exists() and manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {"seed": seed, "n_controls": n_controls,
                    "master_moe_ids_sha256": json_hash(master["moe_id"].tolist())}
        if all(saved.get(k) == v for k, v in expected.items()):
            with np.load(npz_path) as data:
                arrays = {k: data[k].astype(np.int64) for k in data.files}
            return arrays, saved, True
        raise RuntimeError("Existing control indices do not match this seed/count/master. Use a new output folder.")

    all_rows = np.arange(len(master), dtype=np.int64)
    audit_rows = []
    for anchor, group in members.groupby("family_anchor", sort=False):
        family_id = group["family_id"].iloc[0]
        family_rows = group.sort_values("member_order")["master_row_index"].to_numpy(np.int64)
        eligible = np.setdiff1d(all_rows, family_rows, assume_unique=True)
        fseed = derived_seed(seed, family_id)
        rng = np.random.default_rng(fseed)
        samples = np.vstack([rng.choice(eligible, len(family_rows), replace=False)
                             for _ in range(n_controls)]).astype(np.int64)
        key = family_id
        arrays[key] = samples
        if np.any(np.apply_along_axis(lambda x: len(np.unique(x)) != len(x), 1, samples)):
            raise AssertionError(f"Within-control duplication detected for {anchor}.")
        if np.isin(samples, family_rows).any():
            raise AssertionError(f"Target-family leakage detected for {anchor}.")
        for sample_idx, sample in enumerate(samples, 1):
            for position, row_idx in enumerate(sample, 1):
                audit_rows.append({
                    "family_id": family_id, "family_anchor": anchor,
                    "sample_id": sample_idx, "position": position,
                    "master_row_index": int(row_idx),
                    "blinded_character": master.at[row_idx, "blinded_character"],
                    "moe_id": master.at[row_idx, "moe_id"],
                })
        family_manifest.append({"family_id": family_id, "family_anchor": anchor,
                                "n_f": len(family_rows), "eligible_pool_n": len(eligible),
                                "derived_seed": fseed})
    np.savez_compressed(npz_path, **arrays)
    pd.DataFrame(audit_rows).to_csv(long_path, index=False, encoding="utf-8-sig")
    manifest = {"version": VERSION, "created_utc": now_iso(), "seed": seed,
                "n_controls": n_controls,
                "sampling": "family-specific simple random samples; without replacement within group; target family excluded only",
                "master_moe_ids_sha256": json_hash(master["moe_id"].tolist()),
                "families": family_manifest, "npz_sha256": sha256_file(npz_path)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return arrays, manifest, False


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.all(np.isfinite(x)) or np.any(norms <= 0):
        raise RuntimeError("Embeddings contain non-finite or zero-length vectors.")
    return (x / norms).astype(np.float32, copy=False)


class Int8QwenEncoder:
    def __init__(self, hf_id: str, max_length: int):
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-Embedding-4B INT8 requires a CUDA GPU (T4 approved).")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, padding_side="left")
        self.model = AutoModel.from_pretrained(
            hf_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0}, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        from tqdm.auto import tqdm
        output = []
        with torch.inference_mode():
            for start in tqdm(range(0, len(texts), batch_size), desc="Qwen4B unique texts"):
                tokens = self.tokenizer(texts[start:start + batch_size], padding=True,
                                        truncation=True, max_length=self.max_length,
                                        return_tensors="pt").to("cuda")
                hidden = self.model(**tokens).last_hidden_state
                # Left padding makes the last position the final real token for every row.
                pooled = torch.nn.functional.normalize(hidden[:, -1], p=2, dim=1)
                output.append(pooled.float().cpu().numpy())
        return np.concatenate(output).astype(np.float32, copy=False)


def load_model(model_key: str, spec: dict, max_length: int):
    if spec.get("int8"):
        return Int8QwenEncoder(spec["hf_id"], max_length)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(spec["hf_id"], trust_remote_code=spec["trust_remote_code"],
                                device=device)
    model.max_seq_length = max_length
    return model


def actual_inputs(texts: list[str], spec: dict, no_qwen_instruction: bool) -> list[str]:
    if spec["qwen"] and not no_qwen_instruction:
        return [QWEN_INSTRUCTION + t for t in texts]
    return texts


def resolved_model_revision(model) -> str:
    """Best-effort Hub commit hash recorded by Transformers/ST after loading."""
    candidates = [getattr(model, "model", None), model]
    try:
        candidates.extend(list(model._modules.values()))
    except Exception:
        pass
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        revision = getattr(config, "_commit_hash", None)
        if revision:
            return str(revision)
        auto_model = getattr(candidate, "auto_model", None)
        revision = getattr(getattr(auto_model, "config", None), "_commit_hash", None)
        if revision:
            return str(revision)
    return "unavailable"


def cache_config(model_key: str, spec: dict, channel: str, texts: list[str],
                 max_length: int, no_qwen_instruction: bool, model_revision: str) -> dict:
    inputs = actual_inputs(texts, spec, no_qwen_instruction)
    return {
        "cache_schema": 1, "model_key": model_key, "hf_id": spec["hf_id"],
        "requested_revision": "default Hugging Face revision",
        "resolved_model_revision": model_revision,
        "dimension": spec["dimension"], "pooling": spec["pooling"],
        "normalization": "L2 float32 after pooling", "precision": spec["precision"],
        "channel": channel, "source_column": CHANNELS[channel],
        "max_length": max_length,
        "qwen_instruction": (QWEN_INSTRUCTION if spec["qwen"] and not no_qwen_instruction else ""),
        "input_text_table_sha256": json_hash(inputs), "character_order_sha256": None,
    }


def encode_channel(model, inputs: list[str], batch_size: int, is_int8: bool) -> np.ndarray:
    codes, unique = pd.factorize(pd.Index(inputs), sort=False)
    unique_texts = unique.tolist()
    if is_int8:
        unique_vectors = model.encode(unique_texts, batch_size=batch_size)
    else:
        unique_vectors = model.encode(unique_texts, batch_size=batch_size,
                                      normalize_embeddings=True, convert_to_numpy=True,
                                      show_progress_bar=True)
    return l2_normalize(np.asarray(unique_vectors)[codes]), len(unique_texts)


def load_or_build_cache(model, model_key: str, spec: dict, channel: str,
                        master: pd.DataFrame, cache_root: Path, batch_size: int,
                        max_length: int, no_qwen_instruction: bool):
    cache_dir = cache_root / model_key / channel
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = cache_dir / "embeddings.npy"
    index_path = cache_dir / "character_index.csv"
    meta_path = cache_dir / "cache_metadata.json"
    texts = master[CHANNELS[channel]].tolist()
    cfg = cache_config(model_key, spec, channel, texts, max_length,
                       no_qwen_instruction, resolved_model_revision(model))
    cfg["character_order_sha256"] = json_hash(master["moe_id"].tolist())
    cfg_hash = json_hash(cfg)
    if vectors_path.exists() and index_path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("cache_config_sha256") == cfg_hash:
            vectors = np.load(vectors_path)
            if vectors.shape == (len(master), spec["dimension"]):
                return vectors, old, True
        raise RuntimeError(f"Stale/incompatible cache at {cache_dir}; use a new output folder or remove that cache.")
    inputs = actual_inputs(texts, spec, no_qwen_instruction)
    vectors, unique_n = encode_channel(model, inputs, batch_size, spec.get("int8", False))
    if vectors.shape != (len(master), spec["dimension"]):
        raise RuntimeError(f"Unexpected embedding shape {vectors.shape} for {model_key}/{channel}.")
    np.save(vectors_path, vectors, allow_pickle=False)
    master[["blinded_character", "moe_id", "original_character"]].assign(
        matrix_row=np.arange(len(master))).to_csv(index_path, index=False, encoding="utf-8-sig")
    meta = {**cfg, "cache_config_sha256": cfg_hash, "created_utc": now_iso(),
            "master_rows": len(master), "unique_texts_embedded": unique_n,
            "duplicate_rows_saved": len(master) - unique_n,
            "embeddings_sha256": sha256_file(vectors_path),
            "index_sha256": sha256_file(index_path)}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return vectors, meta, False


def fsc_one(vectors: np.ndarray, indices: np.ndarray) -> float:
    x = vectors[indices]
    gram = x @ x.T
    n = len(indices)
    return float((gram.sum(dtype=np.float64) - np.trace(gram, dtype=np.float64)) / (n * (n - 1)))


def fsc_many(vectors: np.ndarray, indices: np.ndarray, chunk: int = 250) -> np.ndarray:
    out = np.empty(len(indices), dtype=np.float64)
    for start in range(0, len(indices), chunk):
        selected = vectors[indices[start:start + chunk]]
        grams = np.matmul(selected, np.swapaxes(selected, 1, 2))
        n = selected.shape[1]
        totals = grams.sum(axis=(1, 2), dtype=np.float64)
        traces = np.trace(grams, axis1=1, axis2=2).astype(np.float64)
        out[start:start + len(selected)] = (totals - traces) / (n * (n - 1))
    return out


def score_channel(vectors: np.ndarray, model_key: str, spec: dict, channel: str,
                  members: pd.DataFrame, controls: dict[str, np.ndarray]):
    long_rows, summary_rows = [], []
    for anchor, group in members.groupby("family_anchor", sort=False):
        group = group.sort_values("member_order")
        family_id = group["family_id"].iloc[0]
        test_indices = group["master_row_index"].to_numpy(np.int64)
        null = fsc_many(vectors, controls[family_id])
        test = fsc_one(vectors, test_indices)
        mean = float(null.mean())
        sd = float(null.std(ddof=1))
        ge = int(np.count_nonzero(null >= test))
        less = int(np.count_nonzero(null < test))
        equal = int(np.count_nonzero(np.isclose(null, test, rtol=0, atol=1e-12)))
        percentile = 100.0 * (less + 0.5 * equal) / len(null)
        summary_rows.append({
            "family_id": family_id, "family_anchor": anchor, "n_f": len(test_indices),
            "model": model_key, "model_identifier": spec["hf_id"],
            "embedding_dimension": spec["dimension"], "weight_precision": spec["precision"],
            "semantic_channel": channel, "source_column": CHANNELS[channel],
            "FSC_test": test, "control_n": len(null), "control_mean": mean,
            "control_median": float(np.median(null)), "control_sd": sd,
            "control_min": float(null.min()), "control_max": float(null.max()),
            "delta_FSC": test - mean, "z_score": ((test - mean) / sd if sd > 0 else np.nan),
            "empirical_percentile": percentile,
            "empirical_upper_tail_p": (1 + ge) / (len(null) + 1),
            "upper_tail_count_ge_test": ge,
        })
        long_rows.extend({"family_id": family_id, "family_anchor": anchor,
                          "sample_id": i + 1, "model": model_key,
                          "semantic_channel": channel, "FSC_control": float(value)}
                         for i, value in enumerate(null))
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def write_qc(output: Path, master: pd.DataFrame, members: pd.DataFrame,
             channels: list[str], n_controls: int, seed: int, cache_events: list[dict],
             controls_hit: bool, failures: list[dict], summary: pd.DataFrame):
    lines = ["# Phase II-D QC Report v1", "", "## Outcome", "",
             f"- Run completed UTC: `{now_iso()}`", f"- Master rows: **{len(master):,}**",
             f"- Frozen test memberships: **{len(members)}**", f"- Families: **{members.family_anchor.nunique()}**",
             f"- Controls per family: **{n_controls:,}**", f"- Master seed: **{seed}**",
             f"- Control-index cache: **{'HIT' if controls_hit else 'CREATED'}**", "",
             "## Family-size validation", "",
             "| Anchor | Expected n | Observed n | Status |", "|---|---:|---:|---|"]
    observed = members.groupby("family_anchor", sort=False).size()
    for a, n in EXPECTED_FAMILIES.items():
        lines.append(f"| {a} | {n} | {int(observed[a])} | PASS |")
    lines += ["", "## Semantic inputs", "", "| Channel | Source column | Missing | Unique texts |",
              "|---|---|---:|---:|"]
    for c in channels:
        s = master[CHANNELS[c]]
        lines.append(f"| {c} | `{CHANNELS[c]}` | {int((s == '').sum())} | {s.nunique()} |")
    lines += ["", "## Embedding caches", "", "| Model | Channel | Status | Unique embedded | Duplicate rows saved |",
              "|---|---|---|---:|---:|"]
    for e in cache_events:
        lines.append(f"| {e['model']} | {e['channel']} | {e['status']} | {e['unique']} | {e['saved']} |")
    lines += ["", "## Control integrity", "",
              "- Target family excluded from its own controls: **PASS**",
              "- Sampling without replacement within every group: **PASS**",
              "- Family-specific control indices persisted and reused across models/channels: **PASS**",
              "", "## Scoring integrity", "",
              "- Frozen membership retained; no member silently dropped: **PASS**",
              "- FSC uses all unique unordered within-channel pairs: **PASS**",
              "- Empirical upper-tail p-values use the +1 correction: **PASS**",
              f"- Completed summary rows: **{len(summary):,}**", "",
              "## Pilot comparison / deviations", "",
              "- Phase II computes within-channel FSC per the Phase II-D specification; it does not add Pilot14 cross-channel FSC pathways.",
              "- The Qwen task instruction is recorded in every Qwen cache; use `--no-qwen-instruction` only as a separately versioned ablation.",
              "- Qing14 replication was not run because its frozen audit artifact was not supplied."]
    if failures:
        lines += ["", "## Failures", ""] + [f"- `{x['model']}`: {x['error']}" for x in failures]
    else:
        lines += ["", "## Failures", "", "None."]
    (output / "Phase_II_D_QC_Report_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", required=True, type=Path)
    p.add_argument("--members", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("Phase_II_D_results_v1"))
    p.add_argument("--n-controls", type=int, default=1000)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--qwen4b-batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--include-exploratory-google-character", action="store_true")
    p.add_argument("--no-qwen-instruction", action="store_true",
                   help="Ablation only: embed raw definitions without the pilot Qwen instruction.")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--continue-on-model-error", action="store_true")
    args = p.parse_args()
    if args.n_controls < 2 or min(args.batch_size, args.qwen4b_batch_size) < 1:
        raise ValueError("Need at least 2 controls and batch sizes >= 1.")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    channels = PRIMARY_CHANNELS.copy()
    if args.include_exploratory_google_character:
        channels.append("en_google_character")
    master, members = validate_inputs(args.master, args.members, channels)
    master.to_csv(args.output / "validated_semantic_master_index_v1.csv", index=False,
                  columns=["blinded_character", "moe_sequence", "moe_id", "unicode", "original_character"],
                  encoding="utf-8-sig")
    members.to_csv(args.output / "validated_frozen10_members_v1.csv", index=False, encoding="utf-8-sig")
    controls, control_manifest, controls_hit = create_or_load_controls(
        master, members, args.output, args.n_controls, args.seed)

    long_path = args.output / "Phase_II_D_FSC_Control_Long_v1.csv"
    summary_path = args.output / "Phase_II_D_FSC_Test_Summary_v1.csv"
    completed_long, completed_summary = [], []
    cache_events, failures, model_metadata = [], [], {}
    for model_key in args.models:
        spec = MODELS[model_key]
        print(f"\nLoading {model_key}: {spec['hf_id']}")
        model_start = time.time()
        try:
            model = load_model(model_key, spec, args.max_length)
            batch = args.qwen4b_batch_size if spec.get("int8") else args.batch_size
            for channel in channels:
                vectors, cache_meta, hit = load_or_build_cache(
                    model, model_key, spec, channel, master, args.output / "embedding_caches",
                    batch, args.max_length, args.no_qwen_instruction)
                cache_events.append({"model": model_key, "channel": channel,
                                     "status": "HIT" if hit else "CREATED",
                                     "unique": cache_meta["unique_texts_embedded"],
                                     "saved": cache_meta["duplicate_rows_saved"]})
                long_df, summary_df = score_channel(vectors, model_key, spec, channel,
                                                    members, controls)
                completed_long.append(long_df)
                completed_summary.append(summary_df)
                pd.concat(completed_long, ignore_index=True).to_csv(long_path, index=False, encoding="utf-8-sig")
                pd.concat(completed_summary, ignore_index=True).to_csv(summary_path, index=False, encoding="utf-8-sig")
                del vectors
            model_metadata[model_key] = {**spec, "status": "completed",
                                         "runtime_seconds": time.time() - model_start}
            del model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as exc:
            failure = {"model": model_key, "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
            failures.append(failure)
            model_metadata[model_key] = {**spec, "status": "failed", "error": failure["error"]}
            if not args.continue_on_model_error:
                raise
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    all_long = pd.concat(completed_long, ignore_index=True) if completed_long else pd.DataFrame()
    all_summary = pd.concat(completed_summary, ignore_index=True) if completed_summary else pd.DataFrame()
    expected_summary = len(args.models) * len(channels) * len(EXPECTED_FAMILIES) - len(failures) * len(channels) * len(EXPECTED_FAMILIES)
    if not failures and len(all_summary) != expected_summary:
        raise AssertionError(f"Expected {expected_summary} summary rows, found {len(all_summary)}.")
    if not all_long.empty and len(all_long) != len(all_summary) * args.n_controls:
        raise AssertionError("Long-control row count does not match summary rows × controls.")

    metadata = {
        "artifact": "Phase_II_D_Run_Metadata_v1", "script_version": VERSION,
        "started_utc_approx": datetime.fromtimestamp(time.time() - (time.time() - started), timezone.utc).isoformat(),
        "finished_utc": now_iso(), "runtime_seconds": time.time() - started,
        "inputs": {"master": str(args.master), "master_sha256": sha256_file(args.master),
                   "members": str(args.members), "members_sha256": sha256_file(args.members)},
        "master_rows": len(master), "family_members": len(members),
        "channels": channels, "models": model_metadata, "n_controls": args.n_controls,
        "master_seed": args.seed, "control_manifest": control_manifest,
        "qwen_instruction": ("" if args.no_qwen_instruction else QWEN_INSTRUCTION),
        "normalization": "L2", "FSC": "mean cosine over n(n-1)/2 unique unordered pairs",
        "hardware": {"platform": platform.platform(), "cuda_available": torch.cuda.is_available(),
                     "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "cuda_runtime": torch.version.cuda},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "sentence_transformers": sentence_transformers.__version__},
        "cache_events": cache_events, "failures": failures,
        "output_rows": {"control_long": len(all_long), "test_summary": len(all_summary)},
    }
    (args.output / "Phase_II_D_Run_Metadata_v1.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_qc(args.output, master, members, channels, args.n_controls, args.seed,
             cache_events, controls_hit, failures, all_summary)
    zip_path = shutil.make_archive(str(args.output.resolve()), "zip", root_dir=args.output.resolve())
    print(f"\nCompleted {len(all_summary)} test summaries and {len(all_long)} control rows.")
    print(f"Results: {args.output.resolve()}\nZIP: {zip_path}")


if __name__ == "__main__":
    main()
