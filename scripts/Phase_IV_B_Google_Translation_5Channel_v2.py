#!/usr/bin/env python3
"""Phase IV-B Google Translation + five-channel preparation (Colab-ready).

Colab setup (run in separate cells):

    !pip -q install --upgrade google-cloud-translate pandas

    from google.colab import auth
    auth.authenticate_user()

Test both jobs on five rows (separate output directory):

    !python /content/Phase_IV_B_Google_Translation_5Channel_v2.py \
      --input /content/Phase_IV_Relational_11151_Data_v1/relational_character_master_v1.csv \
      --output-dir /content/Phase_IV_B_translation_test_v2 \
      --project-id YOUR_PROJECT_ID --mode test --test-rows 5

Full resumable run:

    !python /content/Phase_IV_B_Google_Translation_5Channel_v2.py \
      --input /content/Phase_IV_Relational_11151_Data_v1/relational_character_master_v1.csv \
      --output-dir /content/Phase_IV_B_Translation_v2 \
      --project-id YOUR_PROJECT_ID --mode full

Rerun the identical full command after interruption. Completed, source-matched
rows are skipped. No credential or secret is written to an artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


VERSION = "2.0.0"
OUTPUT_CSV = "relational_character_master_5channel_translated_v2.csv"
DEF_LOG = "google_translate_definition_raw_log_v2.jsonl"
CHAR_LOG = "google_translate_character_raw_log_v2.jsonl"
MANIFEST = "google_translate_run_manifest_v2.json"
QC_MD = "google_translate_QC_v2.md"

REQUIRED = [
    "char_id", "original_character", "definition_cn",
    "embed_text_cn_moe", "definition_en_unihan",
]

CJK_RE = re.compile(
    "[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "\U00020000-\U0002ffff\U00030000-\U000323af]"
)
PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


@dataclass(frozen=True)
class Job:
    key: str
    source_col: str
    target_col: str
    embed_col: str
    hash_col: str
    log_name: str


JOBS = {
    "definition": Job(
        "definition", "definition_cn", "definition_en_from_cn_definition",
        "embed_text_en_google_from_moe", "_google_definition_source_sha256", DEF_LOG,
    ),
    "character": Job(
        "character", "original_character", "definition_en_google_character",
        "embed_text_en_google_character", "_google_character_source_sha256", CHAR_LOG,
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    os.replace(tmp, path)


def atomic_write_json(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict) -> None:
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")
    if len(df) == 0:
        raise ValueError("Input has zero rows")
    if df["char_id"].map(clean).eq("").any():
        raise ValueError("Blank char_id values are not allowed")
    if df["char_id"].map(clean).duplicated().any():
        examples = df.loc[df["char_id"].map(clean).duplicated(False), "char_id"].head().tolist()
        raise ValueError(f"Duplicate char_id values prevent safe resume; examples: {examples}")


def initialize_channels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["embed_text_zh_char"] = out["original_character"].map(clean)
    if "embed_text_en_unihan" not in out.columns:
        out["embed_text_en_unihan"] = out["definition_en_unihan"].map(clean)
    else:
        # Existing values are preserved exactly; blanks are not Google-imputed.
        out["embed_text_en_unihan"] = out["embed_text_en_unihan"].map(clean)
    for job in JOBS.values():
        for col in (job.target_col, job.embed_col, job.hash_col):
            if col not in out.columns:
                out[col] = ""
            else:
                out[col] = out[col].map(clean)
    return out


def load_or_initialize(input_path: Path, output_path: Path, row_limit: int | None = None) -> pd.DataFrame:
    base = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    validate_input(base)
    if row_limit is not None:
        base = base.iloc[:row_limit].copy().reset_index(drop=True)
    base = initialize_channels(base)
    if not output_path.exists():
        return base

    prior = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    validate_input(prior)
    if prior["char_id"].tolist() != base["char_id"].tolist():
        raise RuntimeError(
            "Resume refused: output char_id sequence differs from the current input. "
            "Use the original frozen source or a new output directory."
        )
    # Frozen source/channel columns must not drift between runs.
    protected = [
        "original_character", "definition_cn", "embed_text_cn_moe",
        "definition_en_unihan", "embed_text_en_unihan", "embed_text_zh_char",
    ]
    for col in protected:
        if col not in prior.columns or prior[col].map(clean).tolist() != base[col].map(clean).tolist():
            raise RuntimeError(f"Resume refused: frozen column changed: {col}")
    # Retain source columns from current input, translation/provenance from prior.
    for job in JOBS.values():
        for col in (job.target_col, job.embed_col, job.hash_col):
            if col in prior.columns:
                base[col] = prior[col].map(clean)
    return base


def check_completed_source_integrity(df: pd.DataFrame, job: Job) -> None:
    for idx, row in df.iterrows():
        target, saved_hash = clean(row[job.target_col]), clean(row[job.hash_col])
        if not target:
            continue
        current_hash = source_hash(clean(row[job.source_col]))
        if not saved_hash:
            raise RuntimeError(
                f"Row {idx} ({row['char_id']}) has a completed {job.target_col} but no "
                f"source fingerprint. Refusing to guess whether it is safe to resume."
            )
        if saved_hash != current_hash:
            raise RuntimeError(
                f"Row {idx} ({row['char_id']}) completed target belongs to a different "
                f"{job.source_col}. Use a new output directory."
            )
        if clean(row[job.embed_col]) != target:
            raise RuntimeError(f"Row {idx}: {job.embed_col} is not an exact target copy")


def pending_indices(df: pd.DataFrame, job: Job) -> list[int]:
    check_completed_source_integrity(df, job)
    pending = []
    for idx, row in df.iterrows():
        source, target = clean(row[job.source_col]), clean(row[job.target_col])
        if target:
            continue
        if not source:
            # Blank sources are QC findings, not valid API submissions.
            continue
        pending.append(idx)
    return pending


def make_batches(
    indices: Sequence[int], sources: Sequence[str], max_items: int, max_codepoints: int
) -> Iterable[list[int]]:
    batch: list[int] = []
    chars = 0
    for idx in indices:
        length = len(sources[idx])
        if length > max_codepoints:
            raise ValueError(f"Row {idx} alone exceeds --max-codepoints ({length:,})")
        if batch and (len(batch) >= max_items or chars + length > max_codepoints):
            yield batch
            batch, chars = [], 0
        batch.append(idx)
        chars += length
    if batch:
        yield batch


def package_version() -> str:
    try:
        return importlib.metadata.version("google-cloud-translate")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def make_client(location: str):
    try:
        from google.cloud import translate_v3
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-translate is not installed. In Colab run: "
            "!pip -q install --upgrade google-cloud-translate pandas"
        ) from exc
    kwargs = {}
    if location != "global":
        kwargs["client_options"] = {"api_endpoint": f"{location}-translate.googleapis.com"}
    return translate_v3.TranslationServiceClient(**kwargs)


def translate_with_retries(client, request: dict, max_attempts: int, timeout: float):
    from google.api_core import exceptions as gx

    retryable = (
        gx.DeadlineExceeded, gx.InternalServerError, gx.ResourceExhausted,
        gx.ServiceUnavailable, gx.TooManyRequests,
    )
    attempts = []
    for attempt in range(1, max_attempts + 1):
        started = time.monotonic()
        try:
            response = client.translate_text(request=request, timeout=timeout)
            attempts.append({"attempt": attempt, "status": "success", "elapsed_seconds": round(time.monotonic()-started, 3)})
            return response, attempts
        except retryable as exc:
            attempts.append({
                "attempt": attempt, "status": "retryable_error",
                "error_type": type(exc).__name__, "error": str(exc)[:1000],
                "elapsed_seconds": round(time.monotonic()-started, 3),
            })
            if attempt == max_attempts:
                raise
            delay = min(60.0, 2 ** (attempt - 1)) + random.random()
            print(f"    retryable {type(exc).__name__}; waiting {delay:.1f}s", flush=True)
            time.sleep(delay)


def run_job(df: pd.DataFrame, job: Job, args, output_path: Path, log_path: Path, client) -> dict:
    sources = df[job.source_col].map(clean).tolist()
    pending = pending_indices(df, job)
    batches = list(make_batches(pending, sources, args.batch_items, args.max_codepoints))
    stats = {"job": job.key, "pending_at_start": len(pending), "batches": len(batches), "translated_this_run": 0, "retry_count": 0}
    print(f"{job.key}: {len(pending):,} pending rows in {len(batches):,} batches", flush=True)

    parent = f"projects/{args.project_id}/locations/{args.location}"
    model = args.model or f"{parent}/models/general/nmt"
    for number, batch in enumerate(batches, 1):
        contents = [sources[i] for i in batch]
        request = {
            "parent": parent, "contents": contents, "mime_type": "text/plain",
            "source_language_code": args.source_language,
            "target_language_code": args.target_language, "model": model,
        }
        batch_id = str(uuid.uuid4())
        call_started = utc_now()
        try:
            response, attempts = translate_with_retries(client, request, args.max_attempts, args.timeout)
        except Exception as exc:
            append_jsonl(log_path, {
                "schema_version": 2, "batch_id": batch_id, "job": job.key,
                "status": "failed", "utc": utc_now(), "row_indices": batch,
                "char_ids": [clean(df.at[i, "char_id"]) for i in batch],
                "source_sha256": [source_hash(s) for s in contents],
                "error_type": type(exc).__name__, "error": str(exc)[:2000],
            })
            print(f"FAILED {job.key} batch {number}/{len(batches)}; checkpoint remains resumable", flush=True)
            raise

        translations = list(response.translations)
        if len(translations) != len(batch):
            append_jsonl(log_path, {
                "schema_version": 2, "batch_id": batch_id, "job": job.key,
                "status": "response_length_mismatch", "utc": utc_now(),
                "submitted_count": len(batch), "returned_count": len(translations),
                "row_indices": batch,
            })
            raise RuntimeError(
                f"{job.key} response mismatch: submitted {len(batch)}, returned {len(translations)}"
            )
        returned = [clean(t.translated_text) for t in translations]
        if any(not x for x in returned):
            raise RuntimeError(f"{job.key} batch {batch_id} returned at least one blank translation")

        # Durable raw record first. If interrupted before CSV checkpoint, the
        # batch is safely resubmitted; no completed CSV value is overwritten.
        append_jsonl(log_path, {
            "schema_version": 2, "batch_id": batch_id, "job": job.key,
            "status": "success", "call_started_utc": call_started,
            "call_finished_utc": utc_now(), "row_indices": batch,
            "char_ids": [clean(df.at[i, "char_id"]) for i in batch],
            "sources": contents, "source_sha256": [source_hash(s) for s in contents],
            "translations": returned,
            "detected_language_codes": [clean(getattr(t, "detected_language_code", "")) for t in translations],
            "model": model, "location": args.location,
            "source_language": args.source_language, "target_language": args.target_language,
            "codepoints": sum(len(s) for s in contents), "attempts": attempts,
        })
        for idx, source, translated in zip(batch, contents, returned):
            # This assertion guards against future loop/refactor alignment bugs.
            if clean(df.at[idx, job.source_col]) != source:
                raise RuntimeError(f"Internal alignment failure at row {idx}")
            df.at[idx, job.target_col] = translated
            df.at[idx, job.embed_col] = translated
            df.at[idx, job.hash_col] = source_hash(source)
        atomic_write_csv(df, output_path)
        retry_count = sum(1 for a in attempts if a["status"] != "success")
        stats["retry_count"] += retry_count
        stats["translated_this_run"] += len(batch)
        print(
            f"  {job.key} batch {number}/{len(batches)}: {len(batch)} rows; "
            f"{stats['translated_this_run']:,}/{len(pending):,} checkpointed",
            flush=True,
        )
    return stats


def duplicate_nonblank_count(series: pd.Series) -> int:
    vals = [clean(v) for v in series if clean(v)]
    counts = Counter(vals)
    return sum(n for n in counts.values() if n > 1)


def pct(n: int, d: int) -> str:
    return f"{(100*n/d):.2f}%" if d else "n/a"


def log_stats(path: Path) -> tuple[int, int]:
    failures = retries = 0
    if not path.exists():
        return failures, retries
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue
            failures += int(x.get("status") == "failed")
            retries += sum(int(a.get("status") == "retryable_error") for a in x.get("attempts", []))
    return failures, retries


def suspicious_length_counts(sources: pd.Series, targets: pd.Series) -> tuple[int, int]:
    short = long = 0
    for source, target in zip(sources.map(clean), targets.map(clean)):
        if not target:
            continue
        short += int(len(target.strip()) <= 1)
        # Broad flag only; preserve raw output for human QC.
        long += int(len(target) > max(300, len(source) * 8))
    return short, long


def diagnostic_mask(df: pd.DataFrame) -> pd.Series:
    fixed = set("店掂惦堯尧僥侥饒饶繞绕燒烧曉晓蹺跷翹翘青清情晴請请精睛靜静靖菁倩")
    mask = df["original_character"].map(clean).isin(fixed)
    # If the relational master exposes a component/phonophore column, include
    # every explicit 青/堯 family member instead of relying only on the fixed list.
    family_cols = [c for c in df.columns if any(k in c.lower() for k in ("phonophore", "component", "family"))]
    for col in family_cols:
        mask |= df[col].map(clean).str.contains("青|堯", regex=True, na=False)
    return mask


def generate_qc(df: pd.DataFrame, input_rows: int, output_dir: Path) -> str:
    n = len(df)
    unihan_nonblank = int(df["embed_text_en_unihan"].map(clean).ne("").sum())
    def_target = df[JOBS["definition"].target_col].map(clean)
    char_target = df[JOBS["character"].target_col].map(clean)
    def_short, def_long = suspicious_length_counts(df["definition_cn"], def_target)
    char_short, char_long = suspicious_length_counts(df["original_character"], char_target)
    def_fail, def_retry = log_stats(output_dir / DEF_LOG)
    char_fail, char_retry = log_stats(output_dir / CHAR_LOG)
    cn_mismatch = int((df["definition_cn"].map(clean) != df["embed_text_cn_moe"].map(clean)).sum())
    # Mismatch is reported, not automatically treated as an error: the frozen
    # DataCollector channel may intentionally format the definition differently.
    lines = [
        "# Phase IV-B Google Translation QC v2", "",
        f"Generated UTC: `{utc_now()}`", "",
        "## Population integrity", "",
        f"- Input rows: {input_rows:,}", f"- Output rows: {n:,}",
        f"- Duplicate `char_id`: {int(df['char_id'].map(clean).duplicated().sum()):,}",
        f"- Duplicate `original_character`: {int(df['original_character'].map(clean).duplicated().sum()):,}",
        "- Row order: preserved by construction and verified on resume", "",
        "## MOE Chinese", "",
        f"- Blank `definition_cn`: {int(df['definition_cn'].map(clean).eq('').sum()):,}",
        f"- Blank `embed_text_cn_moe`: {int(df['embed_text_cn_moe'].map(clean).eq('').sum()):,}",
        f"- Literal `definition_cn` vs `embed_text_cn_moe` mismatches (reported, not repaired): {cn_mismatch:,}", "",
        "## Unihan gloss", "",
        f"- Nonblank coverage: {unihan_nonblank:,} / {n:,} ({pct(unihan_nonblank,n)})",
        f"- Missing: {n-unihan_nonblank:,} / {n:,} ({pct(n-unihan_nonblank,n)})",
        f"- Rows participating in duplicated nonblank glosses: {duplicate_nonblank_count(df['embed_text_en_unihan']):,}",
        "- Missing values were not imputed", "",
        "## Google definition translation", "",
        f"- Nonblank sources / submitted population: {int(df['definition_cn'].map(clean).ne('').sum()):,}",
        f"- Translated rows: {int(def_target.ne('').sum()):,}",
        f"- Blank source rows: {int(df['definition_cn'].map(clean).eq('').sum()):,}",
        f"- Blank target rows: {int(def_target.eq('').sum()):,}",
        f"- Rows participating in duplicated sources: {duplicate_nonblank_count(df['definition_cn']):,}",
        f"- Rows participating in duplicated translations: {duplicate_nonblank_count(def_target):,}",
        f"- Targets containing Chinese script: {int(def_target.map(lambda x: bool(CJK_RE.search(x))).sum()):,}",
        f"- Suspiciously short targets (<=1 character): {def_short:,}",
        f"- Suspiciously long targets: {def_long:,}",
        f"- Failed API batches logged: {def_fail:,}", f"- Retry attempts logged: {def_retry:,}", "",
        "## Google character gloss", "",
        f"- Nonblank sources / submitted population: {int(df['original_character'].map(clean).ne('').sum()):,}",
        f"- Translated rows: {int(char_target.ne('').sum()):,}",
        f"- Blank target rows: {int(char_target.eq('').sum()):,}",
        f"- Unchanged-character outputs: {int((char_target == df['original_character'].map(clean)).sum()):,}",
        f"- Targets containing Chinese script: {int(char_target.map(lambda x: bool(CJK_RE.search(x))).sum()):,}",
        f"- Punctuation-only outputs: {int(char_target.map(lambda x: bool(x) and bool(PUNCT_ONLY_RE.fullmatch(x))).sum()):,}",
        f"- Transliteration-like outputs: heuristic not auto-classified; inspect diagnostic/raw tables",
        f"- Suspiciously short targets (<=1 character): {char_short:,}",
        f"- Suspiciously long or multi-sentence targets: {char_long + int(char_target.map(lambda x: x.count('.') > 1).sum()):,}",
        f"- Failed API batches logged: {char_fail:,}", f"- Retry attempts logged: {char_retry:,}", "",
        "## Diagnostic five-channel audit", "",
    ]
    diag = df.loc[diagnostic_mask(df), [
        "original_character", "definition_cn", "definition_en_unihan",
        "definition_en_from_cn_definition", "definition_en_google_character",
    ]].copy()
    if diag.empty:
        lines.append("No requested diagnostic characters were present.")
    else:
        headers = ["Character", "MOE Chinese definition", "Unihan English gloss", "Google English from MOE", "Google direct-character gloss"]
        lines += ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*5) + "|"]
        for row in diag.itertuples(index=False, name=None):
            escaped = [clean(x).replace("|", "\\|").replace("\n", "<br>") for x in row]
            lines.append("| " + " | ".join(escaped) + " |")
    lines += ["", "Unusual Google outputs are flags for review only; none were replaced or post-edited.", ""]
    return "\n".join(lines)


def build_manifest(args, input_path: Path, output_path: Path, df: pd.DataFrame, run_stats: list[dict], started: str) -> dict:
    input_digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    return {
        "schema_version": 2, "script_version": VERSION, "run_started_utc": started,
        "manifest_written_utc": utc_now(), "mode": args.mode,
        "input_filename": input_path.name, "input_sha256": input_digest,
        "output_filename": output_path.name, "rows": len(df),
        "google_cloud_project_id": args.project_id,
        "api": "Cloud Translation Advanced v3 translate_text",
        "source_language": args.source_language, "target_language": args.target_language,
        "location": args.location,
        "model_identifier": args.model or f"projects/{args.project_id}/locations/{args.location}/models/general/nmt",
        "google_cloud_translate_package_version": package_version(),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "batch_items": args.batch_items, "max_codepoints_per_request": args.max_codepoints,
        "timeout_seconds": args.timeout, "max_attempts": args.max_attempts,
        "retry_backoff": "exponential 1,2,4,... seconds capped at 60 plus [0,1) jitter",
        "jobs_requested": args.jobs, "run_stats": run_stats,
        "analytical_channels": {
            "zh_char": "embed_text_zh_char", "cn_def": "embed_text_cn_moe",
            "en_gloss_unihan": "embed_text_en_unihan",
            "en_def_translated_cn_def": "embed_text_en_google_from_moe",
            "en_gloss_google": "embed_text_en_google_character",
        },
        "credentials_written": False,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--project-id", required=True)
    p.add_argument("--location", default="global")
    p.add_argument("--model", default="", help="Full model resource name; default is general/nmt")
    p.add_argument("--source-language", default="zh-TW")
    p.add_argument("--target-language", default="en")
    p.add_argument("--mode", choices=("test", "full", "qc-only"), default="test")
    p.add_argument("--test-rows", type=int, default=5)
    p.add_argument("--jobs", choices=("both", "definition", "character"), default="both")
    p.add_argument("--batch-items", type=int, default=100)
    p.add_argument("--max-codepoints", type=int, default=25000,
                   help="Conservative cap below Google's 30,000-codepoint request limit")
    p.add_argument("--max-attempts", type=int, default=6)
    p.add_argument("--timeout", type=float, default=120.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "test" and not 3 <= args.test_rows <= 10:
        raise ValueError("--test-rows must be between 3 and 10")
    if args.batch_items < 1 or args.max_codepoints < 1 or args.max_codepoints > 30000:
        raise ValueError("Invalid batch limits; max codepoints must be 1..30000")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / OUTPUT_CSV
    started = utc_now()
    row_limit = args.test_rows if args.mode == "test" else None
    df = load_or_initialize(args.input, output_path, row_limit=row_limit)
    input_rows = len(df)
    atomic_write_csv(df, output_path)

    run_stats: list[dict] = []
    if args.mode != "qc-only":
        client = make_client(args.location)
        selected = list(JOBS.values()) if args.jobs == "both" else [JOBS[args.jobs]]
        for job in selected:
            run_stats.append(run_job(df, job, args, output_path, args.output_dir / job.log_name, client))

    qc = generate_qc(df, input_rows if args.mode != "test" else len(df), args.output_dir)
    (args.output_dir / QC_MD).write_text(qc, encoding="utf-8")
    manifest = build_manifest(args, args.input, output_path, df, run_stats, started)
    atomic_write_json(manifest, args.output_dir / MANIFEST)
    print(f"Done. Output: {output_path}")
    print(f"QC: {args.output_dir / QC_MD}")
    print(f"Manifest: {args.output_dir / MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
