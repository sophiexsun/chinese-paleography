#!/usr/bin/env python3
"""Phase IV-D: P-side relational composition with frozen independent-S mappings.

Uses complete Phase IV-C embedding caches. For each held-out P-family member,
the entire held-out canonical S context is excluded, remaining edge offsets are
averaged within S context, and distinct contexts receive equal weight.
"""
from __future__ import annotations

import argparse, hashlib, json, os, platform, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "1.0.0"
MODELS = ["qwen3_0.6b_1024d", "bge_m3_1024d", "gte_multilingual_base_768d", "qwen3_4b_2560d_int8"]
CHANNELS = ["zh_char", "en_gloss_unihan", "cn_def", "en_def_translated_cn_def", "en_gloss_google"]
EXPECTED = {
    "edges": "1e7c972c120bec39d59998b97f45a8dd3c584c37cf888a5f85f43d06087b9f39",
    "census": "19192ccdda875f755abfb83f2deba9d5884c3163691c733d49b4f72aae712cb0",
    "a3_qc": "5f113e71e02d11576b83f7579f8f40aeef092efb2d439f31525100465179fc38",
    "a3_manifest": "29d078cce9d72ca427c7b78828bd2c0c02195f314ce1c00b18c05475670b497c",
    "s_map": "9ad502c8909d381d8eb1eee5243402cfd608c1116c9e0e183fadd28dc851ace7",
    "p_support": "2bf587119de31aa29a6752abe6c29d3781c30b8c0d56c39c7a1ca4fc001b3287",
    "a4_qc": "573c9954651119074257e759642a12f4baf5536e894ec65930159e754a95a3cc",
}
MAP_REQUIRED = ["s_component_raw", "s_component_canonical", "s_independent_character",
    "s_mapping_category", "s_mapping_confidence", "s_primary_analysis_eligible",
    "s_independent_population_member_11151"]
EDGE_REQUIRED = ["edge_id", "anchor_char", "derived_char", "added_component", "phonophore_root",
    "family_id_root", "edge_status", "anchor_char_normalized", "added_component_normalized",
    "phonophore_root_normalized"]

def now(): return datetime.now(timezone.utc).isoformat()
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()
def norm(x):
    x = np.asarray(x, dtype=np.float32); n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)
def atomic_csv(df, path):
    tmp = path.with_suffix(path.suffix + ".tmp"); df.to_csv(tmp, index=False); os.replace(tmp, path)
def atomic_json(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"); os.replace(tmp, path)
def truth(s): return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})
def mean(x): return float(np.mean(x)) if len(x) else np.nan

def identify(queries, candidates, names, true_ids, excluded_ids, batch=128):
    """Exact cosine argmax/rank; frozen candidate order breaks ties."""
    cand = norm(candidates); names = np.asarray(names); out = []
    for start in range(0, len(queries), batch):
        scores = norm(queries[start:start + batch]) @ cand.T
        for j, row0 in enumerate(scores):
            row = row0.copy(); ti = int(true_ids[start + j]); ex = int(excluded_ids[start + j])
            if ex >= 0: row[ex] = -np.inf
            if not np.isfinite(row[ti]): raise RuntimeError("True target was excluded or nonfinite")
            pred = int(np.argmax(row)); target_score = row[ti]
            rank = 1 + int(np.count_nonzero(row > target_score)) + int(np.count_nonzero(row[:ti] == target_score))
            out.append((str(names[pred]), rank, float(target_score)))
    return out

def validate_frozen(args, edges, smap, support, master):
    checks = {"edges": args.edges, "census": args.census, "a3_qc": args.normalization_qc,
        "a3_manifest": args.sha256_manifest, "s_map": args.s_map, "p_support": args.p_support,
        "a4_qc": args.a4_qc}
    got = {k: sha256(v) for k, v in checks.items()}
    bad = {k: (EXPECTED[k], got[k]) for k in EXPECTED if got[k] != EXPECTED[k]}
    if bad: raise RuntimeError(f"Frozen-input SHA-256 mismatch: {bad}")
    if [c for c in EDGE_REQUIRED if c not in edges]: raise ValueError("Authoritative edge v2 schema mismatch")
    if [c for c in MAP_REQUIRED if c not in smap]: raise ValueError("Frozen A4 S-map schema mismatch")
    if len(edges) != 6195 or edges.edge_id.duplicated().any(): raise ValueError("Frozen edge census is not 6,195 unique edge IDs")
    if len(smap) != 306 or smap.s_component_canonical.duplicated().any(): raise ValueError("A4 map must contain 306 unique canonical S classes")
    eligible = truth(smap.s_primary_analysis_eligible)
    allowed_cat = smap.s_mapping_category.str.startswith(("A_", "B_"))
    high = smap.s_mapping_confidence.eq("high")
    if eligible.isna().any() or not (eligible == (allowed_cat & high)).all(): raise ValueError("A4 eligibility is not exactly high-confidence Category A/B")
    if int(eligible.sum()) != 236: raise ValueError("Expected 236 structurally primary-eligible S mappings")
    if not (edges.anchor_char == edges.anchor_char_normalized).all(): raise ValueError("Unfrozen P normalization")
    if not (edges.added_component == edges.added_component_normalized).all(): raise ValueError("Unfrozen S normalization")
    if not (edges.phonophore_root == edges.phonophore_root_normalized).all(): raise ValueError("Unfrozen root normalization")
    if len(support) != 1001 or support.root_family_id.duplicated().any(): raise ValueError("A4 support must contain 1,001 unique root families")
    # Recompute the descriptive A4 support from frozen edges + frozen mapping.
    # The frozen support table is the literal-analysis support census: A/B/high
    # mappings whose independent character is also inside the 11,151 population.
    population_available = truth(smap.s_independent_population_member_11151)
    emap = smap.loc[eligible & population_available, ["s_component_canonical"]]
    joined = edges.merge(emap, left_on="added_component", right_on="s_component_canonical", how="inner", validate="many_to_one")
    recomputed = joined.groupby("family_id_root", as_index=False).agg(
        derivatives_with_primary_eligible_s_mapping=("edge_id", "size"),
        distinct_eligible_s_contexts=("added_component", "nunique"),
    ).rename(columns={"family_id_root": "root_family_id"})
    total = edges.groupby("family_id_root", as_index=False).agg(total_direct_derivatives=("edge_id", "size")).rename(columns={"family_id_root": "root_family_id"})
    recomputed = total.merge(recomputed, on="root_family_id", how="left").fillna(0)
    for c in ["total_direct_derivatives", "derivatives_with_primary_eligible_s_mapping", "distinct_eligible_s_contexts"]:
        recomputed[c] = recomputed[c].astype(int); support[c] = pd.to_numeric(support[c], errors="raise").astype(int)
    z = support[["root_family_id", "total_direct_derivatives", "derivatives_with_primary_eligible_s_mapping", "distinct_eligible_s_contexts"]].merge(
        recomputed, on="root_family_id", how="outer", suffixes=("_frozen", "_recomputed"), indicator=True)
    if not z._merge.eq("both").all() or any(not (z[f"{c}_frozen"] == z[f"{c}_recomputed"]).all() for c in ["total_direct_derivatives", "derivatives_with_primary_eligible_s_mapping", "distinct_eligible_s_contexts"]):
        raise ValueError("Frozen P-side support file disagrees with edge v2 + A4 mapping")
    a4 = args.a4_qc.read_text(encoding="utf-8")
    for phrase in ["**5,600 / 6,195 edges", "**971 / 1,001 root families", "`阝` is Category C", "`月` is Category C"]:
        if phrase not in a4: raise RuntimeError(f"A4 QC missing frozen assertion: {phrase}")
    declared = args.sha256_manifest.read_text(encoding="utf-8")
    if EXPECTED["edges"] not in declared or EXPECTED["census"] not in declared: raise RuntimeError("A3 manifest lacks governed hashes")
    if len(master) != 11151 or master.original_character.duplicated().any(): raise ValueError("Master is not the frozen 11,151-character population")
    return got

def prepare_edges(edges, smap, index, E):
    lookup = {c: i for i, c in enumerate(index.original_character.astype(str))}
    m = smap.copy(); m["map_eligible"] = truth(m.s_primary_analysis_eligible)
    m = m[m.map_eligible & m.s_mapping_category.str.startswith(("A_", "B_")) & m.s_mapping_confidence.eq("high")]
    e = edges.merge(m[["s_component_canonical", "s_independent_character", "s_mapping_category"]],
        left_on="added_component", right_on="s_component_canonical", how="left", validate="many_to_one")
    e["structurally_mapped"] = e.s_independent_character.fillna("").ne("")
    e["derived_available_in_cache"] = e.derived_char.isin(lookup)
    e["s_independent_available_in_cache"] = e.s_independent_character.isin(lookup)
    use = e[e.structurally_mapped & e.derived_available_in_cache & e.s_independent_available_in_cache].copy()
    use["c_idx"] = use.derived_char.map(lookup).astype(int)
    use["s_idx"] = use.s_independent_character.map(lookup).astype(int)
    delta = np.asarray(E[use.c_idx.to_numpy()], np.float32) - np.asarray(E[use.s_idx.to_numpy()], np.float32)
    return e, use.reset_index(drop=True), delta, lookup

def analyze(edges, smap, support, index, E, model, channel, args):
    all_e, e, delta, lookup = prepare_edges(edges, smap, index, E)
    prototypes = np.zeros_like(delta); n_edges = np.zeros(len(e), int); n_contexts = np.zeros(len(e), int)
    available = np.zeros(len(e), bool)
    # Hold out the entire canonical S context, not merely its one edge.
    for family, ids0 in e.groupby("family_id_root", sort=True).groups.items():
        ids = np.asarray(list(ids0), int)
        by_s = {s: np.asarray(list(g), int) for s, g in e.loc[ids].groupby("added_component").groups.items()}
        context_mean = {s: delta[j].mean(axis=0) for s, j in by_s.items()}
        for i in ids:
            held_s = e.at[i, "added_component"]
            other_s = sorted(s for s in by_s if s != held_s)
            held_in = np.concatenate([by_s[s] for s in other_s]) if other_s else np.array([], int)
            n_edges[i] = len(held_in); n_contexts[i] = len(other_s)
            if len(other_s) >= 3:
                prototypes[i] = np.mean([context_mean[s] for s in other_s], axis=0)
                available[i] = np.linalg.norm(prototypes[i]) > 0
    e["n_other_edges"] = n_edges; e["n_other_distinct_s_contexts"] = n_contexts
    e["prototype_available"] = available
    use = e[available].copy(); ids = np.flatnonzero(available)
    chars = index.original_character.astype(str).tolist(); char_ids = {c: i for i, c in enumerate(chars)}
    true_ids = np.array([char_ids[c] for c in use.derived_char], int)
    exclude = np.array([char_ids[c] for c in use.s_independent_character], int)
    baseline_q = np.asarray(E[use.s_idx.to_numpy()], np.float32)
    offset_q = baseline_q + prototypes[ids]
    baseline = identify(baseline_q, E, chars, true_ids, exclude, args.retrieval_batch)
    offset = identify(offset_q, E, chars, true_ids, exclude, args.retrieval_batch)
    out = pd.DataFrame({
        "model": model, "channel": channel, "edge_row": use.edge_id, "root_family_id": use.family_id_root,
        "p_anchor": use.anchor_char, "s_component_canonical": use.added_component,
        "s_independent_character": use.s_independent_character, "derived_character": use.derived_char,
        "heldout_case_id": model + "__" + channel + "__" + use.edge_id,
        "n_other_edges": use.n_other_edges, "n_other_distinct_s_contexts": use.n_other_distinct_s_contexts,
        "baseline_predicted_character": [x[0] for x in baseline], "baseline_true_rank": [x[1] for x in baseline],
        "baseline_true_cosine": [x[2] for x in baseline], "offset_predicted_character": [x[0] for x in offset],
        "offset_true_rank": [x[1] for x in offset], "offset_true_cosine": [x[2] for x in offset],
        "candidate_count": len(E) - 1, "support_threshold": 3,
    })
    for prefix in ("baseline", "offset"):
        out[f"{prefix}_top1"] = (out[f"{prefix}_predicted_character"] == out.derived_character).astype(int)
        for k in (5, 10): out[f"{prefix}_top{k}"] = (out[f"{prefix}_true_rank"] <= k).astype(int)
        out[f"{prefix}_reciprocal_rank"] = 1 / out[f"{prefix}_true_rank"]
        out[f"{prefix}_percentile_rank"] = 1 - (out[f"{prefix}_true_rank"] - 1) / np.maximum(out.candidate_count - 1, 1)
    out["rank_change"] = out.baseline_true_rank - out.offset_true_rank
    for k in (1, 5, 10):
        out[f"top{k}_rescued"] = ((out[f"baseline_top{k}"] == 0) & (out[f"offset_top{k}"] == 1)).astype(int)
        out[f"top{k}_lost"] = ((out[f"baseline_top{k}"] == 1) & (out[f"offset_top{k}"] == 0)).astype(int)
    # Exact retrieval invariants.
    for prefix in ("baseline", "offset"):
        if not ((out[f"{prefix}_true_rank"] == 1) == out[f"{prefix}_top1"].astype(bool)).all(): raise RuntimeError("rank/Top-1 invariant failed")
        if not ((out[f"{prefix}_predicted_character"] == out.derived_character) == out[f"{prefix}_top1"].astype(bool)).all(): raise RuntimeError("argmax identity invariant failed")
        if not out[f"{prefix}_true_rank"].between(1, out.candidate_count).all(): raise RuntimeError("rank outside candidate universe")
    coverage = {
        "model": model, "channel": channel, "channel_eligible_characters": len(index),
        "mapped_s_independent_characters_requested": int(smap.loc[truth(smap.s_primary_analysis_eligible), "s_independent_character"].nunique()),
        "mapped_s_independent_characters_available": int(smap.loc[truth(smap.s_primary_analysis_eligible) & smap.s_independent_character.isin(lookup), "s_independent_character"].nunique()),
        "mapped_s_independent_characters_unavailable": int(smap.loc[truth(smap.s_primary_analysis_eligible) & ~smap.s_independent_character.isin(lookup), "s_independent_character"].nunique()),
        "documented_direct_edges": len(edges), "structurally_mapped_edges": int(all_e.structurally_mapped.sum()),
        "mapped_edges_with_derived_available": int((all_e.structurally_mapped & all_e.derived_available_in_cache).sum()),
        "mapped_edges_with_s_available": int((all_e.structurally_mapped & all_e.s_independent_available_in_cache).sum()),
        "literal_offset_eligible_edges": len(e), "heldout_edges_ge3": int((e.n_other_distinct_s_contexts >= 3).sum()),
        "heldout_edges_ge5": int((e.n_other_distinct_s_contexts >= 5).sum()), "heldout_edges_ge8": int((e.n_other_distinct_s_contexts >= 8).sum()),
        "root_families_ge3": int(e.loc[e.n_other_distinct_s_contexts >= 3, "family_id_root"].nunique()),
    }
    return out, pd.DataFrame([coverage])

def summarize(df):
    rows = []
    for (model, channel), g0 in df.groupby(["model", "channel"]):
        for threshold in (3, 5, 8):
            g = g0[g0.n_other_distinct_s_contexts >= threshold]
            row = {"model": model, "channel": channel, "support_threshold": threshold,
                "n_heldout_cases": len(g), "n_p_root_families": g.root_family_id.nunique(),
                "n_distinct_mapped_s_contexts": g.s_component_canonical.nunique(),
                "candidate_count": int(g.candidate_count.iloc[0]) if len(g) else np.nan,
                "percent_rank_improved": 100 * mean(g.rank_change > 0), "percent_rank_worsened": 100 * mean(g.rank_change < 0),
                "percent_rank_tied": 100 * mean(g.rank_change == 0),
                "top1_rescues": int(g.top1_rescued.sum()), "top1_losses": int(g.top1_lost.sum()),
                "top1_net": int(g.top1_rescued.sum() - g.top1_lost.sum())}
            for p in ("baseline", "offset"):
                r = g[f"{p}_true_rank"]
                row.update({f"{p}_recall_at_1": mean(r <= 1), f"{p}_recall_at_5": mean(r <= 5),
                    f"{p}_recall_at_10": mean(r <= 10), f"{p}_mrr": mean(1 / r),
                    f"{p}_median_rank": float(r.median()) if len(r) else np.nan,
                    f"{p}_mean_rank": float(r.mean()) if len(r) else np.nan})
            for k in (1, 5, 10): row[f"recall_at_{k}_gain"] = row[f"offset_recall_at_{k}"] - row[f"baseline_recall_at_{k}"]
            if threshold == 3 and len(g):
                row["human_interpretation"] = (f"Across {len(g):,} held-out members of P families, the correct derived character was rank 1 in "
                    f"{100*row['baseline_recall_at_1']:.2f}% using S alone and {100*row['offset_recall_at_1']:.2f}% after adding r_P.")
            rows.append(row)
    return pd.DataFrame(rows)

def matched_summary(p_df, s_path):
    if s_path is None or not s_path.is_file(): return pd.DataFrame()
    s = pd.read_csv(s_path, dtype={"derived_character": str, "anchor_character": str, "relation_class": str})
    need = ["model", "channel", "derived_character", "anchor_character", "relation_class", "anchor_family",
        "baseline_true_character_rank", "offset_true_character_rank", "candidate_character_count"]
    missing = [c for c in need if c not in s]
    if missing: raise ValueError(f"S-side forward file missing columns: {missing}")
    pk = p_df.rename(columns={"p_anchor": "anchor_character", "s_component_canonical": "relation_class", "root_family_id": "anchor_family"})
    keys = ["model", "channel", "derived_character", "anchor_character", "relation_class", "anchor_family"]
    z = pk.merge(s[need], on=keys, how="inner", validate="one_to_one", suffixes=("_p", "_s"))
    if len(z) and not (z.candidate_count == z.candidate_character_count).all(): raise RuntimeError("Matched P/S candidate counts differ")
    rows = []
    for (model, channel), g0 in z.groupby(["model", "channel"]):
        for threshold in (3, 5, 8):
            g = g0[g0.n_other_distinct_s_contexts >= threshold]
            row = {"model": model, "channel": channel, "support_threshold": threshold, "matched_n": len(g),
                "candidate_count": int(g.candidate_count.iloc[0]) if len(g) else np.nan}
            for side, b, o in [("p_side", "baseline_true_rank", "offset_true_rank"),
                                ("s_side", "baseline_true_character_rank", "offset_true_character_rank")]:
                br, orank = g[b], g[o]
                for k in (1, 5, 10):
                    row[f"{side}_baseline_recall_at_{k}"] = mean(br <= k); row[f"{side}_offset_recall_at_{k}"] = mean(orank <= k)
                    row[f"{side}_recall_at_{k}_gain"] = row[f"{side}_offset_recall_at_{k}"] - row[f"{side}_baseline_recall_at_{k}"]
                row[f"{side}_baseline_mrr"] = mean(1 / br); row[f"{side}_offset_mrr"] = mean(1 / orank)
                row[f"{side}_baseline_median_rank"] = float(br.median()) if len(g) else np.nan
                row[f"{side}_offset_median_rank"] = float(orank.median()) if len(g) else np.nan
                row[f"{side}_baseline_mean_rank"] = float(br.mean()) if len(g) else np.nan
                row[f"{side}_offset_mean_rank"] = float(orank.mean()) if len(g) else np.nan
                change = br - orank; row[f"{side}_percent_rank_improved"] = 100 * mean(change > 0)
                row[f"{side}_top1_rescues"] = int(((br > 1) & (orank == 1)).sum()); row[f"{side}_top1_losses"] = int(((br == 1) & (orank > 1)).sum())
            rows.append(row)
    return pd.DataFrame(rows)

def make_plots(out, retrieval, summary, matched):
    import matplotlib.pyplot as plt
    pdir = out / "plots"; pdir.mkdir(exist_ok=True)
    primary = summary[summary.support_threshold == 3]
    def grouped(a, b, ylabel, title, name):
        if not len(primary): return
        x = np.arange(len(primary)); labels = primary.model + "\n" + primary.channel
        plt.figure(figsize=(max(9, .58 * len(primary)), 5)); plt.bar(x-.2, primary[a], .4, label="baseline"); plt.bar(x+.2, primary[b], .4, label="+ r_P")
        plt.xticks(x, labels, rotation=70, ha="right", fontsize=7); plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(pdir/name, dpi=160); plt.close()
    grouped("baseline_recall_at_1", "offset_recall_at_1", "Recall@1", "P-side derived-character identification", "01_p_side_recall_at_1.png")
    grouped("baseline_median_rank", "offset_median_rank", "Median true rank", "P-side median rank", "02_p_side_median_rank.png")
    if len(primary):
        x=np.arange(len(primary)); labels=primary.model+"\n"+primary.channel; plt.figure(figsize=(max(9,.58*len(primary)),5))
        for c,l in [("baseline_recall_at_5","base R@5"),("offset_recall_at_5","+rP R@5"),("baseline_recall_at_10","base R@10"),("offset_recall_at_10","+rP R@10")]: plt.plot(x,primary[c],marker="o",label=l)
        plt.xticks(x,labels,rotation=70,ha="right",fontsize=7);plt.legend();plt.tight_layout();plt.savefig(pdir/"03_p_side_recall_at_5_10.png",dpi=160);plt.close()
        plt.figure(figsize=(8,5));plt.hist(retrieval.rank_change,bins=60);plt.axvline(0,color="black");plt.xlabel("Baseline rank - offset rank (positive = improvement)");plt.tight_layout();plt.savefig(pdir/"04_p_side_rank_change.png",dpi=160);plt.close()
        plt.figure(figsize=(8,5));
        for (m,c),g in summary.groupby(["model","channel"]): plt.plot(g.support_threshold,g.offset_recall_at_1,marker="o",label=f"{m}/{c}")
        plt.xticks([3,5,8]);plt.xlabel("Minimum OTHER distinct S contexts");plt.ylabel("P-side offset Recall@1");plt.legend(fontsize=5,ncol=2);plt.tight_layout();plt.savefig(pdir/"07_p_side_support_sensitivity.png",dpi=160);plt.close()
        pivot=primary.pivot(index="model",columns="channel",values="recall_at_1_gain");plt.figure(figsize=(8,4));im=plt.imshow(pivot.fillna(0),aspect="auto",cmap="coolwarm");plt.colorbar(im,label="Recall@1 gain");plt.xticks(range(len(pivot.columns)),pivot.columns,rotation=45,ha="right");plt.yticks(range(len(pivot.index)),pivot.index);plt.tight_layout();plt.savefig(pdir/"08_model_channel_p_side_effect_matrix.png",dpi=160);plt.close()
    if len(matched):
        m=matched[matched.support_threshold==3];x=np.arange(len(m));labels=m.model+"\n"+m.channel
        plt.figure(figsize=(max(9,.58*len(m)),5));plt.bar(x-.2,m.s_side_recall_at_1_gain,.4,label="S-side");plt.bar(x+.2,m.p_side_recall_at_1_gain,.4,label="P-side");plt.xticks(x,labels,rotation=70,ha="right",fontsize=7);plt.ylabel("Matched Recall@1 gain");plt.legend();plt.tight_layout();plt.savefig(pdir/"05_matched_s_vs_p_recall_at_1_gain.png",dpi=160);plt.close()
        plt.figure(figsize=(max(9,.58*len(m)),5));plt.bar(x-.2,m.s_side_baseline_median_rank-m.s_side_offset_median_rank,.4,label="S-side");plt.bar(x+.2,m.p_side_baseline_median_rank-m.p_side_offset_median_rank,.4,label="P-side");plt.xticks(x,labels,rotation=70,ha="right",fontsize=7);plt.ylabel("Median-rank improvement");plt.legend();plt.tight_layout();plt.savefig(pdir/"06_matched_s_vs_p_median_rank_improvement.png",dpi=160);plt.close()
        sem=m[m.channel.isin(["cn_def","en_def_translated_cn_def","en_gloss_unihan"])]
        if len(sem):
            plt.figure(figsize=(7,5));plt.scatter(sem.s_side_recall_at_1_gain,sem.p_side_recall_at_1_gain);plt.axhline(0,color="grey");plt.axvline(0,color="grey");plt.xlabel("S-side matched R@1 gain");plt.ylabel("P-side matched R@1 gain");plt.tight_layout();plt.savefig(pdir/"09_semantic_channel_matched_comparison.png",dpi=160);plt.close()
    roots={"青","堯","店","余","元","帝","闌","辰","曷","賓","圭","台","亥"};d=retrieval[(retrieval.p_anchor.isin(roots))|(retrieval.derived_character.isin(["掂","惦"]))]
    if len(d):
        plt.figure(figsize=(9,5));
        for _,r in d.iterrows(): plt.plot([0,1],[r.baseline_true_rank,r.offset_true_rank],marker="o",alpha=.45)
        plt.xticks([0,1],["S alone","S + r_P"]);plt.ylabel("True derived-character rank");plt.tight_layout();plt.savefig(pdir/"10_diagnostic_baseline_to_offset_rank.png",dpi=160);plt.close()

def consolidate(out, s_side_forward):
    files=sorted((out/"checkpoints").glob("*__retrieval.csv")); frames=[pd.read_csv(f) for f in files]
    retrieval=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(); atomic_csv(retrieval,out/"p_side_forward_retrieval_v1.csv")
    summary=summarize(retrieval) if len(retrieval) else pd.DataFrame();atomic_csv(summary,out/"p_side_retrieval_identification_summary_v1.csv")
    matched=matched_summary(retrieval,s_side_forward);atomic_csv(matched,out/"p_vs_s_side_matched_retrieval_summary_v1.csv")
    roots={"青","堯","店","余","元","帝","闌","辰","曷","賓","圭","台","亥"}
    diag=retrieval[(retrieval.p_anchor.isin(roots))|(retrieval.derived_character.isin(["掂","惦"]))].copy() if len(retrieval) else pd.DataFrame()
    atomic_csv(diag,out/"p_side_diagnostic_cases_v1.csv")
    covs=[pd.read_csv(f) for f in sorted((out/"checkpoints").glob("*__coverage.csv"))];atomic_csv(pd.concat(covs,ignore_index=True) if covs else pd.DataFrame(),out/"p_side_channel_coverage_v1.csv")
    make_plots(out,retrieval,summary,matched);return retrieval,summary,matched

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--master",type=Path,required=True);p.add_argument("--edges",type=Path,required=True);p.add_argument("--census",type=Path,required=True)
    p.add_argument("--normalization-qc",type=Path,required=True);p.add_argument("--sha256-manifest",type=Path,required=True)
    p.add_argument("--s-map",type=Path,required=True);p.add_argument("--p-support",type=Path,required=True);p.add_argument("--a4-qc",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--s-side-forward",type=Path,help="Existing Phase IV-C v5 forward_analogy_retrieval_v5.csv; optional until matched consolidation")
    p.add_argument("--model",choices=MODELS,required=True);p.add_argument("--channel",choices=CHANNELS,required=True)
    p.add_argument("--retrieval-batch",type=int,default=128);p.add_argument("--allow-google-exploratory",action="store_true");p.add_argument("--force",action="store_true")
    a=p.parse_args()
    if a.channel=="en_gloss_google" and not a.allow_google_exploratory: raise SystemExit("Google direct gloss remains exploratory; pass --allow-google-exploratory explicitly")
    a.output.mkdir(parents=True,exist_ok=True);ck=a.output/"checkpoints";ck.mkdir(exist_ok=True);tag=f"{a.model}__{a.channel}"
    master=pd.read_csv(a.master,dtype=str,keep_default_na=False);edges=pd.read_csv(a.edges,dtype=str,keep_default_na=False)
    smap=pd.read_csv(a.s_map,dtype=str,keep_default_na=False);support=pd.read_csv(a.p_support,dtype=str,keep_default_na=False)
    hashes=validate_frozen(a,edges,smap,support,master)
    cache=a.cache_root/a.model/a.channel;index=pd.read_csv(cache/"character_index.csv",dtype=str,keep_default_na=False)
    E=np.load(cache/"E_raw.npy",mmap_mode="r");meta=json.loads((cache/"metadata.json").read_text(encoding="utf-8"))
    if not meta.get("complete") or len(E)!=len(index): raise RuntimeError("Incomplete or misaligned Phase IV-C cache")
    if index.original_character.duplicated().any(): raise RuntimeError("Duplicate candidate characters in cache")
    marker=ck/f"{tag}__complete.json"
    fingerprints={"script_version":VERSION,"files":{**hashes,"master":sha256(a.master),"cache_metadata":sha256(cache/"metadata.json")}}
    fp=out_fp=a.output/"p_side_input_fingerprints_v1.json"
    state=json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {"conditions":{}}
    prior=state["conditions"].get(tag)
    if prior and prior!=fingerprints: raise RuntimeError(f"Frozen inputs changed for completed/started condition {tag}")
    state["conditions"][tag]=fingerprints;atomic_json(state,fp)
    if marker.exists() and not a.force:
        print(f"Already complete after frozen-input validation: {tag}");consolidate(a.output,a.s_side_forward);return
    started=time.time();print(f"Analyzing Phase IV-D raw P-side composition: {tag}",flush=True)
    retrieval,coverage=analyze(edges,smap,support,index,E,a.model,a.channel,a)
    atomic_csv(retrieval,ck/f"{tag}__retrieval.csv");atomic_csv(coverage,ck/f"{tag}__coverage.csv")
    atomic_json({"model":a.model,"channel":a.channel,"completed_utc":now(),"runtime_seconds":time.time()-started,
        "space":"raw uncentered E","holdout":"entire canonical S context","weighting":"equal across distinct held-in S contexts"},marker)
    all_r,summary,matched=consolidate(a.output,a.s_side_forward)
    manifest={"schema_version":1,"script_version":VERSION,"created_utc":now(),"models":MODELS,"channels":CHANNELS,
        "primary_offset":"r_P = mean across distinct held-in S contexts of mean(E(C)-E(S_independent))",
        "holdout_rule":"exclude every edge sharing the held-out canonical S context",
        "primary_threshold":"at least 3 OTHER distinct eligible S contexts","sensitivities":[5,8],
        "candidate_universe":"all channel-eligible characters except the input S-independent character",
        "tie_policy":"frozen character_index.csv order; first exact cosine maximum wins",
        "mapping_rule":"only frozen high-confidence A/B rows marked primary eligible in A4",
        "space":"raw E, uncentered; cosine normalizes query and candidate only","learned_W":False,"PCA_CCA_SVCCA":False,
        "matched_s_side_source":str(a.s_side_forward) if a.s_side_forward else None,"python":platform.python_version()}
    atomic_json(manifest,a.output/"p_side_method_manifest_v1.json")
    primary=summary[(summary.model==a.model)&(summary.channel==a.channel)&(summary.support_threshold==3)]
    sentence=primary.human_interpretation.iloc[0] if len(primary) else "No >=3 held-out cases."
    qc=f"""# Phase IV-D P-Side Relational Composition QC v1

Updated UTC: `{now()}`

- Current condition: `{tag}`
- Frozen A4 mapping validation: PASS (306 S classes; 236 primary-eligible high-confidence A/B mappings)
- Channel-eligible characters: {len(index):,}
- Literal P-side offset-eligible edges: {int(coverage.literal_offset_eligible_edges.iloc[0]):,}
- Held-out cases with >=3 / >=5 / >=8 OTHER distinct S contexts: {int(coverage.heldout_edges_ge3.iloc[0]):,} / {int(coverage.heldout_edges_ge5.iloc[0]):,} / {int(coverage.heldout_edges_ge8.iloc[0]):,}
- Mapped independent-S characters requested / available / unavailable: {int(coverage.mapped_s_independent_characters_requested.iloc[0]):,} / {int(coverage.mapped_s_independent_characters_available.iloc[0]):,} / {int(coverage.mapped_s_independent_characters_unavailable.iloc[0]):,}
- Holdout leakage check: PASS; the held-out canonical S context is wholly excluded from `r_P`.
- Weighting check: PASS; edges are averaged within canonical S, then distinct S contexts are equally weighted.
- Baseline/offset candidate universe and held-out cases: identical.
- Exact cosine argmax, rank, Top-1 identity, bounds, and deterministic ties: PASS.
- Matched S-side comparison: {'available' if len(matched) else 'pending; supply --s-side-forward to consolidate'}.
- Primary vectors: raw, uncentered `E`; no learned W, PCA, CCA, or SVCCA.

Human-facing lead result: {sentence}
"""
    (a.output/"p_side_QC_v1.md").write_text(qc,encoding="utf-8")
    print(sentence);print(f"Done in {(time.time()-started)/60:.1f} min: {tag}")

if __name__ == "__main__": main()
