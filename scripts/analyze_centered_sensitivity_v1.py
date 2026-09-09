from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path("centered_validation_v1")
RAW_ROOT = Path("phase4_4x4_complete")
OUT = Path("centered_validation_analysis_v1")
OUT.mkdir(exist_ok=True)


def gain(df, prefix, mode, k):
    direct = f"{prefix}recall_at_{k}_gain_{mode}"
    if direct in df.columns:
        return df[direct]
    return df[f"{prefix}offset_recall_at_{k}_{mode}"] - df[f"{prefix}baseline_recall_at_{k}_{mode}"]


def median_gain(df, prefix, mode):
    return df[f"{prefix}baseline_median_rank_{mode}"] - df[f"{prefix}offset_median_rank_{mode}"]


def cell_summary():
    c = pd.read_csv(ROOT / "raw_vs_centered_C2_summary_v1.csv")
    c = c[c.minimum_other_training_families.eq(3)].copy()
    d = pd.read_csv(ROOT / "raw_vs_centered_D_summary_v1.csv")
    d = d[d.support_threshold.eq(3)].copy()

    raw_forward = pd.read_csv(RAW_ROOT / "Phase_IV_C_RawOffset_E_v5" / "forward_analogy_retrieval_v5.csv")
    cen_forward = pd.read_csv(ROOT / "Centered_Sensitivity_C2_v1" / "forward_analogy_retrieval_v5.csv")
    cen_forward["channel"] = cen_forward.channel.str.replace("__centered_sensitivity", "", regex=False)
    raw_forward = raw_forward[raw_forward.training_family_count.ge(3)]
    cen_forward = cen_forward[cen_forward.training_family_count.ge(3)]
    raw_forward_pos = raw_forward.groupby(["model", "channel"]).rank_change.apply(lambda s: 100 * (s > 0).mean())
    cen_forward_pos = cen_forward.groupby(["model", "channel"]).rank_change.apply(lambda s: 100 * (s > 0).mean())

    rows = []
    for side, frame, prefix in [("S_forward", c, "forward_"), ("S_inverse", c, "inverse_"), ("P_forward", d, "")]:
        for _, x in frame.iterrows():
            row = {"side": side, "model": x.model, "channel": x.channel}
            for mode in ["raw", "centered"]:
                if side == "S_forward":
                    lookup = raw_forward_pos if mode == "raw" else cen_forward_pos
                    row[f"rank_improved_pct_{mode}"] = lookup.loc[(x.model, x.channel)]
                elif side == "S_inverse":
                    row[f"rank_improved_pct_{mode}"] = x[f"inverse_percent_rank_improved_{mode}"]
                else:
                    row[f"rank_improved_pct_{mode}"] = x[f"percent_rank_improved_{mode}"]
                row[f"median_rank_gain_{mode}"] = (
                    x[f"{prefix}baseline_median_rank_{mode}"] - x[f"{prefix}offset_median_rank_{mode}"]
                    if prefix else x[f"baseline_median_rank_{mode}"] - x[f"offset_median_rank_{mode}"]
                )
                for k in [1, 5, 10]:
                    row[f"r{k}_gain_{mode}"] = gain(pd.DataFrame([x]), prefix, mode, k).iloc[0]
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "cell_level_raw_vs_centered_primary.csv", index=False)
    return out


def family_summary():
    outputs = []
    for side, fn, famcol in [
        ("S", "raw_vs_centered_S_family_effects_v1.csv", "anchor_family"),
        ("P", "raw_vs_centered_P_family_effects_v1.csv", "root_family_id"),
    ]:
        d = pd.read_csv(ROOT / fn)
        for (model, channel), x in d.groupby(["model", "channel"]):
            raw = x.rank_change_raw.to_numpy(float)
            cen = x.rank_change_centered.to_numpy(float)
            nonzero = (raw != 0) & (cen != 0)
            outputs.append({
                "side": side,
                "model": model,
                "channel": channel,
                "family_n": len(x),
                "raw_mean": raw.mean(),
                "centered_mean": cen.mean(),
                "raw_median": np.median(raw),
                "centered_median": np.median(cen),
                "raw_positive_share": (raw > 0).mean(),
                "centered_positive_share": (cen > 0).mean(),
                "sign_concordance_nonzero": (np.sign(raw[nonzero]) == np.sign(cen[nonzero])).mean(),
                "pearson_r": pearsonr(raw, cen).statistic,
                "spearman_rho": spearmanr(raw, cen).statistic,
            })
    out = pd.DataFrame(outputs)
    out.to_csv(OUT / "family_level_raw_vs_centered_concordance.csv", index=False)
    return out


def matched_summary():
    raw = pd.read_csv(RAW_ROOT / "Phase_IV_D_P_Side_4x4_v1" / "p_vs_s_side_matched_retrieval_summary_v1.csv")
    cen = pd.read_csv(ROOT / "Centered_Sensitivity_D_v1" / "p_vs_s_side_matched_retrieval_summary_v1.csv")
    cen["channel"] = cen.channel.str.replace("__centered_sensitivity", "", regex=False)
    raw = raw[raw.support_threshold.eq(3)].copy()
    cen = cen[cen.support_threshold.eq(3)].copy()
    keys = ["model", "channel", "support_threshold"]
    m = raw.merge(cen, on=keys, suffixes=("_raw", "_centered"), validate="one_to_one")
    rows = []
    for _, x in m.iterrows():
        row = {k: x[k] for k in keys}
        row["matched_n_raw"] = x.matched_n_raw
        row["matched_n_centered"] = x.matched_n_centered
        for side in ["s", "p"]:
            for mode in ["raw", "centered"]:
                suffix = f"_{mode}"
                row[f"{side}_rank_improved_pct_{mode}"] = x[f"{side}_side_percent_rank_improved{suffix}"]
                row[f"{side}_median_rank_gain_{mode}"] = x[f"{side}_side_baseline_median_rank{suffix}"] - x[f"{side}_side_offset_median_rank{suffix}"]
                for k in [1, 5, 10]:
                    row[f"{side}_r{k}_gain_{mode}"] = x[f"{side}_side_recall_at_{k}_gain{suffix}"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "matched_s_vs_p_raw_vs_centered_primary.csv", index=False)
    return out


def compact_stats(cells, fam, matched):
    lines = []
    for side in ["S_forward", "S_inverse", "P_forward"]:
        x = cells[cells.side.eq(side)]
        lines.append({
            "scope": side,
            "cells": len(x),
            "raw_rank_majority_positive": int((x.rank_improved_pct_raw > 50).sum()),
            "centered_rank_majority_positive": int((x.rank_improved_pct_centered > 50).sum()),
            "raw_median_gain_positive": int((x.median_rank_gain_raw > 0).sum()),
            "centered_median_gain_positive": int((x.median_rank_gain_centered > 0).sum()),
            "raw_r10_gain_positive": int((x.r10_gain_raw > 0).sum()),
            "centered_r10_gain_positive": int((x.r10_gain_centered > 0).sum()),
            "raw_rank_improved_pct_mean": x.rank_improved_pct_raw.mean(),
            "centered_rank_improved_pct_mean": x.rank_improved_pct_centered.mean(),
            "raw_median_gain_median": x.median_rank_gain_raw.median(),
            "centered_median_gain_median": x.median_rank_gain_centered.median(),
            "raw_r10_gain_mean": x.r10_gain_raw.mean(),
            "centered_r10_gain_mean": x.r10_gain_centered.mean(),
        })
    pd.DataFrame(lines).to_csv(OUT / "overall_cell_robustness_summary.csv", index=False)

    # Aggregate matched effects across models by channel, preserving equal model weight.
    agg = []
    for channel, x in matched.groupby("channel"):
        row = {"channel": channel, "model_n": len(x)}
        for side in ["s", "p"]:
            for mode in ["raw", "centered"]:
                for metric in ["rank_improved_pct", "median_rank_gain", "r1_gain", "r5_gain", "r10_gain"]:
                    row[f"{side}_{metric}_{mode}_mean"] = x[f"{side}_{metric}_{mode}"].mean()
        agg.append(row)
    pd.DataFrame(agg).to_csv(OUT / "matched_s_vs_p_channel_means.csv", index=False)

    # Concise family aggregate by side and semantic/nonsemantic scope.
    fam2 = fam.assign(scope=np.where(fam.channel.eq("zh_char"), "zh_char", "semantic"))
    agg2 = fam2.groupby(["side", "scope"]).agg(
        cells=("model", "size"),
        families=("family_n", "sum"),
        raw_positive_share_mean=("raw_positive_share", "mean"),
        centered_positive_share_mean=("centered_positive_share", "mean"),
        sign_concordance_mean=("sign_concordance_nonzero", "mean"),
        spearman_mean=("spearman_rho", "mean"),
        raw_median_mean=("raw_median", "mean"),
        centered_median_mean=("centered_median", "mean"),
    ).reset_index()
    agg2.to_csv(OUT / "family_concordance_scope_summary.csv", index=False)


if __name__ == "__main__":
    cells = cell_summary()
    fam = family_summary()
    matched = matched_summary()
    compact_stats(cells, fam, matched)
    print("Wrote analysis tables to", OUT)
