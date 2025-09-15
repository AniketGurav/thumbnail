#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
insight.py — Analyze thumbnail patterns from saved preds + joint_clean.csv
(hard-coded paths; robust merge; no CLI args needed)

Outputs per target (if preds file exists):
- feature_comparison_<target>.csv   : high vs low means, delta, Cohen's d, corr
- quadrant_share_faces_<target>.csv : average face quadrant shares by bucket
- quadrant_share_text_<target>.csv  : average text quadrant shares by bucket
- threshold_text_area_<target>.csv  : best cutoff for text_area_pct
"""

import os
import numpy as np
import pandas as pd

# ----------------- HARDCODED PATHS ----------------- #
BASE = "/cluster/datastore/aniketag/urumi/metadata"
DATA_PATH = os.path.join(BASE, "data", "joint_clean.csv")
PREDS_DIR = os.path.join(BASE, "data", "40_2")
PREDS_FILES = {
    "raw_views": os.path.join(PREDS_DIR, "preds_main3_raw_views_linear_40_2.csv"),
    "views_per_sub": os.path.join(PREDS_DIR, "preds_main3_views_per_sub_linear_40_2.csv"),
    "views_per_sub_per_day": os.path.join(PREDS_DIR, "preds_main3_views_per_sub_per_day_linear_40_2.csv"),
    "log_views": os.path.join(PREDS_DIR, "preds_main3_log_views_linear_40_2.csv"),
}
# --------------------------------------------------- #

FACE_FEATURES = [
    "faces_n","faces_area_mean_pct","faces_area_max_pct",
    "faces_center_std_norm","faces_edge_near_frac",
    "faces_q1_count","faces_q2_count","faces_q3_count","faces_q4_count",
]
TEXT_FEATURES = [
    "text_boxes_n","text_area_pct","text_area_mean_pct","text_area_median_pct",
    "text_horiz_coverage_pct","text_vert_coverage_pct",
    "text_q1_count","text_q2_count","text_q3_count","text_q4_count",
]
TITLE_FEATURES = [
    "title_length_chars","title_length_words","title_upper_ratio",
    "title_exclam_count","title_question_count",
]
META_FEATURES = [
    "days_since_upload","channel_follower_count"
]

ALL_GROUPS = [
    ("faces", FACE_FEATURES),
    ("text", TEXT_FEATURES),
    ("title", TITLE_FEATURES),
    ("meta", META_FEATURES),
]

# ----------------- helpers ----------------- #
def cohen_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = np.mean(a), np.mean(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    n1, n2 = len(a), len(b)
    sp2 = ((n1-1)*sa*sa + (n2-1)*sb*sb) / max(1, (n1+n2-2))
    sp = np.sqrt(max(sp2, 1e-12))
    return float((ma - mb) / sp)

def safe_corr(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def make_buckets(y):
    y = np.asarray(y)
    q25, q75 = np.quantile(y, 0.25), np.quantile(y, 0.75)
    return np.where(y >= q75, "high", np.where(y <= q25, "low", "mid"))

def summarize_group(df, cols, y_col, bucket_col):
    hi, lo = df[df[bucket_col] == "high"], df[df[bucket_col] == "low"]
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        mean_hi, mean_lo = float(hi[c].mean()), float(lo[c].mean())
        d = cohen_d(hi[c].values, lo[c].values)
        corr = safe_corr(df[c].values, df[y_col].values)
        rows.append({
            "feature": c,
            "mean_high": mean_hi,
            "mean_low": mean_lo,
            "delta_mean": mean_hi - mean_lo,
            "cohen_d": d,
            "corr": corr
        })
    return pd.DataFrame(rows)

def quadrant_shares(df, prefix, bucket_col):
    qcols = [f"{prefix}_q{i}_count" for i in range(1, 5)]
    qcols = [c for c in qcols if c in df.columns]
    if not qcols:
        return pd.DataFrame()
    qsum = df[qcols].sum(axis=1).replace(0, 1)
    norm = df[qcols].div(qsum, axis=0)
    tmp = pd.concat([df[[bucket_col]].reset_index(drop=True),
                     norm.reset_index(drop=True)], axis=1)
    return tmp.groupby(bucket_col)[qcols].mean().reset_index()

def threshold_sweep(series, y):
    x = pd.to_numeric(series, errors="coerce").fillna(0.0).values
    y = np.asarray(y)
    if len(x) < 3:
        return pd.DataFrame()
    ps = np.linspace(0.05, 0.95, 19)
    grid = np.unique(np.quantile(x, ps)).tolist()
    rows = []
    for t in grid:
        left = y[x <= t]
        right = y[x > t]
        if len(left) < 10 or len(right) < 10:
            continue
        rows.append({
            "threshold": float(t),
            "n_left": int(len(left)),
            "n_right": int(len(right)),
            "mean_left": float(np.mean(left)),
            "mean_right": float(np.mean(right)),
            "delta_mean": float(np.mean(right) - np.mean(left))
        })
    df_thr = pd.DataFrame(rows)
    if not df_thr.empty:
        df_thr = df_thr.sort_values("delta_mean", ascending=False).reset_index(drop=True)
    return df_thr

def detect_and_rename_pred_index(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure preds has a column named 'index' that maps back to the original row ids
    saved by main3.py. Handles common variants like 'orig_index' or 'Unnamed: 0'.
    """
    cols = set(preds.columns)
    if "index" in cols:
        return preds
    for cand in ("orig_index", "Unnamed: 0", "Unnamed: 0.1", "row", "row_index"):
        if cand in cols:
            preds = preds.rename(columns={cand: "index"})
            return preds
    # If there's NO clear index column, but preds has same length as the full df,
    # we can fabricate a sequential index — but this usually won't match test-only files.
    # Better to raise a helpful error listing columns.
    raise ValueError(f"Preds file missing an 'index' column. Available columns: {list(preds.columns)}")

def analyze_target(df: pd.DataFrame, preds_path: str, target: str):
    print(f"\n=== Analyzing target: {target} ===")
    preds = pd.read_csv(preds_path)
    preds = detect_and_rename_pred_index(preds)

    # Left DF: keep original row id as 'orig_index'
    left = df.reset_index().rename(columns={"index": "orig_index"})
    merged = left.merge(preds, left_on="orig_index", right_on="index", how="inner")

    if merged.empty:
        raise ValueError(f"Merged dataset is empty for target '{target}'. "
                         f"Check that preds 'index' maps to original row indices.")

    # Buckets on y_true
    merged["__bucket__"] = make_buckets(merged["y_true"].values)

    # Per-group summaries
    out_frames = []
    for name, cols in ALL_GROUPS:
        summary = summarize_group(merged, cols, y_col="y_true", bucket_col="__bucket__")
        if not summary.empty:
            summary.insert(0, "group", name)
            out_frames.append(summary)

    if out_frames:
        comp = pd.concat(out_frames, ignore_index=True)
        comp_path = f"feature_comparison_{target}.csv"
        comp.to_csv(comp_path, index=False)
        print(f"[{target}] wrote {comp_path}")
        print("[top shifts] (by |delta_mean|):")
        print(comp.reindex(comp["delta_mean"].abs().sort_values(ascending=False).index)
                 .head(5)[["group","feature","mean_high","mean_low","delta_mean","cohen_d","corr"]]
                 .to_string(index=False))
    else:
        print(f"[{target}] No known feature groups present to summarize.")

    # Quadrant shares
    for prefix in ["faces", "text"]:
        share = quadrant_shares(merged, prefix, "__bucket__")
        if not share.empty:
            outp = f"quadrant_share_{prefix}_{target}.csv"
            share.to_csv(outp, index=False)
            print(f"[{target}] wrote {outp}")

    # Threshold sweep on text_area_pct
    if "text_area_pct" in merged.columns:
        thr = threshold_sweep(merged["text_area_pct"], merged["y_true"])
        if not thr.empty:
            thr_path = f"threshold_text_area_{target}.csv"
            thr.to_csv(thr_path, index=False)
            best = thr.iloc[0]
            print(f"[{target}] best text_area_pct threshold = {best['threshold']:.6f} "
                  f"(Δmean={best['delta_mean']:.6f}; n_left={int(best['n_left'])}, n_right={int(best['n_right'])})")
            print(f"[{target}] wrote {thr_path}")
        else:
            print(f"[{target}] threshold sweep: insufficient variation or small splits.")
    else:
        print(f"[{target}] 'text_area_pct' not found; skipping threshold sweep.")

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"joint_clean.csv not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    for target, path in PREDS_FILES.items():
        if os.path.exists(path):
            analyze_target(df, path, target)
        else:
            print(f"[{target}] preds file not found at {path}")

if __name__ == "__main__":
    main()
