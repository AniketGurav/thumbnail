# main3.py
# Train/evaluate on joint_clean.csv with configurable target metric.
# This version FIXES target leakage by excluding view-derived features from X.
# It supports 4 targets: raw_views, views_per_sub, views_per_sub_per_day, log_views.
# Metrics: RMSE, MAE, R^2. Saves per-target preds and a combined summary + pivots.

import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.models3 import build_models

# ------------------------- Configurable Targets ------------------------- #
# Targets supported:
#   1) raw_views               : views
#   2) views_per_sub           : views / channel_follower_count
#   3) views_per_sub_per_day   : views / (channel_follower_count * days_since_upload)
#   4) log_views               : log1p(views)
#
# For (1)-(3) you can optionally apply log1p *during training* via --log1p,
# but evaluation is always reported in the ORIGINAL space of that target.
# For (4) 'log_views', the target IS log(views+1), so no extra log1p is applied.

TARGET_CHOICES = ("raw_views", "views_per_sub", "views_per_sub_per_day", "log_views")

# Columns that are either identifiers/paths or directly target-like and must be excluded.
IDENT_EXCLUDE = {
    "id", "url", "channel", "channel_id", "channel_url",
    "playlist_id", "playlist_title", "thumbnail", "thumbnail_path",
    "fullPath", "subtitle_path", "upload_date", "title",
    "views"  # never feed direct target into X
}

# Columns that are VIEW-DERIVED and cause leakage for ANY view-based target.
# (They either contain views in numerator/denominator or are algebraically tied to targets.)
VIEW_DERIVED_ALWAYS_EXCLUDE = {
    "views_per_subscriber",   # = views / subs
    "like_ratio",             # = likes / views
    "comment_ratio",          # = comments / views
}

def compute_target(df: pd.DataFrame, target_type: str) -> np.ndarray:
    views = pd.to_numeric(df.get("views", 0), errors="coerce").fillna(0.0)
    subs = pd.to_numeric(df.get("channel_follower_count", 0), errors="coerce").fillna(0.0)
    subs_safe = subs.replace(0, 1.0)

    days = pd.to_numeric(df.get("days_since_upload", 0), errors="coerce").fillna(0.0)
    days_safe = days.replace(0, 1.0)

    if target_type == "raw_views":
        y = views.values.astype(float)
    elif target_type == "views_per_sub":
        y = (views / subs_safe).values.astype(float)
    elif target_type == "views_per_sub_per_day":
        y = (views / (subs_safe * days_safe)).values.astype(float)
    elif target_type == "log_views":
        # Target space IS log1p(views); evaluated in log space as well.
        y = np.log1p(np.clip(views.values.astype(float), a_min=0.0, a_max=None))
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    # Replace inf/nan after division
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y

def maybe_log1p(y: np.ndarray, use_log1p: bool, target_type: str) -> Tuple[np.ndarray, Any, Any, str]:
    """
    Return (y_trainable, fwd, inv, metric_space_label)
      - fwd is function to transform y for training
      - inv is inverse function applied to predictions for eval/reporting
      - metric_space_label is 'original' for (1)-(3) and 'log' for (4)
    NOTE: For target_type == 'log_views', we DO NOT apply extra log1p.
    """
    if target_type == "log_views":
        # Already in log space; train there and report metrics in log space
        return y, (lambda a: a), (lambda a: a), "log"
    if not use_log1p:
        return y, (lambda a: a), (lambda a: a), "original"
    return np.log1p(np.clip(y, a_min=0.0, a_max=None)), np.log1p, np.expm1, "original"

def choose_features(df: pd.DataFrame, target_type: str) -> List[str]:
    """
    Auto-select numeric engineered columns while removing:
      - identifiers/paths
      - the direct target ('views')
      - ALL view-derived features that cause leakage (e.g., views_per_subscriber, like_ratio, comment_ratio)
    This unified exclusion avoids algebraic reconstruction of the target for any target_type.
    """
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set(IDENT_EXCLUDE) | set(VIEW_DERIVED_ALWAYS_EXCLUDE)

    # If you later add more engineered ratios containing 'views', add them here defensively.

    # Final feature list
    feats = [c for c in numeric if c not in exclude]
    return sorted(feats)

def clean_for_modeling(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    # Replace inf/nan with zeros (we also have explicit missing flags in the dataset)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Ensure all numeric dtype
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X

def leak_checks(df: pd.DataFrame, target_type: str, y_raw: np.ndarray) -> None:
    """
    Print correlations between the target and a set of suspicious columns.
    Also scan all numeric columns for near-perfect correlation with the target (|r| > 0.98).
    This is an advisory print (features are already excluded above).
    """
    print(f"[leak-check] Running leakage checks for target = {target_type}")
    sus_cols = ["views_per_subscriber", "like_ratio", "comment_ratio", "channel_follower_count", "days_since_upload"]
    for col in sus_cols:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
            if np.std(x) > 0 and np.std(y_raw) > 0:
                r = np.corrcoef(y_raw, x)[0, 1]
                print(f"[leak-check] corr({col}, target) = {r:.6f}")

    # Generic scan for near-perfect linear correlation (warning only)
    num_cols = df.select_dtypes(include=[np.number]).columns
    flagged = []
    for col in num_cols:
        if col in IDENT_EXCLUDE or col in VIEW_DERIVED_ALWAYS_EXCLUDE or col == "views":
            continue
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
        if np.std(x) > 0 and np.std(y_raw) > 0:
            r = np.corrcoef(y_raw, x)[0, 1]
            if abs(r) > 0.98:
                flagged.append((col, r))
    if flagged:
        print("[leak-check] WARNING: The following features are ~linearly identical to the target (|r|>0.98). "
              "Consider excluding if unexpected:")
        for col, r in sorted(flagged, key=lambda t: -abs(t[1])):
            print(f"  - {col:30s} r={r:.6f}")
    else:
        print("[leak-check] No features with |corr| > 0.98 against target.")

# ----------------------------- Training Loop ----------------------------- #

def train_and_eval(
    df: pd.DataFrame,
    target_type: str,
    use_log1p: bool,
    config: Dict[str, Any],
    run_tag: str
) -> pd.DataFrame:
    """
    Train the full model suite on a chosen target_type and return a results DataFrame.
    Prints a compact report and returns per-model metrics.
    """
    assert target_type in TARGET_CHOICES, f"target_type must be one of {TARGET_CHOICES}"

    y_raw = compute_target(df, target_type)
    leak_checks(df, target_type, y_raw)

    feature_cols = choose_features(df, target_type)
    X = clean_for_modeling(df, feature_cols)

    y_trainable, _, inv, metric_space = maybe_log1p(y_raw, use_log1p=use_log1p, target_type=target_type)

    # Split
    test_size = float(config.get("test_size", 0.4))
    seed = int(config.get("random_state", 42))
    X_train, X_test, y_train, y_test, idx_tr, idx_te = train_test_split(
        X, y_trainable, np.arange(len(X)), test_size=test_size, random_state=seed
    )

    models = build_models(config)

    rows = []
    for name, model in models.items():
        if name == "ResNet+MLP":
            # Skipping image training path here; can be wired if needed.
            continue

        # Fit
        model.fit(X_train, y_train)

        # Predict in trainable space -> invert for reporting (if applicable)
        y_pred_trainable = model.predict(X_test)
        y_pred_trainable = np.asarray(y_pred_trainable).reshape(-1)
        y_pred = inv(y_pred_trainable)

        # Compute metrics in the appropriate reporting space:
        rmse = mean_squared_error(y_raw[idx_te], y_pred, squared=False)
        r2 = r2_score(y_raw[idx_te], y_pred)
        mae = mean_absolute_error(y_raw[idx_te], y_pred)

        rows.append({
            "run": run_tag,
            "target_type": target_type,
            "metric_space": metric_space,  # 'original' or 'log'
            "use_log1p_train": (use_log1p and target_type != "log_views"),
            "model": name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": X.shape[1]
        })

    res = pd.DataFrame(rows).sort_values(by=["target_type", "rmse", "model"])
    print("\n" + "="*100)
    print(f"[{run_tag}] Target: {target_type} | metric_space={metric_space} | log1p_train={use_log1p and target_type!='log_views'} | Features: {X.shape[1]}")
    print(res.to_string(index=False))
    print("="*100 + "\n")

    # Save predictions for the best model (top-1 by RMSE within this target)
    if not res.empty:
        best_idx = res["rmse"].values.argmin()
        best_name = res.iloc[best_idx]["model"]
        best_model = build_models(config)[best_name]
        best_model.fit(X_train, y_train)
        y_best_pred = inv(best_model.predict(X_test))
        dump = pd.DataFrame({
            "index": idx_te,
            "y_true": y_raw[idx_te],
            "y_pred": y_best_pred
        })
        out_preds = f"preds_{run_tag}_{target_type}_{'logtrain' if (use_log1p and target_type!='log_views') else 'linear'}.csv"
        dump.to_csv(out_preds, index=False)
        print(f"[{run_tag}] Saved predictions to {out_preds}")

    return res

# --------------------------------- CLI Main -------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    # DO NOT CHANGE: keep your hard-coded default path
    ap.add_argument("--data", default="/cluster/datastore/aniketag/urumi/metadata/data/joint_clean.csv", help="Path to joint_clean.csv")
    # Include ALL four targets by default
    ap.add_argument("--metrics", default="raw_views,views_per_sub,views_per_sub_per_day,log_views",
                    help="Comma-separated target choices")
    ap.add_argument("--log1p", action="store_true", help="Apply log1p to target during training (ignored for log_views)")
    ap.add_argument("--test_size", type=float, default=0.4)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--xgb", action="store_true", help="Enable XGBoost if installed")
    args = ap.parse_args()

    # Load
    df = pd.read_csv(args.data)

    # If some critical engineered columns are missing, fill safe defaults
    for col, default in [
        ("days_since_upload", df.get("days_since_upload", pd.Series([0]*len(df)))),
        ("faces_missing", 0), ("text_missing", 0),
    ]:
        if col not in df.columns:
            df[col] = default

    # Build config for models
    config: Dict[str, Any] = {
        "test_size": args.test_size,
        "random_state": args.random_state,
        "use_images": False,
        "enable_xgb": args.xgb,
        # Tunables
        "rf_n_estimators": 500,
        "rf_max_depth": 14,
        "svr_C": 10.0,
        "svr_epsilon": 0.2,
        "mlp_hidden": (512, 256, 128),
        "mlp_lr": 1e-3,
        "mlp_max_iter": 400,
        "xgb_n_estimators": 700,
        "xgb_lr": 0.05,
        "xgb_max_depth": 6,
        "xgb_subsample": 0.8,
        "xgb_colsample": 0.8
    }

    # Run multiple passes (one per metric)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    all_res = []
    for metric in metrics:
        if metric not in TARGET_CHOICES:
            print(f"[main3] Skipping unknown metric: {metric}")
            continue
        res = train_and_eval(
            df=df,
            target_type=metric,
            use_log1p=args.log1p,
            config=config,
            run_tag="main3"
        )
        all_res.append(res)

    if all_res:
        final = pd.concat(all_res, axis=0, ignore_index=True)
        # Order columns nicely
        cols = ["run", "target_type", "metric_space", "use_log1p_train", "model", "rmse", "mae", "r2", "n_train", "n_test", "n_features"]
        final = final[cols]
        final.to_csv("results_main3_summary.csv", index=False)
        print("[main3] Wrote summary to results_main3_summary.csv")
        # Also print a compact pivoted comparison by target & model
        print("\n[main3] Compact comparison (RMSE):")
        try:
            print(final.pivot_table(index="model", columns="target_type", values="rmse").round(6).to_string())
        except Exception:
            print("(pivot failed)")
        print("\n[main3] Compact comparison (R2):")
        try:
            print(final.pivot_table(index="model", columns="target_type", values="r2").round(6).to_string())
        except Exception:
            print("(pivot failed)")

if __name__ == "__main__":
    main()
