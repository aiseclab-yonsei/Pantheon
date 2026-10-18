#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quality-aware Ensemble Multi-Dataset Analysis (Table-friendly + Plot)
====================================================================
- Frame metrics: AUC/ACC (NaN/Inf-safe)
- Video metrics: STRICT (mean over frames, drop mixed-label videos)

Models:
  raw, c40, c23, all, q_routed, s_soft, ens_mix_high(Pantheon), ens_adapt(Pantheon(Gate)), upper_bound

Gate:
  - trained on FF++ (all real + fake=c40)
  - features = BASE + Qspread = [conf, q_std, q_margin, q_range]

PATCHED video_id rules:
  - Default: ".../frames/<video_id>/<frame>.png" -> <video_id>, else parent folder
  - WildDeepfake: anchor('real'/'fake') + context to avoid collisions
  - DF40: last 2 dirs above file (non-video folders excluded for video metrics)
  - FFIW (NEW, as requested):
      video_name = parent folder of the png file  (i.e., parts[-2])
      label_tag  = infer from path containing '/real/' or '/fake/' (closest anchor)
      video_id   = f"{label_tag}/{video_name}"
    -> This exactly matches your statement:
       "png 위 폴더가 영상이름" + "real/fake 구분을 추가해서 video-level grouping"

OUTPUT:
  - per-dataset result table
  - ALL_results_frame_video_auc_with_deltas.csv
  - SUMMARY_across_datasets_mean_var_std.csv (per model mean/var/std over datasets)

PLOT:
  - video_auc_strict_table_run.png
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HAS_SK = True
try:
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:
    HAS_SK = False

plt.rcParams["font.size"] = 20

LABEL_MAP = {
    "raw": "Global(raw)",
    "c23": "Global(c23)",
    "c40": "Global(c40)",
    "all": "Global(all)",
    "s_soft": "Soft(RBF)",
    "q_routed": "Q-route",
    "ens_mix_high": "Pantheon",
    "ens_adapt": "Pantheon(Gate)",
    "upper_bound": "Oracle(UB)",
}

FFPP_STD_CSV = (
    "/home/compu/eunbi/iqa/0.module/analysis/qcn_crfiqa/"
    "normalize_minmax_median0_v2/standardized_csv/"
    "FaceForensics++_standardized.csv"
)

RBF_CENTERS: Optional[np.ndarray] = None

DF40_NONVIDEO_FOLDERS = [
    "DiT", "pixart", "RDDM", "sd2.1", "SiT",
    "StyleGAN2", "StyleGAN3", "StyleGANXL",
    "StyleGANXL", "VQGAN",
]

# ===========================================================
# Utility
# ===========================================================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def sanitize_filename(s: str) -> str:
    s = str(s)
    s = s.replace("/", "_").replace("\\", "_").replace("|", "_").replace(":", "_")
    s = s.replace(" ", "_")
    return s

def read_csv_for_dataset(ds: str, merge_dir: str) -> pd.DataFrame:
    path = os.path.join(merge_dir, f"merged_{ds}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[ERROR] Missing merged csv: {path}")
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise KeyError(f"[ERROR] {path} missing 'label'")
    df["label"] = df["label"].astype(int)
    if "quadrant" in df.columns:
        df["quadrant"] = df["quadrant"].astype(int)
    return df

def _finite_mask(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.isfinite(a)

def safe_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores).astype(float)

    m = _finite_mask(s)
    y = y[m]
    s = s[m]

    if len(y) == 0:
        return None
    if len(np.unique(y)) < 2:
        return None
    if np.allclose(s, s[0]):
        return 0.5
    return float(roc_auc_score(y, s))

def safe_acc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores).astype(float)

    m = _finite_mask(s)
    y = y[m]
    s = s[m]

    if len(y) == 0:
        return None
    preds = (s >= 0.5).astype(int)
    return float(accuracy_score(y, preds))

def metric_frame_auc_acc(df: pd.DataFrame, score_col: str) -> Dict[str, Optional[float]]:
    if score_col not in df.columns:
        return {"auc": None, "acc": None, "n_valid": 0, "n_total": int(len(df))}
    y = df["label"].astype(int).values
    s = pd.to_numeric(df[score_col], errors="coerce").astype(float).values
    m = _finite_mask(s)
    auc = safe_auc(y, s)
    acc = safe_acc(y, s)
    return {
        "auc": auc,
        "acc": acc,
        "n_valid": int(m.sum()),
        "n_total": int(len(s)),
    }

def save_nonfinite_paths(df: pd.DataFrame, ds: str, out_dir: str, score_col: str) -> int:
    if score_col not in df.columns:
        return 0
    s = pd.to_numeric(df[score_col], errors="coerce").astype(float)
    bad = ~np.isfinite(s.values)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return 0

    txt_path = os.path.join(out_dir, f"nonfinite_paths_{sanitize_filename(ds)}__{sanitize_filename(score_col)}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for p in df.loc[bad, "path"].astype(str).tolist():
            f.write(p + "\n")

    csv_path = os.path.join(out_dir, f"nonfinite_rows_{sanitize_filename(ds)}__{sanitize_filename(score_col)}.csv")
    df.loc[bad, ["path", "label", score_col]].to_csv(csv_path, index=False)

    print(f"[NON-FINITE] {ds} | {score_col}: n_bad={n_bad:,} -> saved")
    return n_bad

def col_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

# ===========================================================
# Video id extraction
# ===========================================================
_FRAMES_RE = re.compile(r"/frames/([^/]+)/[^/]+$")

def _find_last_anchor(parts: List[str], anchors=("real", "fake")) -> Optional[Tuple[str, int]]:
    low = [x.lower() for x in parts]
    for i in range(len(low) - 1, -1, -1):
        if low[i] in anchors:
            return low[i], i
    return None

def infer_video_id_from_path(p: str, ds: str) -> str:
    """
    Default:
      - ".../frames/<video_id>/<frame>.png" -> <video_id>
      - else parent folder

    WildDeepfake:
      - anchor('real'/'fake') and compose context to avoid collisions

    DF40:
      - last 2 dirs above file

    FFIW (requested exact behavior):
      - video_name = parent folder of png file  (folder directly above png)
      - label_tag  = inferred from path containing anchor '/real/' or '/fake/' (nearest from end)
      - video_id   = f"{label_tag}/{video_name}"
    """
    s = str(p).replace("\\", "/")
    ds_l = str(ds).lower()
    parts = [x for x in s.split("/") if x != ""]
    dirs = parts[:-1] if len(parts) >= 2 else parts  # drop filename

    # DF40
    if ds_l == "df40":
        if len(dirs) >= 2:
            return "/".join(dirs[-2:])
        if len(dirs) == 1:
            return dirs[-1]
        return s

    # WildDeepfake
    if "wild" in ds_l and "deepfake" in ds_l:
        anchor = _find_last_anchor(dirs, anchors=("real", "fake"))
        if anchor is not None:
            _, idx = anchor
            st = max(0, idx - 2)
            ed = min(len(dirs), idx + 2)  # include anchor + 1 child
            return "/".join(dirs[st:ed])
        if len(dirs) >= 2:
            return "/".join(dirs[-2:])
        return s

    # FFIW (your exact rule)
    if ds_l == "ffiw":
        # parent folder of png is video name
        video_name = dirs[-1] if len(dirs) >= 1 else "NA"
        anchor = _find_last_anchor(dirs, anchors=("real", "fake"))
        label_tag = anchor[0] if anchor is not None else "unknown"
        return f"{label_tag}/{video_name}"

    # Default frames/<video_id>/<frame>
    m = _FRAMES_RE.search(s)
    if m:
        return m.group(1)

    # fallback parent dir
    if len(dirs) >= 1:
        return dirs[-1]
    return s

def get_video_id_series(df: pd.DataFrame, ds: str) -> pd.Series:
    if "video" in df.columns:
        return df["video"].astype(str)
    return df["path"].astype(str).apply(lambda x: infer_video_id_from_path(x, ds))

def filter_df40_video_frames(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if dataset_name != "DF40":
        return df

    def is_non_video(p: str) -> bool:
        p = str(p).replace("\\", "/")
        for name in DF40_NONVIDEO_FOLDERS:
            if f"/{name}/" in p:
                return True
        return False

    mask = ~df["path"].astype(str).apply(is_non_video)
    removed = int((~mask).sum())
    if removed > 0:
        print(f"[DF40 FILTER] Exclude {removed} non-video frames for video-level metrics.")
    return df[mask].copy()

# ===========================================================
# STRICT video handling
# ===========================================================
def strict_video_filter_once(
    df: pd.DataFrame,
    ds: str,
    out_dir: str,
    video_id_col: str = "video_id",
    save_examples: bool = True,
    max_video_ids: int = 200,
    paths_per_label: int = 3,
    save_all_mixed_paths_txt: bool = True,
    max_mixed_paths: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    g = df.groupby(video_id_col)["label"].nunique(dropna=False)
    mixed = g[g > 1].sort_values(ascending=False)

    n_videos = int(g.shape[0])
    n_mixed = int(mixed.shape[0])
    ratio = (n_mixed / max(n_videos, 1)) * 100.0
    print(f"[MIXED-LABEL CHECK] {ds}: total_videos={n_videos:,} | mixed_videos={n_mixed:,} ({ratio:.4f}%)")

    if n_mixed > 0 and save_all_mixed_paths_txt:
        mixed_ids = mixed.index.astype(str).tolist()
        mdf = df[df[video_id_col].astype(str).isin(mixed_ids)][[video_id_col, "path", "label"]].copy()
        if max_mixed_paths is not None and len(mdf) > max_mixed_paths:
            mdf = mdf.head(int(max_mixed_paths))
            print(f"[WARN] mixed paths capped to first {max_mixed_paths} rows")

        txt_path = os.path.join(out_dir, f"mixed_label_frame_paths_{sanitize_filename(ds)}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for p in mdf["path"].astype(str).tolist():
                f.write(p + "\n")
        print(f"[SAVED] {txt_path} (n_paths={len(mdf):,})")

    if save_examples and n_mixed > 0:
        mixed_ids = mixed.index.tolist()[:max_video_ids]
        rows = []
        for vid in mixed_ids:
            d = df[df[video_id_col] == vid][["path", "label"]].copy()
            ex_real = d[d["label"] == 0].head(paths_per_label)
            ex_fake = d[d["label"] == 1].head(paths_per_label)
            ex = pd.concat([ex_real, ex_fake], axis=0)
            for _, r in ex.iterrows():
                rows.append({
                    "dataset": ds,
                    "video_id": str(vid),
                    "label": int(r["label"]),
                    "path": str(r["path"]),
                })
        rep = pd.DataFrame(rows)
        save_path = os.path.join(out_dir, f"mixed_label_videos_{sanitize_filename(ds)}.csv")
        rep.to_csv(save_path, index=False)
        print(f"[SAVED] {save_path}")

    good_video_ids = set(g[g == 1].index.astype(str).tolist())
    df_strict = df[df[video_id_col].astype(str).isin(good_video_ids)].copy()

    info = {
        "dataset": ds,
        "total_videos": n_videos,
        "mixed_videos": n_mixed,
        "mixed_ratio_pct": ratio,
        "strict_frames": int(len(df_strict)),
        "strict_videos": int(len(good_video_ids)),
    }
    return df_strict, info

def compute_video_metrics_strict_from_frames(
    df_frames_strict: pd.DataFrame,
    score_col: str,
    video_id_col: str = "video_id",
) -> Dict[str, Optional[float]]:
    if score_col not in df_frames_strict.columns:
        return {"auc": None, "acc": None, "n_valid_videos": 0, "n_total_videos": 0}

    if len(df_frames_strict) == 0:
        return {"auc": None, "acc": None, "n_valid_videos": 0, "n_total_videos": 0}

    dfx = df_frames_strict.copy()
    dfx[score_col] = pd.to_numeric(dfx[score_col], errors="coerce").astype(float)
    dfx = dfx[np.isfinite(dfx[score_col].values)].copy()
    if len(dfx) == 0:
        return {"auc": None, "acc": None, "n_valid_videos": 0, "n_total_videos": 0}

    vd = (
        dfx.groupby(video_id_col, as_index=False)
           .agg(label=("label", "max"), score=(score_col, "mean"))
    )

    n_total = int(len(vd))
    vd = vd[np.isfinite(vd["score"].astype(float).values)].copy()
    n_valid = int(len(vd))

    if len(vd) == 0:
        return {"auc": None, "acc": None, "n_valid_videos": 0, "n_total_videos": n_total}

    y_true = vd["label"].astype(int).values
    y_score = vd["score"].astype(float).values
    return {
        "auc": safe_auc(y_true, y_score),
        "acc": safe_acc(y_true, y_score),
        "n_valid_videos": n_valid,
        "n_total_videos": n_total,
    }

# ===========================================================
# RBF centers
# ===========================================================
def load_ffpp_rbf_centers_from_train(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    need = ["cr_fiqa_norm_centered", "qcn_norm_centered"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"[ERROR] FF++ standardized CSV missing '{c}'")

    x = df["cr_fiqa_norm_centered"].astype(float).values
    y = df["qcn_norm_centered"].astype(float).values

    q1 = (x >= 0) & (y >= 0)
    q2 = (x < 0) & (y >= 0)
    q3 = (x < 0) & (y < 0)
    q4 = (x >= 0) & (y < 0)

    centers = np.zeros((4, 2), dtype=float)

    def _center(mask, name):
        if mask.sum() == 0:
            print(f"[WARN] FF++ train {name} empty -> (0,0)")
            return 0.0, 0.0
        return float(x[mask].mean()), float(y[mask].mean())

    centers[0] = _center(q1, "Q1")
    centers[1] = _center(q2, "Q2")
    centers[2] = _center(q3, "Q3")
    centers[3] = _center(q4, "Q4")

    print("[INFO] FF++ RBF centers (Q1~Q4):")
    print(centers)
    return centers

# ===========================================================
# Routing & soft scores
# ===========================================================
def make_qrouted(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["quadrant", "q1", "q2", "q3", "q4"]:
        if c not in df.columns:
            raise KeyError(f"[ERROR] merged csv missing '{c}' for q_routed.")
    df = df.copy()
    df["q_routed"] = df.apply(lambda r: r[f"q{int(r['quadrant'])}"], axis=1)
    return df

def quality_weights_rbf(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    global RBF_CENTERS
    if RBF_CENTERS is None:
        raise RuntimeError("RBF_CENTERS is None (load FF++ centers first).")
    z = np.stack([x, y], axis=1)
    diff = z[:, None, :] - RBF_CENTERS[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    W = np.exp(-d2 / (2.0 * sigma ** 2))
    return W / (W.sum(axis=1, keepdims=True) + 1e-8)

def compute_quality_soft_scores(df: pd.DataFrame, sigma_rbf: float = 1.0) -> np.ndarray:
    need = ["crfiqa", "qcn", "q1", "q2", "q3", "q4"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"[ERROR] merged csv missing '{c}' for quality-soft.")
    x = pd.to_numeric(df["crfiqa"], errors="coerce").astype(float).values
    y = pd.to_numeric(df["qcn"], errors="coerce").astype(float).values
    W = quality_weights_rbf(x, y, sigma=sigma_rbf)
    q = df[["q1", "q2", "q3", "q4"]].astype(float).values
    return (W * q).sum(axis=1)

def compute_error(score: np.ndarray, label: np.ndarray) -> np.ndarray:
    if score.ndim == 1:
        score = score[:, None]
    return np.where(label[:, None] == 1, 1.0 - score, score)

def compute_upper_bound_scores(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    cols2 = [c for c in cols if c in df.columns]
    if len(cols2) == 0:
        raise RuntimeError("upper_bound has no available candidate columns.")
    labels = df["label"].astype(int).values
    scores = df[cols2].astype(float).values
    scores = np.where(np.isfinite(scores), scores, 0.5)
    errs = compute_error(scores, labels)
    best_idx = np.argmin(errs, axis=1)
    return scores[np.arange(len(df)), best_idx]

# ===========================================================
# Gate features (BASE+Qspread)
# ===========================================================
def add_gate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    q = df[["q1", "q2", "q3", "q4"]].astype(float).values
    q_std = q.std(axis=1)
    q_sorted = np.sort(q, axis=1)
    q_max = q_sorted[:, -1]
    q_second = q_sorted[:, -2]
    q_min = q_sorted[:, 0]

    df["q_std"] = q_std
    df["q_margin"] = q_max - q_second
    df["q_range"] = q_max - q_min

    s = pd.to_numeric(df["c23"], errors="coerce").astype(float).values
    y = df["label"].astype(int).values
    df["conf"] = np.abs(s - 0.5) * 2.0
    df["y_correct"] = ((s >= 0.5).astype(int) == y).astype(int)
    return df

# ===========================================================
# FF++ c40-only filter for gate training
# ===========================================================
def parse_ffpp_quality(p: str) -> str:
    s = str(p).replace("\\", "/").lower()
    if "/c40/" in s or "faceforensics++_c40" in s:
        return "c40"
    if "/c23/" in s or "faceforensics++_c23" in s:
        return "c23"
    if "/raw/" in s or "faceforensics++_raw" in s:
        return "raw"
    return "unknown"

def filter_ffpp_c40_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    q = df["path"].astype(str).apply(parse_ffpp_quality)
    is_real = (df["label"] == 0)
    is_fake_c40 = (df["label"] == 1) & (q == "c40")
    out = df[is_real | is_fake_c40].copy()
    print(f"[FF++] keep fake=c40 + all_real: N={len(out):,} (real={(out['label']==0).sum():,}, fake={(out['label']==1).sum():,})")
    return out

# ===========================================================
# Gate model
# ===========================================================
def fit_gate_hgb(df_train: pd.DataFrame, feat_cols: List[str], seed: int = 0):
    d = df_train.dropna(subset=feat_cols + ["y_correct"]).copy()
    X = d[feat_cols].values.astype(float)
    y = d["y_correct"].values.astype(int)

    model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=400, random_state=seed
    )
    model.fit(X, y)

    p = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, (p >= 0.5).astype(int)) * 100.0
    auc = roc_auc_score(y, p) * 100.0 if len(np.unique(y)) > 1 else float("nan")
    print("\n[Gate HGB trained on FF++ (real + fake=c40)]")
    print(f"  feats      = {feat_cols}")
    print(f"  train_acc  = {acc:.2f}%")
    print(f"  train_auc  = {auc:.2f}%")
    return model

def gate_predict_p_correct(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]

# ===========================================================
# Mixers
# ===========================================================
def build_mix_static(s_c23: np.ndarray, s_soft: np.ndarray, low_mask: np.ndarray, lam_base: float) -> np.ndarray:
    out = np.empty_like(s_c23, dtype=np.float32)
    out[low_mask] = s_soft[low_mask]
    hi = ~low_mask
    out[hi] = lam_base * s_c23[hi] + (1.0 - lam_base) * s_soft[hi]
    return out

def lambda_from_pwrong_piecewise(p_wrong: np.ndarray, lam_base: float, lam_min: float, wrong_low: float, wrong_high: float) -> np.ndarray:
    pw = np.clip(p_wrong, 0.0, 1.0)
    lam = np.empty_like(pw, dtype=np.float32)
    a = pw <= wrong_low
    b = pw >= wrong_high
    m = (~a) & (~b)
    lam[a] = lam_base
    lam[b] = lam_min
    if m.any():
        t = (pw[m] - wrong_low) / max((wrong_high - wrong_low), 1e-12)
        lam[m] = lam_base + t * (lam_min - lam_base)
    return lam

def build_mix_adaptive(
    s_c23: np.ndarray,
    s_soft: np.ndarray,
    low_mask: np.ndarray,
    p_correct: np.ndarray,
    lam_base: float,
    lam_min: float,
    wrong_low: float,
    wrong_high: float,
) -> Tuple[np.ndarray, np.ndarray]:
    out = np.empty_like(s_c23, dtype=np.float32)
    out[low_mask] = s_soft[low_mask]
    hi = ~low_mask
    p_wrong = 1.0 - p_correct
    lam_all = lambda_from_pwrong_piecewise(p_wrong, lam_base, lam_min, wrong_low, wrong_high)
    out[hi] = lam_all[hi] * s_c23[hi] + (1.0 - lam_all[hi]) * s_soft[hi]
    return out, lam_all

# ===========================================================
# Shared preprocessing
# ===========================================================
def make_qrouted(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["quadrant", "q1", "q2", "q3", "q4"]:
        if c not in df.columns:
            raise KeyError(f"[ERROR] merged csv missing '{c}' for q_routed.")
    df = df.copy()
    df["q_routed"] = df.apply(lambda r: r[f"q{int(r['quadrant'])}"], axis=1)
    return df

def build_df_common(df: pd.DataFrame, ds: str) -> pd.DataFrame:
    df = df.copy()
    df = make_qrouted(df)
    df["s_soft"] = compute_quality_soft_scores(df, sigma_rbf=RBF_SIGMA)
    df = add_gate_features(df)
    df["__dataset__"] = ds
    return df

# ===========================================================
# Plot
# ===========================================================
# def plot_video_auc_strict(
#     datasets: List[str],
#     series: Dict[str, List[Optional[float]]],
#     out_dir: str,
#     fname: str = "video_auc_strict.png",
# ):
#     x = np.arange(len(datasets))
#     plt.figure(figsize=(12, 5))

#     order = ["ens_adapt", "ens_mix_high", "c23", "all", "q_routed", "upper_bound"]

#     for k in order:
#         if k not in series:
#             continue
#         vals = series[k]
#         label = LABEL_MAP.get(k, k)

#         if k == "upper_bound":
#             plt.plot(x, vals, linewidth=2, linestyle="--", color="gray", label=label)
#         elif k == "ens_adapt":
#             plt.plot(x, vals, linewidth=4, color="red", label=label)
#         else:
#             plt.plot(x, vals, linewidth=2, label=label)

#     plt.xticks(x, datasets, rotation=20)
#     plt.ylabel("Video AUC (%)")
#     plt.grid(True, linestyle="--", alpha=0.35)
#     plt.legend(loc="lower left", fontsize=15)
#     plt.tight_layout()

#     out_path = os.path.join(out_dir, fname)
#     plt.savefig(out_path, dpi=200)
#     plt.close()
#     print(f"[SAVED] {out_path}")
def plot_video_auc_strict(
    datasets: List[str],
    series: Dict[str, List[Optional[float]]],
    out_dir: str,
    fname: str = "video_auc_strict.png",
):
    x = np.arange(len(datasets))
    plt.figure(figsize=(12,5))

    # 원하는 표시 순서 (필요하면 여기 조정)
    order = ["ens_adapt","s_soft", "c23", "all", "c40", "raw","q_routed", "upper_bound"]

    for k in order:
        if k not in series:
            continue
        vals = series[k]
        label = LABEL_MAP.get(k, k)

        # Style rules:
        if k == "upper_bound":
            # thick, dashed, gray
            plt.plot(x, vals, linewidth=4, linestyle="--", color="gray", label=label)
        elif k == "ens_adapt":
            # thick, red
            plt.plot(x, vals, linewidth=5, color="red", label=label)
        else:
            # thin for everything else (globals + soft + etc.)
            plt.plot(x, vals, linewidth=1.5, label=label)

    plt.xticks(x, datasets, rotation=20)
    plt.ylabel("Video AUC (%)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="lower left", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[SAVED] {out_path}")

# ===========================================================
# Summary across datasets
# ===========================================================
def summarize_across_datasets(
    all_rows: pd.DataFrame,
    out_dir: str,
    exclude_prefixes: Tuple[str, ...] = ("FaceForensics++",),
) -> pd.DataFrame:
    df = all_rows.copy()

    # Exclude FaceForensics++ (and its split variants) from summary aggregation
    if exclude_prefixes:
        ds_series = df["dataset"].astype(str)
        mask_ex = np.zeros(len(df), dtype=bool)
        for px in exclude_prefixes:
            mask_ex |= ds_series.str.startswith(px)
        df = df[~mask_ex].copy()

    # numeric casting
    for c in ["frame_auc","frame_acc","video_auc","video_acc","d_frame_auc_vs_c23","d_video_auc_vs_c23"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Model-wise mean summary (including mean deltas)
    agg = (
        df.groupby("model", as_index=False)
          .agg(
              n_datasets=("dataset", "nunique"),

              video_auc_mean=("video_auc", "mean"),
              d_video_auc_mean=("d_video_auc_vs_c23", "mean"),

              frame_auc_mean=("frame_auc", "mean"),
              d_frame_auc_mean=("d_frame_auc_vs_c23", "mean"),

              video_acc_mean=("video_acc", "mean"),
              frame_acc_mean=("frame_acc", "mean"),
          )
    )

    order = ["c23","raw","c40","all","q_routed","s_soft","ens_mix_high","ens_adapt","upper_bound"]
    agg["model"] = pd.Categorical(agg["model"], categories=order, ordered=True)
    agg = agg.sort_values("model").reset_index(drop=True)

    out_csv = os.path.join(out_dir, "SUMMARY_across_datasets_mean_with_delta.csv")
    agg.to_csv(out_csv, index=False)
    print(f"\n[SAVED] {out_csv}")

    def f(x):
        if x is None or (isinstance(x, float) and (not np.isfinite(x))):
            return "  N/A"
        return f"{x:6.2f}"

    show = agg.copy()
    for c in ["video_auc_mean","d_video_auc_mean","frame_auc_mean","d_frame_auc_mean","video_acc_mean","frame_acc_mean"]:
        if c in show.columns:
            show[c] = show[c].apply(f)

    print("\n" + "=" * 120)
    print("[FINAL SUMMARY] mean performance and mean Δ(vs c23)  (FF++ excluded)")
    print("=" * 120)
    cols = [
        "model", "n_datasets",
        "video_auc_mean", "d_video_auc_mean",
        "frame_auc_mean", "d_frame_auc_mean",
        "video_acc_mean", "frame_acc_mean",
    ]
    print(show[cols].to_string(index=False))

    return agg


# ===========================================================
# Pretty table printing
# ===========================================================
def _fmt(x: Optional[float], nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (not np.isfinite(x))):
        return "  N/A"
    return f"{x:6.{nd}f}"

def print_dataset_table(ds: str, rows: List[Dict[str, Any]]):
    df = pd.DataFrame(rows)

    order = ["c23","raw","c40","all","q_routed","s_soft","ens_mix_high","ens_adapt","upper_bound"]
    order = [m for m in order if m in df["model"].unique().tolist()]
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
    df = df.sort_values("model")

    show_cols = [
        "model",
        "frame_auc","d_frame_auc_vs_c23","frame_acc",
        "video_auc","d_video_auc_vs_c23","video_acc",
        "n_valid_frames",
        "n_valid_videos",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    out = df[show_cols].copy()

    for c in ["frame_auc","d_frame_auc_vs_c23","frame_acc","video_auc","d_video_auc_vs_c23","video_acc"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: _fmt(v, 2))

    for c in ["n_valid_frames","n_valid_videos"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: f"{int(v):,}" if pd.notna(v) else "N/A")

    print("\n" + "=" * 110)
    print(f"[RESULT TABLE] {ds}")
    print("=" * 110)
    print(out.to_string(index=False))

def make_ffpp_splits(df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    """
    Split FaceForensics++ evaluation into three subsets by path quality:
      - raw, c23, c40
    Returns list of (split_name, split_df).
    """
    q = df["path"].astype(str).apply(parse_ffpp_quality)
    out = []
    for tag in ["raw", "c23", "c40"]:
        d = df[q == tag].copy()
        out.append((f"FaceForensics++({tag})", d))
    return out

def build_plot_series_from_df_all(
    df_all: pd.DataFrame,
    datasets_order: List[str],
    models: List[str],
) -> Dict[str, List[Optional[float]]]:
    """
    Build plot series from df_all (which stores percent-scale video_auc).
    Returns {model: [video_auc for ds1, ds2, ...]} aligned with datasets_order.
    """
    out: Dict[str, List[Optional[float]]] = {m: [] for m in models}

    # pivot: rows=dataset, cols=model, values=video_auc
    pv = (
        df_all.pivot_table(index="dataset", columns="model", values="video_auc", aggfunc="mean")
    )

    for ds in datasets_order:
        for m in models:
            v = None
            if ds in pv.index and m in pv.columns:
                x = pv.loc[ds, m]
                if pd.notna(x) and np.isfinite(float(x)):
                    v = float(x)
            out[m].append(v)
    return out


# ===========================================================
# MAIN
# ===========================================================
def main():
    global RBF_CENTERS
    ensure_dir(OUT_DIR)

    if not HAS_SK:
        raise RuntimeError("scikit-learn not available. Please install scikit-learn.")

    if not os.path.isfile(FFPP_STD_CSV):
        raise FileNotFoundError(f"[ERROR] Missing FF++ standardized CSV: {FFPP_STD_CSV}")
    RBF_CENTERS = load_ffpp_rbf_centers_from_train(FFPP_STD_CSV)

    # Gate training (requires FF++)
    if "FaceForensics++" not in DATASETS:
        raise ValueError("Include 'FaceForensics++' in DATASETS (required for gate training).")

    df_ff = read_csv_for_dataset("FaceForensics++", MERGE_DIR)
    for c in ["path", "c23", "crfiqa", "qcn", "q1", "q2", "q3", "q4", "quadrant", "all"]:
        if c not in df_ff.columns:
            raise KeyError(f"[ERROR] FF++ merged missing '{c}'")

    df_ff = filter_ffpp_c40_only(df_ff)
    df_ff = build_df_common(df_ff, "FaceForensics++")

    s_c23_ff = pd.to_numeric(df_ff["c23"], errors="coerce").astype(float).values
    low_mask_ff = (s_c23_ff > DOWN) & (s_c23_ff < UP)

    if TRAIN_GATE_ON_HIGHCONF_ONLY:
        df_fit = df_ff[~low_mask_ff].copy()
        print(f"[Train gate] HIGH-CONF only: N={len(df_fit):,}")
    else:
        df_fit = df_ff.copy()
        print(f"[Train gate] ALL frames: N={len(df_fit):,}")

    feats_gate = ["conf", "q_std", "q_margin", "q_range"]

    if RUN_GATE_FEATURE_ANALYSIS:
        d = df_fit.dropna(subset=feats_gate + ["y_correct"]).copy()
        X = d[feats_gate].values.astype(float)
        y = d["y_correct"].values.astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=GATE_HOLDOUT, random_state=SEED,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        gate_tmp = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=400, random_state=SEED)
        gate_tmp.fit(X_tr, y_tr)
        p_te = gate_tmp.predict_proba(X_te)[:, 1]
        te_auc = roc_auc_score(y_te, p_te) * 100.0 if len(np.unique(y_te)) > 1 else float("nan")
        te_acc = accuracy_score(y_te, (p_te >= 0.5).astype(int)) * 100.0
        print("\n" + "=" * 80)
        print("[Gate quick holdout]")
        print("=" * 80)
        print(f"  feats   = {feats_gate}")
        print(f"  acc     = {te_acc:.2f}%")
        print(f"  auc     = {te_auc:.2f}%")
        print("=" * 80)

    gate_model = fit_gate_hgb(df_fit, feats_gate, seed=SEED)

    all_rows_for_csv: List[Dict[str, Any]] = []

    plot_keys = ["all", "c23", "q_routed", "ens_mix_high", "ens_adapt", "upper_bound"]
    video_auc_series_plot: Dict[str, List[Optional[float]]] = {k: [] for k in plot_keys}

    for ds in DATASETS:
        print(f"\n---- Dataset: {ds} ----")
        df0 = read_csv_for_dataset(ds, MERGE_DIR)

        for c in ["path", "c23", "crfiqa", "qcn", "q1", "q2", "q3", "q4", "quadrant", "all"]:
            if c not in df0.columns:
                raise KeyError(f"[ERROR] {ds} merged missing '{c}'")

        df0 = build_df_common(df0, ds)

        # dataset-aware video_id
        df0["video_id"] = get_video_id_series(df0, ds)

        df0 = filter_df40_video_frames(df0, ds)

        # -------------------------------------------------------
        # NEW: split FF++ evaluation by path quality: raw/c23/c40
        # -------------------------------------------------------
        if ds == "FaceForensics++":
            eval_jobs = make_ffpp_splits(df0)  # [("FaceForensics++(raw)", df_raw), ...]
            # optional: drop empty splits safely
            eval_jobs = [(name, d) for (name, d) in eval_jobs if len(d) > 0]
            if len(eval_jobs) == 0:
                print("[WARN] FaceForensics++ splits are empty. Skipping evaluation.")
                continue
        else:
            eval_jobs = [(ds, df0)]

        # Now evaluate each job (either normal dataset, or FF++ split)
        for ds_eval, df_eval in eval_jobs:
            print(f"\n[Eval] {ds_eval}  (N_frames={len(df_eval):,})")

            base_candidates = []
            if "raw" in df_eval.columns:
                base_candidates.append("raw")
            if "c40" in df_eval.columns:
                base_candidates.append("c40")
            base_candidates += ["c23", "all"]

            for col in [c for c in (base_candidates + ["q_routed", "s_soft"]) if col_exists(df_eval, c)]:
                save_nonfinite_paths(df_eval, ds_eval, OUT_DIR, col)

            # mixers
            s_c23 = pd.to_numeric(df_eval["c23"], errors="coerce").astype(float).values
            low_mask = (s_c23 > DOWN) & (s_c23 < UP)

            df_eval["ens_mix_high"] = build_mix_static(
                s_c23=s_c23,
                s_soft=df_eval["s_soft"].values.astype(float),
                low_mask=low_mask,
                lam_base=LAM_BASE
            )

            X = df_eval[feats_gate].values.astype(float)
            p_correct = gate_predict_p_correct(gate_model, X)
            df_eval["ens_adapt"], _ = build_mix_adaptive(
                s_c23=s_c23,
                s_soft=df_eval["s_soft"].values.astype(float),
                low_mask=low_mask,
                p_correct=p_correct,
                lam_base=LAM_BASE,
                lam_min=LAM_MIN,
                wrong_low=WRONG_LOW,
                wrong_high=WRONG_HIGH
            )

            ub_cols = [c for c in (base_candidates + ["q_routed", "s_soft", "ens_mix_high", "ens_adapt"]) if col_exists(df_eval, c)]
            df_eval["upper_bound"] = compute_upper_bound_scores(df_eval, cols=ub_cols)

            model_cols = []
            if "raw" in df_eval.columns:
                model_cols.append("raw")
            if "c40" in df_eval.columns:
                model_cols.append("c40")
            model_cols += ["c23", "all", "q_routed", "s_soft", "ens_mix_high", "ens_adapt", "upper_bound"]

            for col in ["ens_mix_high", "ens_adapt", "upper_bound"]:
                save_nonfinite_paths(df_eval, ds_eval, OUT_DIR, col)

            frame_m = {m: metric_frame_auc_acc(df_eval, m) for m in model_cols}
            c23_frame_auc_pct = None if frame_m["c23"]["auc"] is None else frame_m["c23"]["auc"] * 100.0
            c23_frame_acc_pct = None if frame_m["c23"]["acc"] is None else frame_m["c23"]["acc"] * 100.0

            # STRICT video
            df_strict, mixed_info = strict_video_filter_once(
                df=df_eval,
                ds=ds_eval,
                out_dir=OUT_DIR,
                video_id_col="video_id",
                save_examples=SAVE_MIXED_LABEL_EXAMPLES,
                save_all_mixed_paths_txt=SAVE_MIXED_LABEL_PATHS_TXT,
                max_mixed_paths=MAX_MIXED_PATHS,
            )

            video_m = {m: compute_video_metrics_strict_from_frames(df_strict, m, video_id_col="video_id") for m in model_cols}
            c23_video_auc_pct = None if video_m["c23"]["auc"] is None else video_m["c23"]["auc"] * 100.0
            c23_video_acc_pct = None if video_m["c23"]["acc"] is None else video_m["c23"]["acc"] * 100.0

            # plot series: 원래 plot은 FF++를 안 그리므로 (아래 MAKE_PLOTS에서 DATASETS 재정의),
            # 여기서는 ds_eval이 FF++ split인 경우 plot series에 굳이 append하지 않아도 됨.
            # 다만 현재 구조를 유지하려면, ds_eval이 원 DATASETS에 포함될 때만 append하도록 방어.
            if ds_eval in DATASETS:
                for k in plot_keys:
                    if k in video_m:
                        v = video_m[k]["auc"]
                        video_auc_series_plot[k].append(None if v is None else float(v) * 100.0)
                    else:
                        video_auc_series_plot[k].append(None)

            rows_for_print = []
            for m in model_cols:
                fa_pct = None if frame_m[m]["auc"] is None else frame_m[m]["auc"] * 100.0
                va_pct = None if video_m[m]["auc"] is None else video_m[m]["auc"] * 100.0
                facc_pct = None if frame_m[m]["acc"] is None else frame_m[m]["acc"] * 100.0
                vacc_pct = None if video_m[m]["acc"] is None else video_m[m]["acc"] * 100.0

                d_f = None if (fa_pct is None or c23_frame_auc_pct is None) else float(fa_pct - c23_frame_auc_pct)
                d_v = None if (va_pct is None or c23_video_auc_pct is None) else float(va_pct - c23_video_auc_pct)

                rows_for_print.append({
                    "model": m,
                    "frame_auc": fa_pct,
                    "d_frame_auc_vs_c23": d_f,
                    "frame_acc": facc_pct,
                    "video_auc": va_pct,
                    "d_video_auc_vs_c23": d_v,
                    "video_acc": vacc_pct,
                    "n_valid_frames": frame_m[m]["n_valid"],
                    "n_valid_videos": video_m[m]["n_valid_videos"],
                })

                all_rows_for_csv.append({
                    "dataset": ds_eval,   # 중요: split 이름으로 저장
                    "model": m,
                    "gate_feats": ",".join(feats_gate),
                    **mixed_info,
                    "frame_auc": fa_pct,
                    "frame_acc": facc_pct,
                    "video_auc": va_pct,
                    "video_acc": vacc_pct,
                    "d_frame_auc_vs_c23": d_f,
                    "d_video_auc_vs_c23": d_v,
                    "c23_frame_auc": c23_frame_auc_pct,
                    "c23_video_auc": c23_video_auc_pct,
                    "c23_frame_acc": c23_frame_acc_pct,
                    "c23_video_acc": c23_video_acc_pct,
                    "n_valid_frames": frame_m[m]["n_valid"],
                    "n_total_frames": frame_m[m]["n_total"],
                    "n_valid_videos": video_m[m]["n_valid_videos"],
                    "n_total_videos": video_m[m]["n_total_videos"],
                })

            print_dataset_table(ds_eval, rows_for_print)
            print(f"\n[MIXED SUMMARY] {ds_eval}: mixed_videos={mixed_info['mixed_videos']:,} "
                f"({mixed_info['mixed_ratio_pct']:.4f}%), strict_videos={mixed_info['strict_videos']:,}, "
                f"strict_frames={mixed_info['strict_frames']:,}")

    out_csv = os.path.join(OUT_DIR, "ALL_results_frame_video_auc_with_deltas.csv")
    df_all = pd.DataFrame(all_rows_for_csv)
    df_all.to_csv(out_csv, index=False)
    print(f"\n[SAVED] {out_csv}")

    # summarize_across_datasets(df_all[["dataset","model","frame_auc","frame_acc","video_auc","video_acc"]], OUT_DIR)
    summarize_across_datasets(
        df_all[[
            "dataset","model",
            "frame_auc","frame_acc","video_auc","video_acc",
            "d_frame_auc_vs_c23","d_video_auc_vs_c23"
        ]],
        OUT_DIR
    )


    if MAKE_PLOTS:
        # Cross-dataset only (FF++ 제외)
        PLOT_DATASETS = [
            "Celeb-DF-v1",
            "Celeb-DF-v2",
            "Celeb-DF-v3",
            "DFDC",
            "DFo",
            "Wilddeepfake",
            "FFIW",
            "KoDF",
            "DF40",
            "DF-eval",
        ]

        # 플롯에 포함할 모델들
        plot_keys = ["all", "c23", "q_routed", "s_soft", "ens_mix_high", "ens_adapt", "upper_bound"]

        # df_all에서 플롯 시리즈를 재구성 (정렬/누락 안전)
        series_for_plot = build_plot_series_from_df_all(df_all, PLOT_DATASETS, plot_keys)

        plot_video_auc_strict(
            PLOT_DATASETS,
            series_for_plot,
            OUT_DIR,
            fname="video_auc_strict_table_run.png"
        )


    print("\n===== Done =====")
    print(f"OUT_DIR: {OUT_DIR}")

# ===========================================================
# Entry
# ===========================================================
if __name__ == "__main__":
    DATASETS = [
        "FaceForensics++",
        "Celeb-DF-v1",
        "Celeb-DF-v2",
        "Celeb-DF-v3",
        "DFDC",
        "DFo",
        "Wilddeepfake",
        "FFIW",
        "KoDF",
        "DF40",
        "DF-eval",
    ]

    detector = "gend"  # or "gend" or "prodet"

    MERGE_DIR = (
        "/home/compu/eunbi/iqa/0.module/analysis/"
        f"qcn_crfiqa/normalize_minmax_median0_v2/merge/{detector}"
    )
    OUT_DIR = (
        "/home/compu/eunbi/iqa/0.module/analysis/"
        "qcn_crfiqa/normalize_minmax_median0_v2/"
        f"33.analyze_table_gate_BASE_Qspread/{detector}"
    )
    ensure_dir(OUT_DIR)

    SEED = 0

    DOWN = 0.05
    UP = 0.95

    RBF_SIGMA = 1.0

    LAM_BASE = 0.8
    LAM_MIN = 0.2

    WRONG_LOW = 0.05
    WRONG_HIGH = 0.40

    TRAIN_GATE_ON_HIGHCONF_ONLY = True

    RUN_GATE_FEATURE_ANALYSIS = True
    GATE_HOLDOUT = 0.25

    SAVE_MIXED_LABEL_EXAMPLES = True
    SAVE_MIXED_LABEL_PATHS_TXT = True
    MAX_MIXED_PATHS = None

    MAKE_PLOTS = True

    main()
