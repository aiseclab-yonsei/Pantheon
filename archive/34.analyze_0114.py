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
    "c23": "GenD(c23)",
    "c40": "Global(c40)",
    "all": "Global(all)",
    "s_soft": "Soft(RBF)",
    "q_routed": "Q-route",
    "ens_mix_high": "Pantheon",
    "ens_adapt": "Pantheon",
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

def display_dataset_name(ds: str) -> str:
    return {"Celeb-DF-v3": "Celeb-DF++"}.get(str(ds), str(ds))

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


def plot_video_auc_strict(
    datasets: List[str],
    series: Dict[str, List[Optional[float]]],
    out_dir: str,
    fname: str = "video_auc_strict_table_run.png",
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams.update({"font.size": 20})

    x = np.arange(len(datasets))
    fig = plt.figure(figsize=(9, 5), constrained_layout=True)
    ax = plt.gca()

    # ---- Quadrant-only band: between Q-route and Soft(RBF) ----
    if ("q_routed" in series) and ("s_soft" in series):
        a = np.array([np.nan if v is None else float(v) for v in series["q_routed"]], dtype=float)
        b = np.array([np.nan if v is None else float(v) for v in series["s_soft"]], dtype=float)

        lo = np.fmin(a, b)
        hi = np.fmax(a, b)
        valid = np.isfinite(lo) & np.isfinite(hi)

        # 밴드 표시 (alpha는 취향)
        if valid.any():
            ax.fill_between(x[valid], lo[valid], hi[valid], color="orange",alpha=0.4, label="Quadrant-only range")
        # hatch 패턴까지 주고 싶으면 아래처럼 patch로 처리하는 방법도 있는데,
        # fill_between 자체 hatch는 Matplotlib 버전에 따라 제한적이라 alpha 밴드가 제일 안전합니다.

    # ---- Lines ----
    order = [ "upper_bound", "c23","ens_adapt"]
    for k in order:
        if k not in series:
            continue
        vals = [np.nan if v is None else float(v) for v in series[k]]
        label = LABEL_MAP.get(k, k)

        if k == "upper_bound":
            ax.plot(x, vals, linewidth=4, linestyle="--", color="gray", label=label)
        elif k == "ens_adapt":
            ax.plot(x, vals, linewidth=4, color="red", label=label)
        else:
            ax.plot(x, vals, linewidth=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20)
    ax.set_ylabel("Video AUC (%)")
    ax.grid(True, linestyle="--",linewidth=1,alpha=0.5)
    # ax.legend(loc="lower left")
    ax.set_title(f"Cross-Dataset Video AUC with Pantheon(GenD)",fontsize=20)

    # ----- Legend order: Pantheon(Gate) first -----
    handles, labels = ax.get_legend_handles_labels()
    hmap = {lab: h for h, lab in zip(handles, labels)}

    desired = [
        LABEL_MAP.get("ens_adapt", "ens_adapt"),  # "Pantheon(Gate)"
        LABEL_MAP.get("c23", "c23"),              # "Global(c23)"
        "Quadrant-only range",        # band label (그대로)
        LABEL_MAP.get("upper_bound", "upper_bound"),
    ]

    # 실제 존재하는 것만 남기기
    desired = [lab for lab in desired if lab in hmap]

    ax.legend([hmap[lab] for lab in desired], desired, fontsize=20,loc="lower left")
    # ax.legend([hmap[lab] for lab in desired], desired, fontsize=20,loc="upper right")


    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")

def plot_cross_dataset_mean_video_auc_bar(
    df_all: pd.DataFrame,
    out_dir: str,
    fname: str = "cross_dataset_mean_video_auc_bar.png",
    exclude_prefixes: Tuple[str, ...] = ("FaceForensics++",),
) -> str:
    """
    Plot mean STRICT video AUC over cross-dataset test sets.
    Excludes FF++ by default, consistent with summarize_across_datasets().
    """
    need = ["dataset", "model", "video_auc"]
    miss = [c for c in need if c not in df_all.columns]
    if miss:
        raise KeyError(f"[ERROR] df_all missing columns for cross-dataset mean video AUC plot: {miss}")

    dfx = df_all.copy()
    dfx["video_auc"] = pd.to_numeric(dfx["video_auc"], errors="coerce")

    if exclude_prefixes:
        ds_series = dfx["dataset"].astype(str)
        mask_ex = np.zeros(len(dfx), dtype=bool)
        for px in exclude_prefixes:
            mask_ex |= ds_series.str.startswith(px)
        dfx = dfx[~mask_ex].copy()

    model_order = ["raw", "c23", "c40", "all", "ens_adapt"]
    display_labels = ["Effort(raw)", "Effort(c23)", "Effort(c40)", "Effort(all)", "Pantheon"]

    rows = []
    for model, label in zip(model_order, display_labels):
        g = dfx[dfx["model"].astype(str) == model]["video_auc"].dropna()
        rows.append({
            "model": model,
            "label": label,
            "mean_video_auc": float(g.mean()) if len(g) > 0 else np.nan,
            "n_datasets": int(g.shape[0]),
        })

    mean_df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "cross_dataset_mean_video_auc_bar_values.csv")
    mean_df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")

    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(figsize=(10.2, 4.4), constrained_layout=True)

    x = np.arange(len(mean_df))
    vals = mean_df["mean_video_auc"].astype(float).values
    colors = ["#4c78a8", "#72b7b2", "#4c78a8", "#4c78a8", "#e45756"]
    bars = ax.bar(x, vals, color=colors, width=0.68)

    ax.set_xticks(x)
    ax.set_xticklabels(mean_df["label"].astype(str).tolist())
    ax.set_ylabel("Average Video AUC (%)")
    ax.grid(True, axis="y", linestyle="--", linewidth=1, alpha=0.4)

    finite_vals = vals[np.isfinite(vals)]
    if len(finite_vals) > 0:
        ymin = max(0.0, float(np.nanmin(finite_vals)) - 3.0)
        ymax = min(100.0, float(np.nanmax(finite_vals)) + 2.0)
        if ymax <= ymin:
            ymin, ymax = 0.0, 100.0
        ax.set_ylim(ymin, ymax)

    for label, bar, v in zip(mean_df["label"].astype(str).tolist(), bars, vals):
        if np.isfinite(v):
            is_oracle = "Oracle" in label
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v - 0.35 if is_oracle else v,
                f"{v:.2f}",
                ha="center",
                va="top" if is_oracle else "bottom",
                fontsize=18,
                fontweight="bold",
                color="white" if is_oracle else "black",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ax.get_ylim()[0],
                "N/A",
                ha="center",
                va="bottom",
                fontsize=18,
                fontweight="bold",
            )

    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return out_path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

def plot_real_fake_quadrant_bars_from_df_b(
    df_b,
    out_dir: str,
    prefix: str = "TESTSET",
    figsize=(8, 4),
):
    plt.rcParams.update({"font.size": 22})

    saved_paths = []  # NEW

    for _, r in df_b.iterrows():
        ds = str(r["dataset"])
        q = ["Q1", "Q2", "Q3", "Q4"]
        real = [int(r.get(f"real_q{i}_frames", 0)) for i in [1, 2, 3, 4]]
        fake = [int(r.get(f"fake_q{i}_frames", 0)) for i in [1, 2, 3, 4]]

        x = np.arange(len(q))
        width = 0.35

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax = plt.gca()

        ax.bar(x - width/2, real, width, label="Real")
        ax.bar(x + width/2, fake, width, label="Fake")

        ax.set_xticks(x)
        ax.set_xticklabels(q)

        ax.set_ylabel("Frames")
        ax.set_title(display_dataset_name(ds))

        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()

        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{int(v):,}"))

        out_png = os.path.join(out_dir, f"{prefix}_bar_real_fake_quadrants__{sanitize_filename(ds)}.png")
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"[SAVED] {out_png}")
        saved_paths.append(out_png)  # NEW

    return saved_paths  # NEW
def save_montage_2x5(
    image_paths: List[str],
    out_dir: str,
    prefix: str = "TESTSET",
    cols: int = 5,
    rows: int = 2,
    out_name: str = "TESTSET_bar_real_fake_quadrants__MONTAGE_2x5.png",
):
    """
    image_paths: 길이 10(=2*5) 권장. 더 많으면 앞의 10개 사용.
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import os

    plt.rcParams.update({"font.size": 20})

    n = rows * cols
    paths = image_paths[:n]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.0, rows * 3.7), constrained_layout=True)

    # axes가 2D가 아닐 수도 있어서 정규화
    axes = np.array(axes).reshape(rows, cols)

    for i in range(n):
        ax = axes[i // cols, i % cols]
        if i < len(paths) and os.path.isfile(paths[i]):
            img = mpimg.imread(paths[i])
            ax.imshow(img)
        ax.axis("off")

    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")

def compute_frame_quadrant_error_rates(
    df_eval: pd.DataFrame,
    ds_eval: str,
    model_cols: List[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Frame/sample-level FNR/FPR per quadrant.
      - FNR = false negatives / fake frames
      - FPR = false positives / real frames
    """
    need = ["label", "quadrant"]
    miss = [c for c in need if c not in df_eval.columns]
    if miss:
        raise KeyError(f"[ERROR] {ds_eval} missing columns for quadrant FNR/FPR: {miss}")

    rows = []
    dfx = df_eval.copy()
    dfx["label"] = dfx["label"].astype(int)
    dfx["quadrant"] = dfx["quadrant"].astype(int)

    for model in model_cols:
        if model not in dfx.columns:
            continue
        score = pd.to_numeric(dfx[model], errors="coerce").astype(float)
        valid_score = np.isfinite(score.values)

        for q in [1, 2, 3, 4]:
            m = (dfx["quadrant"].values == q) & valid_score
            dq = dfx.loc[m, ["label"]].copy()
            sq = score.loc[m].astype(float).values

            y = dq["label"].astype(int).values
            pred = (sq >= threshold).astype(int)

            pos = int((y == 1).sum())
            neg = int((y == 0).sum())
            fn = int(((y == 1) & (pred == 0)).sum())
            fp = int(((y == 0) & (pred == 1)).sum())

            rows.append({
                "dataset": ds_eval,
                "model": model,
                "quadrant": q,
                "n_frames": int(len(y)),
                "n_fake": pos,
                "n_real": neg,
                "fn": fn,
                "fp": fp,
                "fnr": np.nan if pos == 0 else float(fn / pos),
                "fpr": np.nan if neg == 0 else float(fp / neg),
                "threshold": float(threshold),
            })

    return pd.DataFrame(rows)

def plot_quadrant_fnr_fpr_overlay(
    df_eval: pd.DataFrame,
    rates_df: pd.DataFrame,
    ds_eval: str,
    out_dir: str,
    models_to_show: List[str],
    prefix: str = "TESTSET",
    sample_n: int = 25000,
    seed: int = 0,
) -> Optional[str]:
    """
    Plot CR-FIQA/QCN quadrants and overlay model FNR/FPR in each quadrant.
    """
    need = ["crfiqa", "qcn", "label", "quadrant"]
    miss = [c for c in need if c not in df_eval.columns]
    if miss or len(df_eval) == 0 or len(rates_df) == 0:
        return None

    dfx = df_eval.copy()
    dfx["crfiqa"] = pd.to_numeric(dfx["crfiqa"], errors="coerce").astype(float)
    dfx["qcn"] = pd.to_numeric(dfx["qcn"], errors="coerce").astype(float)
    dfx = dfx[np.isfinite(dfx["crfiqa"].values) & np.isfinite(dfx["qcn"].values)].copy()
    if len(dfx) == 0:
        return None

    if len(dfx) > sample_n:
        dplot = dfx.sample(n=sample_n, random_state=seed)
    else:
        dplot = dfx

    fig, ax = plt.subplots(figsize=(8.5, 7.0), constrained_layout=True)

    real = dplot["label"].astype(int).values == 0
    fake = ~real
    ax.scatter(
        dplot.loc[real, "crfiqa"], dplot.loc[real, "qcn"],
        s=7, alpha=0.16, c="#2f6fbb", marker="o", linewidths=0, label="Real"
    )
    ax.scatter(
        dplot.loc[fake, "crfiqa"], dplot.loc[fake, "qcn"],
        s=7, alpha=0.16, c="#c43c35", marker="o", linewidths=0, label="Fake"
    )

    ax.axvline(0, color="black", linewidth=1.4)
    ax.axhline(0, color="black", linewidth=1.4)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_xlabel("CR-FIQA (centered)")
    ax.set_ylabel("QCN (centered)")
    ax.set_title(f"{ds_eval}: quadrant FNR/FPR")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_left = xmin + (0 - xmin) * 0.08 if xmin < 0 else xmin
    x_right = 0 + (xmax - 0) * 0.08 if xmax > 0 else xmax
    y_bottom = ymin + (0 - ymin) * 0.08 if ymin < 0 else ymin
    y_top = 0 + (ymax - 0) * 0.92 if ymax > 0 else ymax

    anchors = {
        1: (x_right, y_top, "left", "top"),
        2: (x_left, y_top, "left", "top"),
        3: (x_left, y_bottom, "left", "bottom"),
        4: (x_right, y_bottom, "left", "bottom"),
    }

    shown = [m for m in models_to_show if m in set(rates_df["model"].astype(str))]
    for q in [1, 2, 3, 4]:
        lines = [f"Q{q}"]
        for model in shown:
            r = rates_df[(rates_df["quadrant"] == q) & (rates_df["model"] == model)]
            if len(r) == 0:
                continue
            rr = r.iloc[0]
            fnr = rr["fnr"]
            fpr = rr["fpr"]
            fnr_s = "NA" if pd.isna(fnr) else f"{float(fnr) * 100:.1f}"
            fpr_s = "NA" if pd.isna(fpr) else f"{float(fpr) * 100:.1f}"
            lines.append(f"{LABEL_MAP.get(model, model)} FNR {fnr_s} / FPR {fpr_s}")
        rq = rates_df[rates_df["quadrant"] == q]
        if len(rq) > 0:
            lines.append(f"N={int(rq.iloc[0]['n_frames']):,}")

        x, y, ha, va = anchors[q]
        ax.text(
            x, y, "\n".join(lines),
            ha=ha, va=va, fontsize=11,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.84, boxstyle="round,pad=0.35")
        )

    ax.legend(loc="upper right", fontsize=12)

    out_png = os.path.join(out_dir, f"{prefix}_quadrant_fnr_fpr_overlay__{sanitize_filename(ds_eval)}.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")
    return out_png

def make_c23_vs_quadrant_model_error_table(
    qerr_df: pd.DataFrame,
    out_dir: str,
    fname: str = "FRAME_effort_c23_vs_quadrant_model_fnr_fpr.csv",
) -> pd.DataFrame:
    """
    Dataset/quadrant table comparing effort c23 against the matching
    quadrant-trained model:
      Q1: c23 vs q1
      Q2: c23 vs q2
      Q3: c23 vs q3
      Q4: c23 vs q4

    q_routed is equivalent to selecting q{quadrant} for each sample, so
    the q_routed rows are relabeled as q1~q4 in this table.
    """
    need = ["dataset", "model", "quadrant", "n_frames", "n_fake", "n_real", "fn", "fp", "fnr", "fpr"]
    miss = [c for c in need if c not in qerr_df.columns]
    if miss:
        raise KeyError(f"[ERROR] quadrant error table missing columns: {miss}")

    rows = []
    for (ds, q), g in qerr_df.groupby(["dataset", "quadrant"], sort=False):
        c23 = g[g["model"] == "c23"]
        qr = g[g["model"] == "q_routed"]
        if len(c23) == 0 or len(qr) == 0:
            continue

        c23r = c23.iloc[0]
        qrr = qr.iloc[0]
        q_int = int(q)
        q_model = f"q{q_int}"

        c23_fnr = float(c23r["fnr"]) if pd.notna(c23r["fnr"]) else np.nan
        c23_fpr = float(c23r["fpr"]) if pd.notna(c23r["fpr"]) else np.nan
        q_fnr = float(qrr["fnr"]) if pd.notna(qrr["fnr"]) else np.nan
        q_fpr = float(qrr["fpr"]) if pd.notna(qrr["fpr"]) else np.nan

        rows.append({
            "dataset": ds,
            "quadrant": f"Q{q_int}",
            "q_model": q_model,
            "n_frames": int(c23r["n_frames"]),
            "n_fake": int(c23r["n_fake"]),
            "n_real": int(c23r["n_real"]),
            "c23_fn": int(c23r["fn"]),
            "c23_fp": int(c23r["fp"]),
            "c23_fnr_pct": c23_fnr * 100.0 if np.isfinite(c23_fnr) else np.nan,
            "c23_fpr_pct": c23_fpr * 100.0 if np.isfinite(c23_fpr) else np.nan,
            "q_model_fn": int(qrr["fn"]),
            "q_model_fp": int(qrr["fp"]),
            "q_model_fnr_pct": q_fnr * 100.0 if np.isfinite(q_fnr) else np.nan,
            "q_model_fpr_pct": q_fpr * 100.0 if np.isfinite(q_fpr) else np.nan,
            "delta_fnr_pct_q_minus_c23": (q_fnr - c23_fnr) * 100.0 if np.isfinite(q_fnr) and np.isfinite(c23_fnr) else np.nan,
            "delta_fpr_pct_q_minus_c23": (q_fpr - c23_fpr) * 100.0 if np.isfinite(q_fpr) and np.isfinite(c23_fpr) else np.nan,
        })

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["quadrant_num"] = out["quadrant"].str.extract(r"Q(\d+)").astype(int)
        out = out.sort_values(["dataset", "quadrant_num"]).drop(columns=["quadrant_num"]).reset_index(drop=True)

    out_path = os.path.join(out_dir, fname)
    out.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")
    return out

def plot_c23_vs_quadrant_model_error_bars(
    comp_df: pd.DataFrame,
    out_dir: str,
    prefix: str = "FRAME_effort",
    include_ffpp: bool = True,
) -> List[str]:
    """
    Per-dataset grouped bar plots:
      - top/left panel: FNR (%)
      - bottom/right panel: FPR (%)
      - x-axis: Q1~Q4
      - bars: c23 vs matching q-model (q1/q2/q3/q4)
    """
    need = [
        "dataset", "quadrant", "q_model",
        "c23_fnr_pct", "c23_fpr_pct",
        "q_model_fnr_pct", "q_model_fpr_pct",
    ]
    miss = [c for c in need if c not in comp_df.columns]
    if miss:
        raise KeyError(f"[ERROR] comparison table missing columns for bar plots: {miss}")

    dfx = comp_df.copy()
    if not include_ffpp:
        dfx = dfx[~dfx["dataset"].astype(str).str.startswith("FaceForensics++")].copy()

    if len(dfx) == 0:
        return []

    dfx["quadrant_num"] = dfx["quadrant"].astype(str).str.extract(r"Q(\d+)").astype(int)

    saved_paths: List[str] = []
    for ds, g in dfx.groupby("dataset", sort=False):
        g = g.sort_values("quadrant_num").copy()
        q_labels = g["quadrant"].astype(str).tolist()
        q_models = g["q_model"].astype(str).tolist()

        x = np.arange(len(q_labels))
        width = 0.36

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

        panels = [
            (axes[0], "FNR (%)", "c23_fnr_pct", "q_model_fnr_pct"),
            (axes[1], "FPR (%)", "c23_fpr_pct", "q_model_fpr_pct"),
        ]

        for ax, ylabel, c23_col, q_col in panels:
            c23_vals = pd.to_numeric(g[c23_col], errors="coerce").astype(float).values
            q_vals = pd.to_numeric(g[q_col], errors="coerce").astype(float).values

            ax.bar(x - width / 2, c23_vals, width, label="c23", color="#4c78a8")
            ax.bar(x + width / 2, q_vals, width, label="q-model", color="#f58518")

            ax.set_xticks(x)
            ax.set_xticklabels(q_labels, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            ax.tick_params(axis="y", labelsize=12)

            ymax = np.nanmax(np.concatenate([c23_vals, q_vals])) if len(g) > 0 else 0.0
            if np.isfinite(ymax):
                ax.set_ylim(0, max(5.0, ymax * 1.18))

            for xi, v in zip(x - width / 2, c23_vals):
                if np.isfinite(v):
                    ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, rotation=90)
            for xi, v in zip(x + width / 2, q_vals):
                if np.isfinite(v):
                    ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, rotation=90)

        axes[0].legend(fontsize=12, loc="upper right")
        fig.suptitle(str(ds), fontsize=16)

        out_png = os.path.join(out_dir, f"{prefix}_c23_vs_quadrant_model_bar__{sanitize_filename(ds)}.png")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out_png}")
        saved_paths.append(out_png)

    return saved_paths

def plot_combined_c23_vs_quadrant_model_error_bars(
    comp_df: pd.DataFrame,
    out_dir: str,
    fname: str = "FRAME_effort_c23_vs_quadrant_model_bar__ALL_DATASETS.png",
    include_ffpp: bool = True,
    add_overall: bool = False,
    dataset_order: Optional[List[str]] = None,
    dataset_label_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    One combined 2x4 figure:
      - rows: FNR, FPR
      - columns: Q1~Q4
      - x-axis: datasets
      - bars: c23 vs matching q-model for that quadrant
    """
    need = [
        "dataset", "quadrant",
        "n_fake", "n_real",
        "c23_fn", "c23_fp", "q_model_fn", "q_model_fp",
        "c23_fnr_pct", "c23_fpr_pct",
        "q_model_fnr_pct", "q_model_fpr_pct",
    ]
    miss = [c for c in need if c not in comp_df.columns]
    if miss:
        raise KeyError(f"[ERROR] comparison table missing columns for combined bar plot: {miss}")

    dfx = comp_df.copy()
    if not include_ffpp:
        dfx = dfx[~dfx["dataset"].astype(str).str.startswith("FaceForensics++")].copy()

    if len(dfx) == 0:
        return None

    if dataset_order is None:
        dataset_order = dfx["dataset"].drop_duplicates().astype(str).tolist()
    else:
        existing = set(dfx["dataset"].astype(str).tolist())
        ordered = [x for x in dataset_order if x in existing]
        extras = [x for x in dfx["dataset"].drop_duplicates().astype(str).tolist() if x not in set(ordered)]
        dataset_order = ordered + extras
    if dataset_label_map is None:
        dataset_label_map = {}
    q_order = ["Q1", "Q2", "Q3", "Q4"]

    fig, axes = plt.subplots(2, 4, figsize=(36, 10.5), constrained_layout=True, sharex=False)
    width = 0.38

    for col_idx, q in enumerate(q_order):
        g = dfx[dfx["quadrant"].astype(str) == q].copy()
        g = g.set_index("dataset").reindex(dataset_order).reset_index()

        if add_overall:
            src = dfx[dfx["quadrant"].astype(str) == q].copy()
            n_fake = int(pd.to_numeric(src["n_fake"], errors="coerce").fillna(0).sum())
            n_real = int(pd.to_numeric(src["n_real"], errors="coerce").fillna(0).sum())
            c23_fn = int(pd.to_numeric(src["c23_fn"], errors="coerce").fillna(0).sum())
            c23_fp = int(pd.to_numeric(src["c23_fp"], errors="coerce").fillna(0).sum())
            q_fn = int(pd.to_numeric(src["q_model_fn"], errors="coerce").fillna(0).sum())
            q_fp = int(pd.to_numeric(src["q_model_fp"], errors="coerce").fillna(0).sum())
            overall = pd.DataFrame([{
                "dataset": "ALL",
                "quadrant": q,
                "n_fake": n_fake,
                "n_real": n_real,
                "c23_fn": c23_fn,
                "c23_fp": c23_fp,
                "q_model_fn": q_fn,
                "q_model_fp": q_fp,
                "c23_fnr_pct": np.nan if n_fake == 0 else (c23_fn / n_fake) * 100.0,
                "c23_fpr_pct": np.nan if n_real == 0 else (c23_fp / n_real) * 100.0,
                "q_model_fnr_pct": np.nan if n_fake == 0 else (q_fn / n_fake) * 100.0,
                "q_model_fpr_pct": np.nan if n_real == 0 else (q_fp / n_real) * 100.0,
            }])
            g = pd.concat([g, overall], ignore_index=True)

        labels = g["dataset"].astype(str).tolist()
        display_labels = [dataset_label_map.get(x, x) for x in labels]

        x = np.arange(len(labels))
        panels = [
            (axes[0, col_idx], "FNR (%)", "c23_fnr_pct", "q_model_fnr_pct"),
            (axes[1, col_idx], "FPR (%)", "c23_fpr_pct", "q_model_fpr_pct"),
        ]

        for row_idx, (ax, ylabel, c23_col, q_col) in enumerate(panels):
            c23_vals = pd.to_numeric(g[c23_col], errors="coerce").astype(float).values
            q_vals = pd.to_numeric(g[q_col], errors="coerce").astype(float).values

            ax.bar(x - width / 2, c23_vals, width, label="c23", color="#4c78a8")
            ax.bar(x + width / 2, q_vals, width, label="q-model", color="#f58518")
            ax.set_title(q if row_idx == 0 else "", fontsize=16)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            ax.tick_params(axis="y", labelsize=11)

            ymax = np.nanmax(np.concatenate([c23_vals, q_vals])) if len(g) > 0 else 0.0
            if np.isfinite(ymax):
                ax.set_ylim(0, max(5.0, ymax * 1.15))

            if add_overall and len(labels) > 1:
                ax.axvline(len(labels) - 1.5, color="black", linewidth=1.0, alpha=0.55)

            ax.set_xticks(x)
            ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=14)
    fig.suptitle("c23 vs quadrant model by dataset", fontsize=18)

    out_png = os.path.join(out_dir, fname)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")
    return out_png

def plot_overall_quadrant_error_summary_bars(
    comp_df: pd.DataFrame,
    out_dir: str,
    fname: str = "FRAME_effort_c23_vs_quadrant_model_bar__OVERALL_BY_QUADRANT.png",
    include_ffpp: bool = True,
) -> Optional[str]:
    """
    Aggregate-only weighted comparison by quadrant.
    Counts are summed over datasets first, then FNR/FPR are computed.
    """
    need = ["dataset", "quadrant", "n_fake", "n_real", "c23_fn", "c23_fp", "q_model_fn", "q_model_fp"]
    miss = [c for c in need if c not in comp_df.columns]
    if miss:
        raise KeyError(f"[ERROR] comparison table missing columns for overall bar plot: {miss}")

    dfx = comp_df.copy()
    if not include_ffpp:
        dfx = dfx[~dfx["dataset"].astype(str).str.startswith("FaceForensics++")].copy()
    if len(dfx) == 0:
        return None

    rows = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        g = dfx[dfx["quadrant"].astype(str) == q].copy()
        n_fake = int(pd.to_numeric(g["n_fake"], errors="coerce").fillna(0).sum())
        n_real = int(pd.to_numeric(g["n_real"], errors="coerce").fillna(0).sum())
        c23_fn = int(pd.to_numeric(g["c23_fn"], errors="coerce").fillna(0).sum())
        c23_fp = int(pd.to_numeric(g["c23_fp"], errors="coerce").fillna(0).sum())
        q_fn = int(pd.to_numeric(g["q_model_fn"], errors="coerce").fillna(0).sum())
        q_fp = int(pd.to_numeric(g["q_model_fp"], errors="coerce").fillna(0).sum())
        rows.append({
            "quadrant": q,
            "c23_fnr_pct": np.nan if n_fake == 0 else (c23_fn / n_fake) * 100.0,
            "q_model_fnr_pct": np.nan if n_fake == 0 else (q_fn / n_fake) * 100.0,
            "c23_fpr_pct": np.nan if n_real == 0 else (c23_fp / n_real) * 100.0,
            "q_model_fpr_pct": np.nan if n_real == 0 else (q_fp / n_real) * 100.0,
        })

    agg = pd.DataFrame(rows)
    x = np.arange(len(agg))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    panels = [
        (axes[0], "FNR (%)", "c23_fnr_pct", "q_model_fnr_pct"),
        (axes[1], "FPR (%)", "c23_fpr_pct", "q_model_fpr_pct"),
    ]
    for ax, ylabel, c23_col, q_col in panels:
        c23_vals = agg[c23_col].astype(float).values
        q_vals = agg[q_col].astype(float).values
        ax.bar(x - width / 2, c23_vals, width, label="c23", color="#4c78a8")
        ax.bar(x + width / 2, q_vals, width, label="q-model", color="#f58518")
        ax.set_xticks(x)
        ax.set_xticklabels(agg["quadrant"].astype(str).tolist(), fontsize=12)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ymax = np.nanmax(np.concatenate([c23_vals, q_vals]))
        if np.isfinite(ymax):
            ax.set_ylim(0, max(5.0, ymax * 1.15))
        for xi, v in zip(x - width / 2, c23_vals):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
        for xi, v in zip(x + width / 2, q_vals):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    axes[0].legend(fontsize=12)
    fig.suptitle("Overall weighted error rate by quadrant", fontsize=16)

    out_png = os.path.join(out_dir, fname)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")
    return out_png

def plot_mirrored_c23_vs_quadrant_model_error_bars(
    comp_df: pd.DataFrame,
    out_dir: str,
    fname: str = "FRAME_effort_c23_vs_quadrant_model_mirror__ALL_DATASETS.png",
    include_ffpp: bool = True,
    dataset_order: Optional[List[str]] = None,
    dataset_label_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    One wide 1x4 mirrored figure:
      - columns: Q1~Q4
      - x-axis: datasets
      - bars above zero: FNR (%)
      - bars below zero: FPR (%) as negative values
      - paired bars: c23 vs matching q-model
    """
    need = [
        "dataset", "quadrant",
        "c23_fnr_pct", "c23_fpr_pct",
        "q_model_fnr_pct", "q_model_fpr_pct",
    ]
    miss = [c for c in need if c not in comp_df.columns]
    if miss:
        raise KeyError(f"[ERROR] comparison table missing columns for mirrored plot: {miss}")

    dfx = comp_df.copy()
    if not include_ffpp:
        dfx = dfx[~dfx["dataset"].astype(str).str.startswith("FaceForensics++")].copy()
    if len(dfx) == 0:
        return None

    if dataset_order is None:
        dataset_order = dfx["dataset"].drop_duplicates().astype(str).tolist()
    else:
        existing = set(dfx["dataset"].astype(str).tolist())
        ordered = [x for x in dataset_order if x in existing]
        extras = [x for x in dfx["dataset"].drop_duplicates().astype(str).tolist() if x not in set(ordered)]
        dataset_order = ordered + extras
    if dataset_label_map is None:
        dataset_label_map = {}

    # Use one symmetric y-limit across quadrants to make comparisons honest.
    vals = []
    for c in ["c23_fnr_pct", "q_model_fnr_pct", "c23_fpr_pct", "q_model_fpr_pct"]:
        vals.extend(pd.to_numeric(dfx[c], errors="coerce").astype(float).dropna().tolist())
    ymax = max(vals) if vals else 5.0
    ylim = max(5.0, float(ymax) * 1.18)

    q_order = ["Q1", "Q2", "Q3", "Q4"]
    fig, axes = plt.subplots(1, 4, figsize=(36, 7.2), constrained_layout=True, sharey=True)
    width = 0.36

    for idx, q in enumerate(q_order):
        ax = axes[idx]
        g = dfx[dfx["quadrant"].astype(str) == q].copy()
        g = g.set_index("dataset").reindex(dataset_order).reset_index()

        labels = g["dataset"].astype(str).tolist()
        display_labels = [dataset_label_map.get(x, x) for x in labels]
        x = np.arange(len(labels))

        c23_fnr = pd.to_numeric(g["c23_fnr_pct"], errors="coerce").astype(float).values
        q_fnr = pd.to_numeric(g["q_model_fnr_pct"], errors="coerce").astype(float).values
        c23_fpr = -pd.to_numeric(g["c23_fpr_pct"], errors="coerce").astype(float).values
        q_fpr = -pd.to_numeric(g["q_model_fpr_pct"], errors="coerce").astype(float).values

        ax.bar(x - width / 2, c23_fnr, width, label="c23 FNR", color="#4c78a8", alpha=0.92)
        ax.bar(x + width / 2, q_fnr, width, label="q-model FNR", color="#f58518", alpha=0.92)
        ax.bar(x - width / 2, c23_fpr, width, label="c23 FPR", color="#4c78a8", alpha=0.45)
        ax.bar(x + width / 2, q_fpr, width, label="q-model FPR", color="#f58518", alpha=0.45)

        ax.axhline(0, color="black", linewidth=1.4)
        ax.grid(True, axis="y", linestyle="--", alpha=0.30)
        ax.set_title(q, fontsize=16)
        ax.set_ylim(-ylim, ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        if idx == 0:
            ax.set_ylabel("FNR above 0 / FPR below 0 (%)", fontsize=14)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{abs(v):.0f}"))

        ax.text(
            0.01, 0.97, "FNR",
            transform=ax.transAxes, ha="left", va="top", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75)
        )
        ax.text(
            0.01, 0.03, "FPR",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75)
        )

    handles, labels = axes[0].get_legend_handles_labels()
    unique = {}
    for h, lab in zip(handles, labels):
        unique.setdefault(lab, h)
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=4, fontsize=13)
    fig.suptitle("c23 vs quadrant model: FNR/FPR mirrored by dataset", fontsize=18)

    out_png = os.path.join(out_dir, fname)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")
    return out_png


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

def _mode_int(series: pd.Series) -> int:
    """Return mode of an int series. If tie, return smallest."""
    vc = series.dropna().astype(int).value_counts()
    if len(vc) == 0:
        return -1
    top = vc[vc == vc.max()].index.tolist()
    return int(min(top))

def compute_testset_tableA_stats(
    df_eval: pd.DataFrame,
    df_strict: pd.DataFrame,
    ds_eval: str,
    mixed_info: Dict[str, Any],
    video_id_col: str = "video_id",
) -> Dict[str, Any]:
    """
    Table A용: (split 포함) dataset별
      - Video counts: total/real/fake + mixed/strict
      - Frame counts: total/real/fake + strict/real/fake
      - Quadrant dist (STRICT frames): Q1~Q4 count + % (of total STRICT frames)
      - Quality stats (STRICT frames): median/var of qcn, crfiqa
    """
    out: Dict[str, Any] = {"dataset": str(ds_eval)}

    # ---------- video counts (eval 기준) ----------
    # mixed_info는 strict_video_filter_once에서 나온 값(해당 ds_eval 기준)이므로 그대로 사용
    out["n_videos_total"]  = int(mixed_info.get("total_videos", 0))
    out["n_videos_mixed"]  = int(mixed_info.get("mixed_videos", 0))
    out["n_videos_strict"] = int(mixed_info.get("strict_videos", 0))
    out["mixed_ratio_pct"] = float(mixed_info.get("mixed_ratio_pct", 0.0))

    # STRICT video들에서 real/fake video 수
    if len(df_strict) > 0 and video_id_col in df_strict.columns and "label" in df_strict.columns:
        vd = (
            df_strict.groupby(video_id_col, as_index=False)
                     .agg(label=("label", "max"))
        )
        out["n_videos_real"] = int((vd["label"].astype(int) == 0).sum())
        out["n_videos_fake"] = int((vd["label"].astype(int) == 1).sum())
    else:
        out["n_videos_real"] = 0
        out["n_videos_fake"] = 0

    # ---------- frame counts (total eval) ----------
    out["n_frames_total"] = int(len(df_eval))
    if "label" in df_eval.columns:
        y_eval = df_eval["label"].astype(int)
        out["n_frames_real_total"] = int((y_eval == 0).sum())
        out["n_frames_fake_total"] = int((y_eval == 1).sum())
    else:
        out["n_frames_real_total"] = 0
        out["n_frames_fake_total"] = 0

    # ---------- frame counts (STRICT) ----------
    out["n_frames_strict"] = int(len(df_strict))
    if len(df_strict) > 0 and "label" in df_strict.columns:
        y_str = df_strict["label"].astype(int)
        out["n_frames_real_strict"] = int((y_str == 0).sum())
        out["n_frames_fake_strict"] = int((y_str == 1).sum())
    else:
        out["n_frames_real_strict"] = 0
        out["n_frames_fake_strict"] = 0

    # ---------- quadrant dist + quality stats (STRICT frames 기준) ----------
    # 기본값 세팅
    for q in [1, 2, 3, 4]:
        out[f"q{q}_frames"] = 0
        out[f"q{q}_pct"] = 0.0
    out["qcn_median"] = np.nan
    out["qcn_var"] = np.nan
    out["crfiqa_median"] = np.nan
    out["crfiqa_var"] = np.nan

    if len(df_strict) == 0:
        return out

    need = ["quadrant", "qcn", "crfiqa"]
    miss = [c for c in need if c not in df_strict.columns]
    if miss:
        raise KeyError(f"[ERROR] {ds_eval} df_strict missing columns: {miss}")

    dfx = df_strict.copy()
    dfx["quadrant"] = dfx["quadrant"].astype(int)

    # quality stats
    qcn = pd.to_numeric(dfx["qcn"], errors="coerce").astype(float).values
    crf = pd.to_numeric(dfx["crfiqa"], errors="coerce").astype(float).values
    qcn_f = qcn[np.isfinite(qcn)]
    crf_f = crf[np.isfinite(crf)]

    out["qcn_median"] = float(np.median(qcn_f)) if len(qcn_f) > 0 else np.nan
    out["qcn_var"]    = float(np.var(qcn_f))    if len(qcn_f) > 0 else np.nan
    out["crfiqa_median"] = float(np.median(crf_f)) if len(crf_f) > 0 else np.nan
    out["crfiqa_var"]    = float(np.var(crf_f))    if len(crf_f) > 0 else np.nan

    # quadrant dist (STRICT frames)
    total_strict = max(int(len(dfx)), 1)
    for q in [1, 2, 3, 4]:
        cnt = int((dfx["quadrant"] == q).sum())
        out[f"q{q}_frames"] = cnt
        out[f"q{q}_pct"] = (cnt / total_strict) * 100.0

    return out


def print_and_save_testset_tableA(
    rows: List[Dict[str, Any]],
    out_dir: str,
    fname: str = "TESTSET_TableA_counts_quadrants_quality.csv",
) -> pd.DataFrame:
    if len(rows) == 0:
        print("\n[TESTSET TABLE A] No rows to report.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).copy()

    # save raw
    out_csv = os.path.join(out_dir, fname)
    df.to_csv(out_csv, index=False)
    print(f"\n[SAVED] {out_csv}")

    # pretty print helpers
    def _fmt_int(x) -> str:
        if x is None or (isinstance(x, float) and (not np.isfinite(x))):
            return "N/A"
        try:
            return f"{int(x):,}"
        except Exception:
            return "N/A"

    def _fmt_pct(x) -> str:
        if x is None or (isinstance(x, float) and (not np.isfinite(x))):
            return "N/A"
        return f"{float(x):.2f}"

    def _fmt_float(x, nd=4) -> str:
        if x is None or (isinstance(x, float) and (not np.isfinite(x))):
            return "N/A"
        return f"{float(x):.{nd}f}"

    show = df.copy()

    # int_cols = [
    #     "n_videos_total","n_videos_real","n_videos_fake","n_videos_mixed","n_videos_strict",
    #     "n_frames_total","n_frames_real_total","n_frames_fake_total",
    #     "n_frames_strict","n_frames_real_strict","n_frames_fake_strict",
    #     "q1_frames","q2_frames","q3_frames","q4_frames",
    # ]
    int_cols = [
        "n_videos_total","n_videos_real","n_videos_fake",
        "n_frames_total","n_frames_real_total","n_frames_fake_total",
        "q1_frames","q2_frames","q3_frames","q4_frames",
    ]
    pct_cols = ["q1_pct","q2_pct","q3_pct","q4_pct"]
    float_cols = ["qcn_median","qcn_var","crfiqa_median","crfiqa_var"]

    for c in int_cols:
        if c in show.columns:
            show[c] = show[c].apply(_fmt_int)
    for c in pct_cols:
        if c in show.columns:
            show[c] = show[c].apply(_fmt_pct)
    for c in float_cols:
        if c in show.columns:
            show[c] = show[c].apply(lambda v: _fmt_float(v, 4))

    # column order (원하는 순서)
    cols = [
        "dataset",
        "n_videos_total","n_videos_real","n_videos_fake",
        "n_frames_total","n_frames_real_total","n_frames_fake_total",
        "q1_frames","q1_pct","q2_frames","q2_pct","q3_frames","q3_pct","q4_frames","q4_pct",
        "qcn_median","qcn_var","crfiqa_median","crfiqa_var",
    ]
    cols = [c for c in cols if c in show.columns]

    print("\n" + "=" * 200)
    print("[TESTSET TABLE A] Video/Frame counts + Quadrant distribution (STRICT frames) + Quality stats (STRICT frames)")
    print("=" * 200)
    print(show[cols].to_string(index=False))

    return df



def compute_testset_frame_quadrant_stats(
    df_eval: pd.DataFrame,
    df_strict: pd.DataFrame,
    ds_eval: str,
    mixed_info: Dict[str, Any],
    video_id_col: str = "video_id",
) -> Dict[str, Any]:
    """
    - video counts: total/mixed/strict (mixed_info 활용)
    - frame counts: total(eval)/strict
    - quadrant distribution: STRICT frames 기준 count + percent
    - real/fake quadrant counts: STRICT frames 기준 (표2/plot용)
    - quality stats: STRICT frames 기준 median/var
    """
    out: Dict[str, Any] = {"dataset": ds_eval}

    # --- counts: videos ---
    out["n_videos_total"] = int(mixed_info.get("total_videos", 0))
    out["n_videos_mixed"] = int(mixed_info.get("mixed_videos", 0))
    out["n_videos_strict"] = int(mixed_info.get("strict_videos", 0))
    out["mixed_ratio_pct"] = float(mixed_info.get("mixed_ratio_pct", 0.0))

    # --- counts: frames ---
    out["n_frames_total"] = int(len(df_eval))
    out["n_frames_strict"] = int(len(df_strict))

    # safety
    if len(df_strict) == 0:
        # quadrant (overall)
        for q in [1, 2, 3, 4]:
            out[f"q{q}_frames"] = 0
            out[f"q{q}_pct"] = 0.0
            out[f"real_q{q}_frames"] = 0
            out[f"fake_q{q}_frames"] = 0
        out["real_frames"] = 0
        out["fake_frames"] = 0
        out["qcn_median"] = np.nan
        out["qcn_var"] = np.nan
        out["crfiqa_median"] = np.nan
        out["crfiqa_var"] = np.nan
        return out

    need = ["label", "quadrant", "qcn", "crfiqa"]
    miss = [c for c in need if c not in df_strict.columns]
    if miss:
        raise KeyError(f"[ERROR] {ds_eval} df_strict missing columns: {miss}")

    dfx = df_strict.copy()
    dfx["label"] = dfx["label"].astype(int)
    dfx["quadrant"] = dfx["quadrant"].astype(int)

    # --- quality stats on STRICT frames ---
    qcn = pd.to_numeric(dfx["qcn"], errors="coerce").astype(float).values
    crf = pd.to_numeric(dfx["crfiqa"], errors="coerce").astype(float).values
    qcn_f = qcn[np.isfinite(qcn)]
    crf_f = crf[np.isfinite(crf)]

    out["qcn_median"] = float(np.median(qcn_f)) if len(qcn_f) > 0 else np.nan
    out["qcn_var"] = float(np.var(qcn_f)) if len(qcn_f) > 0 else np.nan
    out["crfiqa_median"] = float(np.median(crf_f)) if len(crf_f) > 0 else np.nan
    out["crfiqa_var"] = float(np.var(crf_f)) if len(crf_f) > 0 else np.nan

    # --- quadrant distribution on STRICT frames (overall) ---
    total_strict = max(int(len(dfx)), 1)
    for q in [1, 2, 3, 4]:
        cnt = int((dfx["quadrant"] == q).sum())
        out[f"q{q}_frames"] = cnt
        out[f"q{q}_pct"] = (cnt / total_strict) * 100.0

    # --- real/fake counts on STRICT frames (for separate table + plot) ---
    out["real_frames"] = int((dfx["label"] == 0).sum())
    out["fake_frames"] = int((dfx["label"] == 1).sum())
    for q in [1, 2, 3, 4]:
        out[f"real_q{q}_frames"] = int(((dfx["label"] == 0) & (dfx["quadrant"] == q)).sum())
        out[f"fake_q{q}_frames"] = int(((dfx["label"] == 1) & (dfx["quadrant"] == q)).sum())

    return out


def print_and_save_testset_tables_and_plots(
    rows: List[Dict[str, Any]],
    out_dir: str,
    prefix: str = "TESTSET",
) -> None:
    """
    rows: dataset별 compute_testset_frame_quadrant_stats() 결과 리스트
    - Table A: video/frame counts + quadrant distribution (overall, %)
    - Table B: real/fake quadrant frame counts
    - Plot: per dataset, Q1~Q4 real/fake grouped bar
    """
    if len(rows) == 0:
        print("\n[TESTSET DISTRIBUTION] No rows to report.")
        return

    df = pd.DataFrame(rows).copy()

    # ---------- Table A ----------
    table_a_cols = [
        "dataset",
        "n_videos_total", "n_videos_mixed", "n_videos_strict", "mixed_ratio_pct",
        "n_frames_total", "n_frames_strict",
        "q1_frames", "q1_pct", "q2_frames", "q2_pct", "q3_frames", "q3_pct", "q4_frames", "q4_pct",
        "qcn_median", "qcn_var", "crfiqa_median", "crfiqa_var",
    ]
    table_a_cols = [c for c in table_a_cols if c in df.columns]
    df_a = df[table_a_cols].copy()

    out_a = os.path.join(out_dir, f"{prefix}_A_overall_quadrant_distribution_frames.csv")
    df_a.to_csv(out_a, index=False)
    print(f"\n[SAVED] {out_a}")

    # pretty print A
    show_a = df_a.copy()
    for c in ["mixed_ratio_pct", "q1_pct", "q2_pct", "q3_pct", "q4_pct"]:
        if c in show_a.columns:
            show_a[c] = show_a[c].apply(lambda v: "  N/A" if (v is None or not np.isfinite(float(v))) else f"{float(v):7.2f}")
    for c in ["qcn_median", "qcn_var", "crfiqa_median", "crfiqa_var"]:
        if c in show_a.columns:
            show_a[c] = show_a[c].apply(lambda v: "  N/A" if (v is None or not np.isfinite(float(v))) else f"{float(v):9.4f}")

    print("\n" + "=" * 160)
    print("[TESTSET TABLE A] video/frame counts + quadrant distribution (STRICT frames) + quality stats (STRICT frames)")
    print("=" * 160)
    print(show_a.to_string(index=False))

    # ---------- Table B ----------
    table_b_cols = [
        "dataset",
        "real_frames", "fake_frames",
        "real_q1_frames", "real_q2_frames", "real_q3_frames", "real_q4_frames",
        "fake_q1_frames", "fake_q2_frames", "fake_q3_frames", "fake_q4_frames",
    ]
    table_b_cols = [c for c in table_b_cols if c in df.columns]
    df_b = df[table_b_cols].copy()

    out_b = os.path.join(out_dir, f"{prefix}_B_real_fake_quadrant_frames.csv")
    df_b.to_csv(out_b, index=False)
    print(f"\n[SAVED] {out_b}")

    print("\n" + "=" * 160)
    print("[TESTSET TABLE B] real/fake quadrant frame counts (STRICT frames)")
    print("=" * 160)
    print(df_b.to_string(index=False))

    # ---------- Plots (per dataset) ----------
    from matplotlib.ticker import FuncFormatter

    # ---------- Plots (per dataset) ----------
    for _, r in df_b.iterrows():
        ds = str(r["dataset"])
        q = ["Q1", "Q2", "Q3", "Q4"]
        real = [int(r.get(f"real_q{i}_frames", 0)) for i in [1, 2, 3, 4]]
        fake = [int(r.get(f"fake_q{i}_frames", 0)) for i in [1, 2, 3, 4]]

        x = np.arange(len(q))
        width = 0.35

        fig, ax = plt.subplots(figsize=(5, 3))

        ax.bar(x - width/2, real, width, label="Real")
        ax.bar(x + width/2, fake, width, label="Fake")

        # ---- font size = 20 everywhere ----
        ax.set_xticks(x)
        ax.set_xticklabels(q, fontsize=20)
        ax.tick_params(axis="y", labelsize=20)
        ax.set_ylabel("Frames", fontsize=20)
        ax.set_title(display_dataset_name(ds), fontsize=20)
        ax.legend(fontsize=20)

        # ---- 핵심: 1e6 같은 scientific/offset 완전 제거 + 콤마 ----
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.get_offset_text().set_visible(False)  # 혹시 남아있으면 강제 숨김
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v):,}"))

        ax.grid(True, linestyle="--", alpha=0.35)

        # tight_layout로 잘리는 경우가 있어 저장 시 bbox_inches 권장
        fig.tight_layout()
        out_png = os.path.join(out_dir, f"{prefix}_bar_real_fake_quadrants__{sanitize_filename(ds)}.png")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"[SAVED] {out_png}")
        # ---------- Plots (per dataset) ----------
    # 기존: for loop로 dataset별 저장
    # 변경: 위에서 반환받은 saved_paths로 montage까지 저장

    saved_paths = plot_real_fake_quadrant_bars_from_df_b(
        df_b=df_b,
        out_dir=out_dir,
        prefix=prefix,
    )

    # NEW: 2x5 montage 한 장 생성
    save_montage_2x5(
        image_paths=saved_paths,
        out_dir=out_dir,
        prefix=prefix,
        cols=5,
        rows=2,
        out_name=f"{prefix}_bar_real_fake_quadrants__MONTAGE_2x5.png",
    )



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

def save_ensadapt_gate_csv(
    df_eval: pd.DataFrame,
    ds_eval: str,
    out_dir: str,
    detector: str,
    tag: str = "frames",
) -> str:
    """
    Save per-frame gate diagnostics + ens_adapt score for a dataset (and detector).
    Ensures ens_adapt column is the last column.
    """
    save_dir = os.path.join(out_dir, "ensadapt_csv")
    ensure_dir(save_dir)

    # 컬럼이 없으면 있는 것만 저장 (안전)
    preferred_cols = [
        "__dataset__",  # 있으면
        "path",
        "video_id",
        "label",

        # base scores
        "c23",
        "all",
        "q_routed",
        "s_soft",
        "ens_mix_high",

        # gate features / diag
        "conf",
        "q_std",
        "q_margin",
        "q_range",

        "gate_p_correct",
        "gate_p_wrong",
        "gate_pred_correct",   # (p_correct>=0.5)
        "gate_pred_wrong",     # (p_wrong>=0.5)
        "gate_lambda",         # piecewise lambda
        "low_mask",            # 1이면 (DOWN,UP) 사이 -> lowconf (soft-only)
        "y_correct",           # c23 기준 정답 여부 (이미 add_gate_features에서 생성)

        # 마지막 컬럼으로 강제
        "ens_adapt",
    ]

    cols = [c for c in preferred_cols if c in df_eval.columns]

    # ens_adapt는 "마지막" 보장
    if "ens_adapt" in cols:
        cols = [c for c in cols if c != "ens_adapt"] + ["ens_adapt"]

    out_name = f"ensadapt_gate__{sanitize_filename(detector)}__{sanitize_filename(ds_eval)}__{tag}.csv"
    out_path = os.path.join(save_dir, out_name)

    df_eval.loc[:, cols].to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}  (rows={len(df_eval):,}, cols={len(cols)})")
    return out_path


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
    testset_dist_rows: List[Dict[str, Any]] = []
    testset_dist_seen: set = set()
    testset_rows: List[Dict[str, Any]] = []
    quadrant_error_rows: List[pd.DataFrame] = []



    plot_keys = ["all", "c23", "q_routed", "ens_mix_high", "ens_adapt", "upper_bound"]
    video_auc_series_plot: Dict[str, List[Optional[float]]] = {k: [] for k in plot_keys}

    for ds in DATASETS:
        print(f"\n---- Dataset: {ds} ----")
        df0 = read_csv_for_dataset(ds, MERGE_DIR)

        for c in ["path", "c23", "crfiqa", "qcn", "q1", "q2", "q3", "q4", "quadrant"]:
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
            base_candidates.append("c23")
            if "all" in df_eval.columns:
                base_candidates.append("all")

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

            # X = df_eval[feats_gate].values.astype(float)
            # p_correct = gate_predict_p_correct(gate_model, X)
            # df_eval["ens_adapt"], _ = build_mix_adaptive(
            #     s_c23=s_c23,
            #     s_soft=df_eval["s_soft"].values.astype(float),
            #     low_mask=low_mask,
            #     p_correct=p_correct,
            #     lam_base=LAM_BASE,
            #     lam_min=LAM_MIN,
            #     wrong_low=WRONG_LOW,
            #     wrong_high=WRONG_HIGH
            # )
            X = df_eval[feats_gate].values.astype(float)
            p_correct = gate_predict_p_correct(gate_model, X)

            # NOTE: lam_all도 같이 받기
            df_eval["ens_adapt"], lam_all = build_mix_adaptive(
                s_c23=s_c23,
                s_soft=df_eval["s_soft"].values.astype(float),
                low_mask=low_mask,
                p_correct=p_correct,
                lam_base=LAM_BASE,
                lam_min=LAM_MIN,
                wrong_low=WRONG_LOW,
                wrong_high=WRONG_HIGH
            )

            # ---- gate diagnostics columns ----
            df_eval["gate_p_correct"] = p_correct.astype(float)
            df_eval["gate_p_wrong"] = (1.0 - p_correct).astype(float)
            df_eval["gate_pred_correct"] = (df_eval["gate_p_correct"].values >= 0.5).astype(int)
            df_eval["gate_pred_wrong"] = (df_eval["gate_p_wrong"].values >= 0.5).astype(int)
            df_eval["gate_lambda"] = lam_all.astype(float)
            df_eval["low_mask"] = low_mask.astype(int)  # 1이면 lowconf 구간(soft-only), 0이면 gate-mix 구간

            ub_cols = [c for c in (base_candidates + ["q_routed", "s_soft", "ens_mix_high", "ens_adapt"]) if col_exists(df_eval, c)]
            df_eval["upper_bound"] = compute_upper_bound_scores(df_eval, cols=ub_cols)

            model_cols = []
            for col in ["c23", "raw", "c40", "all", "q_routed", "s_soft", "ens_mix_high", "ens_adapt", "upper_bound"]:
                if col_exists(df_eval, col):
                    model_cols.append(col)

            quadrant_models = [m for m in ["c23", "all", "q_routed", "s_soft", "ens_adapt"] if m in model_cols]
            qerr = compute_frame_quadrant_error_rates(
                df_eval=df_eval,
                ds_eval=ds_eval,
                model_cols=quadrant_models,
                threshold=0.5,
            )
            quadrant_error_rows.append(qerr)
            plot_quadrant_fnr_fpr_overlay(
                df_eval=df_eval,
                rates_df=qerr,
                ds_eval=ds_eval,
                out_dir=OUT_DIR,
                models_to_show=[m for m in ["c23", "q_routed", "s_soft", "ens_adapt"] if m in quadrant_models],
                prefix="FRAME",
                sample_n=25000,
                seed=SEED,
            )

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

            # ---------------------------
            # NEW: Save per-dataset ens_adapt + gate outputs (FULL frames)
            # ---------------------------
            save_ensadapt_gate_csv(
                df_eval=df_eval,
                ds_eval=ds_eval,
                out_dir=OUT_DIR,
                detector=detector,
                tag="frames_all"
            )
            # ---------------------------
            # NEW (optional): Save STRICT-only frames too
            # ---------------------------
            save_ensadapt_gate_csv(
                df_eval=df_strict,
                ds_eval=ds_eval,
                out_dir=OUT_DIR,
                detector=detector,
                tag="frames_strict"
            )

            # -------------------------------------------------------
            # NEW: per-dataset testset stats (STRICT frames), skip FF++ splits
            # -------------------------------------------------------
            # video metrics (STRICT)  <-- 이 블록이 지금 코드에 빠져있음
            video_m = {
                m: compute_video_metrics_strict_from_frames(df_strict, m, video_id_col="video_id")
                for m in model_cols
            }
            c23_video_auc_pct = None if video_m["c23"]["auc"] is None else video_m["c23"]["auc"] * 100.0
            c23_video_acc_pct = None if video_m["c23"]["acc"] is None else video_m["c23"]["acc"] * 100.0

            if (not str(ds_eval).startswith("FaceForensics++")) and (ds_eval not in testset_dist_seen):
                row = compute_testset_frame_quadrant_stats(
                    df_eval=df_eval,
                    df_strict=df_strict,
                    ds_eval=ds_eval,
                    mixed_info=mixed_info,
                    video_id_col="video_id",
                )
                testset_dist_rows.append(row)
                testset_dist_seen.add(ds_eval)

                        # df_strict, mixed_info 만든 직후
            row = compute_testset_tableA_stats(
                df_eval=df_eval,
                df_strict=df_strict,
                ds_eval=ds_eval,          # FaceForensics++(raw/c23/c40) 포함
                mixed_info=mixed_info,
                video_id_col="video_id",
            )
            testset_rows.append(row)


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

    if len(quadrant_error_rows) > 0:
        df_qerr = pd.concat(quadrant_error_rows, ignore_index=True)
        out_qerr = os.path.join(OUT_DIR, "FRAME_quadrant_fnr_fpr_by_model.csv")
        df_qerr.to_csv(out_qerr, index=False)
        print(f"[SAVED] {out_qerr}")
        df_comp = make_c23_vs_quadrant_model_error_table(
            qerr_df=df_qerr,
            out_dir=OUT_DIR,
            fname="FRAME_effort_c23_vs_quadrant_model_fnr_fpr.csv",
        )
        plot_c23_vs_quadrant_model_error_bars(
            comp_df=df_comp,
            out_dir=OUT_DIR,
            prefix="FRAME_effort",
            include_ffpp=True,
        )
        ordered_datasets_for_error_plot = [
            "FaceForensics++(raw)",
            "FaceForensics++(c23)",
            "FaceForensics++(c40)",
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
        dataset_labels_for_error_plot = {
            "FaceForensics++(raw)": "FF+(raw)",
            "FaceForensics++(c23)": "FF+(c23)",
            "FaceForensics++(c40)": "FF+(c40)",
            "Celeb-DF-v1": "CDFv1",
            "Celeb-DF-v2": "CDFv2",
            "Celeb-DF-v3": "CDFv3",
            "DFDC": "DFDC",
            "DFo": "DFO",
            "Wilddeepfake": "WDF",
            "FFIW": "FFIW",
            "KoDF": "KoDF",
            "DF40": "DF40",
            "DF-eval": "DF-Eval",
        }
        plot_combined_c23_vs_quadrant_model_error_bars(
            comp_df=df_comp,
            out_dir=OUT_DIR,
            fname="FRAME_effort_c23_vs_quadrant_model_bar__ALL_DATASETS.png",
            include_ffpp=True,
            add_overall=False,
            dataset_order=ordered_datasets_for_error_plot,
            dataset_label_map=dataset_labels_for_error_plot,
        )
        plot_mirrored_c23_vs_quadrant_model_error_bars(
            comp_df=df_comp,
            out_dir=OUT_DIR,
            fname="FRAME_effort_c23_vs_quadrant_model_mirror__ALL_DATASETS.png",
            include_ffpp=True,
            dataset_order=ordered_datasets_for_error_plot,
            dataset_label_map=dataset_labels_for_error_plot,
        )

    # summarize_across_datasets(df_all[["dataset","model","frame_auc","frame_acc","video_auc","video_acc"]], OUT_DIR)
    summarize_across_datasets(
        df_all[[
            "dataset","model",
            "frame_auc","frame_acc","video_auc","video_acc",
            "d_frame_auc_vs_c23","d_video_auc_vs_c23"
        ]],
        OUT_DIR
    )
    plot_cross_dataset_mean_video_auc_bar(
        df_all=df_all,
        out_dir=OUT_DIR,
        fname="cross_dataset_mean_video_auc_bar.png",
        exclude_prefixes=("FaceForensics++",),
    )
    # -------------------------------------------------------
    # NEW: testset distribution tables + plots AFTER final summary
    # -------------------------------------------------------
    print_and_save_testset_tables_and_plots(
        rows=testset_dist_rows,
        out_dir=OUT_DIR,
        prefix="TESTSET",
    )

    print_and_save_testset_tableA(
    rows=testset_rows,
    out_dir=OUT_DIR,
    fname="TESTSET_TableA_counts_quadrants_quality.csv",
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
            "Celeb-DF-v2_Low",
            "Celeb-DF-v2_Medium",
            "FaceForensics++_raw_Medium",
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

    # DATASETS = [
    #     "FaceForensics++", "FaceForensics++_raw_Medium",

    # ]
    detector = "effort"  # or "gend" or "prodet"

    MERGE_DIR = (
        "/home/compu/eunbi/iqa/0.module/analysis/"
        f"qcn_crfiqa/normalize_minmax_median0_v2/merge/{detector}"
    )
    OUT_DIR = (
        "/home/compu/eunbi/iqa/0.module/analysis/"
        "qcn_crfiqa/normalize_minmax_median0_v2/"
        f"34.analyze_table_gate_BASE_Qspread/{detector}"
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
