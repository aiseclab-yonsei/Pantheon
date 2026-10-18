"""Numerical core extracted from 34.analyze_0114.py; original formulas preserved."""
import os
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
RBF_CENTERS = None
RBF_SIGMA = 1.0
_FRAMES_RE = re.compile(r"/frames/([^/]+)/[^/]+$")
DF40_NONVIDEO_FOLDERS = ["DiT", "pixart", "RDDM", "sd2.1", "SiT", "StyleGAN2", "StyleGAN3", "StyleGANXL", "VQGAN"]
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def display_dataset_name(ds: str) -> str:
    return {'Celeb-DF-v3': 'Celeb-DF++'}.get(str(ds), str(ds))

def sanitize_filename(s: str) -> str:
    s = str(s)
    s = s.replace('/', '_').replace('\\', '_').replace('|', '_').replace(':', '_')
    s = s.replace(' ', '_')
    return s

def read_csv_for_dataset(ds: str, merge_dir: str) -> pd.DataFrame:
    path = os.path.join(merge_dir, f'merged_{ds}.csv')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'[ERROR] Missing merged csv: {path}')
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        raise KeyError(f"[ERROR] {path} missing 'label'")
    df['label'] = df['label'].astype(int)
    if 'quadrant' in df.columns:
        df['quadrant'] = df['quadrant'].astype(int)
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
        return {'auc': None, 'acc': None, 'n_valid': 0, 'n_total': int(len(df))}
    y = df['label'].astype(int).values
    s = pd.to_numeric(df[score_col], errors='coerce').astype(float).values
    m = _finite_mask(s)
    auc = safe_auc(y, s)
    acc = safe_acc(y, s)
    return {'auc': auc, 'acc': acc, 'n_valid': int(m.sum()), 'n_total': int(len(s))}

def save_nonfinite_paths(df: pd.DataFrame, ds: str, out_dir: str, score_col: str) -> int:
    if score_col not in df.columns:
        return 0
    s = pd.to_numeric(df[score_col], errors='coerce').astype(float)
    bad = ~np.isfinite(s.values)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return 0
    txt_path = os.path.join(out_dir, f'nonfinite_paths_{sanitize_filename(ds)}__{sanitize_filename(score_col)}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        for p in df.loc[bad, 'path'].astype(str).tolist():
            f.write(p + '\n')
    csv_path = os.path.join(out_dir, f'nonfinite_rows_{sanitize_filename(ds)}__{sanitize_filename(score_col)}.csv')
    df.loc[bad, ['path', 'label', score_col]].to_csv(csv_path, index=False)
    print(f'[NON-FINITE] {ds} | {score_col}: n_bad={n_bad:,} -> saved')
    return n_bad

def col_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def _find_last_anchor(parts: List[str], anchors=('real', 'fake')) -> Optional[Tuple[str, int]]:
    low = [x.lower() for x in parts]
    for i in range(len(low) - 1, -1, -1):
        if low[i] in anchors:
            return (low[i], i)
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
    s = str(p).replace('\\', '/')
    ds_l = str(ds).lower()
    parts = [x for x in s.split('/') if x != '']
    dirs = parts[:-1] if len(parts) >= 2 else parts
    if ds_l == 'df40':
        if len(dirs) >= 2:
            return '/'.join(dirs[-2:])
        if len(dirs) == 1:
            return dirs[-1]
        return s
    if 'wild' in ds_l and 'deepfake' in ds_l:
        anchor = _find_last_anchor(dirs, anchors=('real', 'fake'))
        if anchor is not None:
            _, idx = anchor
            st = max(0, idx - 2)
            ed = min(len(dirs), idx + 2)
            return '/'.join(dirs[st:ed])
        if len(dirs) >= 2:
            return '/'.join(dirs[-2:])
        return s
    if ds_l == 'ffiw':
        video_name = dirs[-1] if len(dirs) >= 1 else 'NA'
        anchor = _find_last_anchor(dirs, anchors=('real', 'fake'))
        label_tag = anchor[0] if anchor is not None else 'unknown'
        return f'{label_tag}/{video_name}'
    m = _FRAMES_RE.search(s)
    if m:
        return m.group(1)
    if len(dirs) >= 1:
        return dirs[-1]
    return s

def get_video_id_series(df: pd.DataFrame, ds: str) -> pd.Series:
    if 'video' in df.columns:
        return df['video'].astype(str)
    return df['path'].astype(str).apply(lambda x: infer_video_id_from_path(x, ds))

def filter_df40_video_frames(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if dataset_name != 'DF40':
        return df

    def is_non_video(p: str) -> bool:
        p = str(p).replace('\\', '/')
        for name in DF40_NONVIDEO_FOLDERS:
            if f'/{name}/' in p:
                return True
        return False
    mask = ~df['path'].astype(str).apply(is_non_video)
    removed = int((~mask).sum())
    if removed > 0:
        print(f'[DF40 FILTER] Exclude {removed} non-video frames for video-level metrics.')
    return df[mask].copy()

def strict_video_filter_once(df: pd.DataFrame, ds: str, out_dir: str, video_id_col: str='video_id', save_examples: bool=True, max_video_ids: int=200, paths_per_label: int=3, save_all_mixed_paths_txt: bool=True, max_mixed_paths: Optional[int]=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    g = df.groupby(video_id_col)['label'].nunique(dropna=False)
    mixed = g[g > 1].sort_values(ascending=False)
    n_videos = int(g.shape[0])
    n_mixed = int(mixed.shape[0])
    ratio = n_mixed / max(n_videos, 1) * 100.0
    print(f'[MIXED-LABEL CHECK] {ds}: total_videos={n_videos:,} | mixed_videos={n_mixed:,} ({ratio:.4f}%)')
    if n_mixed > 0 and save_all_mixed_paths_txt:
        mixed_ids = mixed.index.astype(str).tolist()
        mdf = df[df[video_id_col].astype(str).isin(mixed_ids)][[video_id_col, 'path', 'label']].copy()
        if max_mixed_paths is not None and len(mdf) > max_mixed_paths:
            mdf = mdf.head(int(max_mixed_paths))
            print(f'[WARN] mixed paths capped to first {max_mixed_paths} rows')
        txt_path = os.path.join(out_dir, f'mixed_label_frame_paths_{sanitize_filename(ds)}.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            for p in mdf['path'].astype(str).tolist():
                f.write(p + '\n')
        print(f'[SAVED] {txt_path} (n_paths={len(mdf):,})')
    if save_examples and n_mixed > 0:
        mixed_ids = mixed.index.tolist()[:max_video_ids]
        rows = []
        for vid in mixed_ids:
            d = df[df[video_id_col] == vid][['path', 'label']].copy()
            ex_real = d[d['label'] == 0].head(paths_per_label)
            ex_fake = d[d['label'] == 1].head(paths_per_label)
            ex = pd.concat([ex_real, ex_fake], axis=0)
            for _, r in ex.iterrows():
                rows.append({'dataset': ds, 'video_id': str(vid), 'label': int(r['label']), 'path': str(r['path'])})
        rep = pd.DataFrame(rows)
        save_path = os.path.join(out_dir, f'mixed_label_videos_{sanitize_filename(ds)}.csv')
        rep.to_csv(save_path, index=False)
        print(f'[SAVED] {save_path}')
    good_video_ids = set(g[g == 1].index.astype(str).tolist())
    df_strict = df[df[video_id_col].astype(str).isin(good_video_ids)].copy()
    info = {'dataset': ds, 'total_videos': n_videos, 'mixed_videos': n_mixed, 'mixed_ratio_pct': ratio, 'strict_frames': int(len(df_strict)), 'strict_videos': int(len(good_video_ids))}
    return (df_strict, info)

def compute_video_metrics_strict_from_frames(df_frames_strict: pd.DataFrame, score_col: str, video_id_col: str='video_id') -> Dict[str, Optional[float]]:
    if score_col not in df_frames_strict.columns:
        return {'auc': None, 'acc': None, 'n_valid_videos': 0, 'n_total_videos': 0}
    if len(df_frames_strict) == 0:
        return {'auc': None, 'acc': None, 'n_valid_videos': 0, 'n_total_videos': 0}
    dfx = df_frames_strict.copy()
    dfx[score_col] = pd.to_numeric(dfx[score_col], errors='coerce').astype(float)
    dfx = dfx[np.isfinite(dfx[score_col].values)].copy()
    if len(dfx) == 0:
        return {'auc': None, 'acc': None, 'n_valid_videos': 0, 'n_total_videos': 0}
    vd = dfx.groupby(video_id_col, as_index=False).agg(label=('label', 'max'), score=(score_col, 'mean'))
    n_total = int(len(vd))
    vd = vd[np.isfinite(vd['score'].astype(float).values)].copy()
    n_valid = int(len(vd))
    if len(vd) == 0:
        return {'auc': None, 'acc': None, 'n_valid_videos': 0, 'n_total_videos': n_total}
    y_true = vd['label'].astype(int).values
    y_score = vd['score'].astype(float).values
    return {'auc': safe_auc(y_true, y_score), 'acc': safe_acc(y_true, y_score), 'n_valid_videos': n_valid, 'n_total_videos': n_total}

def load_ffpp_rbf_centers_from_train(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    need = ['cr_fiqa_norm_centered', 'qcn_norm_centered']
    for c in need:
        if c not in df.columns:
            raise KeyError(f"[ERROR] FF++ standardized CSV missing '{c}'")
    x = df['cr_fiqa_norm_centered'].astype(float).values
    y = df['qcn_norm_centered'].astype(float).values
    q1 = (x >= 0) & (y >= 0)
    q2 = (x < 0) & (y >= 0)
    q3 = (x < 0) & (y < 0)
    q4 = (x >= 0) & (y < 0)
    centers = np.zeros((4, 2), dtype=float)

    def _center(mask, name):
        if mask.sum() == 0:
            print(f'[WARN] FF++ train {name} empty -> (0,0)')
            return (0.0, 0.0)
        return (float(x[mask].mean()), float(y[mask].mean()))
    centers[0] = _center(q1, 'Q1')
    centers[1] = _center(q2, 'Q2')
    centers[2] = _center(q3, 'Q3')
    centers[3] = _center(q4, 'Q4')
    print('[INFO] FF++ RBF centers (Q1~Q4):')
    print(centers)
    return centers

def make_qrouted(df: pd.DataFrame) -> pd.DataFrame:
    for c in ['quadrant', 'q1', 'q2', 'q3', 'q4']:
        if c not in df.columns:
            raise KeyError(f"[ERROR] merged csv missing '{c}' for q_routed.")
    df = df.copy()
    df['q_routed'] = df.apply(lambda r: r[f"q{int(r['quadrant'])}"], axis=1)
    return df

def quality_weights_rbf(x: np.ndarray, y: np.ndarray, sigma: float=1.0) -> np.ndarray:
    global RBF_CENTERS
    if RBF_CENTERS is None:
        raise RuntimeError('RBF_CENTERS is None (load FF++ centers first).')
    z = np.stack([x, y], axis=1)
    diff = z[:, None, :] - RBF_CENTERS[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    W = np.exp(-d2 / (2.0 * sigma ** 2))
    return W / (W.sum(axis=1, keepdims=True) + 1e-08)

def compute_quality_soft_scores(df: pd.DataFrame, sigma_rbf: float=1.0) -> np.ndarray:
    need = ['crfiqa', 'qcn', 'q1', 'q2', 'q3', 'q4']
    for c in need:
        if c not in df.columns:
            raise KeyError(f"[ERROR] merged csv missing '{c}' for quality-soft.")
    x = pd.to_numeric(df['crfiqa'], errors='coerce').astype(float).values
    y = pd.to_numeric(df['qcn'], errors='coerce').astype(float).values
    W = quality_weights_rbf(x, y, sigma=sigma_rbf)
    q = df[['q1', 'q2', 'q3', 'q4']].astype(float).values
    return (W * q).sum(axis=1)

def compute_error(score: np.ndarray, label: np.ndarray) -> np.ndarray:
    if score.ndim == 1:
        score = score[:, None]
    return np.where(label[:, None] == 1, 1.0 - score, score)

def compute_upper_bound_scores(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    cols2 = [c for c in cols if c in df.columns]
    if len(cols2) == 0:
        raise RuntimeError('upper_bound has no available candidate columns.')
    labels = df['label'].astype(int).values
    scores = df[cols2].astype(float).values
    scores = np.where(np.isfinite(scores), scores, 0.5)
    errs = compute_error(scores, labels)
    best_idx = np.argmin(errs, axis=1)
    return scores[np.arange(len(df)), best_idx]

def add_gate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    q = df[['q1', 'q2', 'q3', 'q4']].astype(float).values
    q_std = q.std(axis=1)
    q_sorted = np.sort(q, axis=1)
    q_max = q_sorted[:, -1]
    q_second = q_sorted[:, -2]
    q_min = q_sorted[:, 0]
    df['q_std'] = q_std
    df['q_margin'] = q_max - q_second
    df['q_range'] = q_max - q_min
    s = pd.to_numeric(df['c23'], errors='coerce').astype(float).values
    y = df['label'].astype(int).values
    df['conf'] = np.abs(s - 0.5) * 2.0
    df['y_correct'] = ((s >= 0.5).astype(int) == y).astype(int)
    return df

def parse_ffpp_quality(p: str) -> str:
    s = str(p).replace('\\', '/').lower()
    if '/c40/' in s or 'faceforensics++_c40' in s:
        return 'c40'
    if '/c23/' in s or 'faceforensics++_c23' in s:
        return 'c23'
    if '/raw/' in s or 'faceforensics++_raw' in s:
        return 'raw'
    return 'unknown'

def filter_ffpp_c40_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    q = df['path'].astype(str).apply(parse_ffpp_quality)
    is_real = df['label'] == 0
    is_fake_c40 = (df['label'] == 1) & (q == 'c40')
    out = df[is_real | is_fake_c40].copy()
    print(f"[FF++] keep fake=c40 + all_real: N={len(out):,} (real={(out['label'] == 0).sum():,}, fake={(out['label'] == 1).sum():,})")
    return out

def fit_gate_hgb(df_train: pd.DataFrame, feat_cols: List[str], seed: int=0):
    d = df_train.dropna(subset=feat_cols + ['y_correct']).copy()
    X = d[feat_cols].values.astype(float)
    y = d['y_correct'].values.astype(int)
    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=400, random_state=seed)
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, (p >= 0.5).astype(int)) * 100.0
    auc = roc_auc_score(y, p) * 100.0 if len(np.unique(y)) > 1 else float('nan')
    print('\n[Gate HGB trained on FF++ (real + fake=c40)]')
    print(f'  feats      = {feat_cols}')
    print(f'  train_acc  = {acc:.2f}%')
    print(f'  train_auc  = {auc:.2f}%')
    return model

def gate_predict_p_correct(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]

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
    m = ~a & ~b
    lam[a] = lam_base
    lam[b] = lam_min
    if m.any():
        t = (pw[m] - wrong_low) / max(wrong_high - wrong_low, 1e-12)
        lam[m] = lam_base + t * (lam_min - lam_base)
    return lam

def build_mix_adaptive(s_c23: np.ndarray, s_soft: np.ndarray, low_mask: np.ndarray, p_correct: np.ndarray, lam_base: float, lam_min: float, wrong_low: float, wrong_high: float) -> Tuple[np.ndarray, np.ndarray]:
    out = np.empty_like(s_c23, dtype=np.float32)
    out[low_mask] = s_soft[low_mask]
    hi = ~low_mask
    p_wrong = 1.0 - p_correct
    lam_all = lambda_from_pwrong_piecewise(p_wrong, lam_base, lam_min, wrong_low, wrong_high)
    out[hi] = lam_all[hi] * s_c23[hi] + (1.0 - lam_all[hi]) * s_soft[hi]
    return (out, lam_all)

def build_df_common(df: pd.DataFrame, ds: str) -> pd.DataFrame:
    df = df.copy()
    df = make_qrouted(df)
    df['s_soft'] = compute_quality_soft_scores(df, sigma_rbf=RBF_SIGMA)
    df = add_gate_features(df)
    df['__dataset__'] = ds
    return df

def make_ffpp_splits(df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    """
    Split FaceForensics++ evaluation into three subsets by path quality:
      - raw, c23, c40
    Returns list of (split_name, split_df).
    """
    q = df['path'].astype(str).apply(parse_ffpp_quality)
    out = []
    for tag in ['raw', 'c23', 'c40']:
        d = df[q == tag].copy()
        out.append((f'FaceForensics++({tag})', d))
    return out
