#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
from scipy.stats import shapiro
import matplotlib.pyplot as plt


# ================================
# 0. 유틸
# ================================
def read_any_csv(path: str) -> pd.DataFrame:
    """구분자 자동 감지 + 앞뒤 공백 제거."""
    df = pd.read_csv(path, sep=None, engine="python", skipinitialspace=True)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


# ================================
# 1. 정규성 체크 (참고용)
# ================================
def check_normality(series: pd.Series, name: str) -> None:
    series = series.dropna()
    stat, p = shapiro(series)
    print(f"[{name}] Shapiro-Wilk 정규성 검정:")
    print(f"  statistic = {stat:.4f}, p-value = {p:.4f}")
    print("  → 정규분포 OK\n" if p > 0.05 else "  → 정규분포 아님\n")


# ================================
# 2. 기준 파라미터 계산 + 저장
#    (IQR 기반 이상치 제거 + median 기준 양쪽 min-max)
# ================================
def compute_and_save_params(df: pd.DataFrame, columns, save_path: str):
    """
    각 컬럼별로:
      1) IQR 기반 이상치 범위 계산
      2) 이상치를 제외한 inlier에서 median, min_inlier(아래쪽), max_inlier(위쪽) 계산
      3) 아래/위 반쪽에 대해 각각 min-max 스케일링을 할 수 있도록
         denom_lower, denom_upper 저장

    저장되는 값 (col별):
        mean, std, median, q1, q3,
        lower_bound, upper_bound,
        min_inlier, max_inlier,
        denom_lower, denom_upper
    """
    params = {}

    for col in columns:
        s = df[col].dropna().astype(float)
        if len(s) == 0:
            raise ValueError(f"[ERROR] 컬럼 '{col}' 에 유효한 값이 없습니다.")

        # IQR 기반 이상치 범위
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 3* iqr
        upper_bound = q3 + 3* iqr

        inlier_mask = (s >= lower_bound) & (s <= upper_bound)
        s_inlier = s[inlier_mask]
        if len(s_inlier) == 0:
            # 혹시 전부 이상치로 잡히면 그냥 전체 사용
            s_inlier = s
            inlier_mask[:] = True
            lower_bound = float(s.min())
            upper_bound = float(s.max())

        median = float(s_inlier.median())

        below = s_inlier[s_inlier <= median]
        above = s_inlier[s_inlier >= median]

        if len(below) == 0:
            min_inlier = median
        else:
            min_inlier = float(below.min())

        if len(above) == 0:
            max_inlier = median
        else:
            max_inlier = float(above.max())

        denom_lower = median - min_inlier
        denom_upper = max_inlier - median

        # 분모가 0이면 스케일링이 불가능하므로 1로 두고, 나중에 값은 전부 0 근처로 모이게 됨
        if denom_lower <= 0:
            denom_lower = 1.0
        if denom_upper <= 0:
            denom_upper = 1.0

        params[col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": median,
            "q1": q1,
            "q3": q3,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "min_inlier": float(min_inlier),
            "max_inlier": float(max_inlier),
            "denom_lower": float(denom_lower),
            "denom_upper": float(denom_upper),
        }

        n_outliers = (~inlier_mask).sum()
        print(
            f"[{col}] inlier 범위 = [{lower_bound:.4f}, {upper_bound:.4f}], "
            f"median={median:.4f}, min_inlier={min_inlier:.4f}, max_inlier={max_inlier:.4f}, "
            f"outliers={n_outliers}"
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(params, f, indent=4)

    print(f"[INFO] 정규화 파라미터 저장 완료 → {save_path}")
    return params


# ================================
# 3. 표준화 적용
#    (median 기준 양쪽 piecewise min-max + median을 0으로 이동)
# ================================
def apply_norm_centered(df: pd.DataFrame, params: dict, columns):
    """
    각 컬럼별로:
      - x를 [lower_bound, upper_bound]로 먼저 clip (이상치는 바운더리로 붙임)
      - median 기준으로 아래/위 두 영역으로 나눔
      - 아래쪽: (x - min_inlier) / (median - min_inlier) - 1   → [-1, 0]
      - 위쪽:   (x - median) / (max_inlier - median)           → [0, 1]
      - 최종 결과는 <col>_norm_centered 에 저장
    """
    df = df.copy()

    for col in columns:
        x = df[col].astype(float)

        if col not in params:
            raise KeyError(f"[ERROR] params 에 컬럼 '{col}' 정보가 없습니다.")

        p = params[col]
        median = p["median"]
        lower_bound = p["lower_bound"]
        upper_bound = p["upper_bound"]
        min_inlier = p["min_inlier"]
        max_inlier = p["max_inlier"]
        denom_lower = p["denom_lower"]
        denom_upper = p["denom_upper"]

        # 이상치는 바운더리로 clip
        x_clipped = x.clip(lower=lower_bound, upper=upper_bound)

        # median 기준으로 분리
        mask_lower = x_clipped <= median

        x_new = pd.Series(index=x.index, dtype=float)

        # 아래쪽: min_inlier → -1, median → 0
        if mask_lower.any():
            x_lower = x_clipped[mask_lower]
            x_lower_scaled = (x_lower - min_inlier) / denom_lower  # 0~1
            x_new[mask_lower] = x_lower_scaled - 1.0               # -1~0

        # 위쪽: median → 0, max_inlier → +1
        if (~mask_lower).any():
            x_upper = x_clipped[~mask_lower]
            x_upper_scaled = (x_upper - median) / denom_upper      # 0~1
            x_new[~mask_lower] = x_upper_scaled                    # 0~1

        df[col + "_norm_centered"] = x_new

    return df


# ================================
# 4. 사분면 분리 + real/fake 카운트 출력
# ================================
def split_quadrants_by_norm_centered(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_dir: str,
    prefix: str = ""
):
    """
    - 사분면별 real/fake/unknown 개수 + 비율 출력
    - rel_path만 저장
    """
    os.makedirs(out_dir, exist_ok=True)

    x, y = df[x_col].values, df[y_col].values

    quads = {
        "Q1_xpos_ypos": (x >= 0) & (y >= 0),
        "Q2_xneg_ypos": (x < 0) & (y >= 0),
        "Q3_xneg_yneg": (x < 0) & (y < 0),
        "Q4_xpos_yneg": (x >= 0) & (y < 0),
    }

    def label_from_relpath(p: str) -> str:
        p = p.lower()
        if "manipulated" in p:
            return "fake"
        if "original" in p:
            return "real"
        return "unknown"

    print("=== Quadrant counts (norm_centered) ===")
    total_all = len(df)

    for name, mask in quads.items():
        df_q = df.loc[mask].copy()
        df_q["label"] = df_q["rel_path"].apply(label_from_relpath)

        total = len(df_q)
        n_real = (df_q["label"] == "real").sum()
        n_fake = (df_q["label"] == "fake").sum()
        n_unknown = (df_q["label"] == "unknown").sum()

        real_ratio = (n_real / total * 100) if total > 0 else 0
        fake_ratio = (n_fake / total * 100) if total > 0 else 0
        unknown_ratio = (n_unknown / total * 100) if total > 0 else 0

        print(
            f"{name}: Total={total} ({total/total_all:.2%}) |  "
            f"real={n_real} ({real_ratio:.1f}%),  "
            f"fake={n_fake} ({fake_ratio:.1f}%),  "
            f"unknown={n_unknown} ({unknown_ratio:.1f}%)"
        )

        out_path = os.path.join(out_dir, f"{prefix}{name}_rel_path.csv")
        df_q[["rel_path"]].to_csv(out_path, index=False)
        print(f"[INFO] 저장 완료 → {out_path}")


# ================================
# 5. 기준 FF++ CSV로 파라미터 학습 + 표준화 + 분리
# ================================
def fit_on_base_csvs(csv_list, params_json, standardized_csv, quad_out_dir):
    if isinstance(csv_list, str):
        csv_list = [csv_list]

    dfs = [read_any_csv(p) for p in csv_list]
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] 기준 merged 개수: {len(merged_df)}")

    check_normality(merged_df["cr_fiqa"], "cr_fiqa")
    check_normality(merged_df["qcn"], "qcn")

    params = compute_and_save_params(
        merged_df, ["cr_fiqa", "qcn"], save_path=params_json
    )

    merged_std = apply_norm_centered(merged_df, params, ["cr_fiqa", "qcn"])

    out_df = merged_std[[
        "rel_path",
        "cr_fiqa_norm_centered",
        "qcn_norm_centered",
    ]]

    os.makedirs(os.path.dirname(standardized_csv), exist_ok=True)
    out_df.to_csv(standardized_csv, index=False)
    print(f"[INFO] 기준 CSV (표준화) 저장 → {standardized_csv}")

    split_quadrants_by_norm_centered(
        out_df,
        x_col="cr_fiqa_norm_centered",
        y_col="qcn_norm_centered",
        out_dir=quad_out_dir,
        prefix="base_",
    )

    return out_df, params


# ================================
# 6. 새 CSV에 동일 기준 적용 및 사분면 분리
# ================================
def normalize_new_csv_and_split(
    input_csv: str,
    output_csv: str,
    params_json: str,
    quad_out_dir: str,
    prefix: str = "",
):
    df = read_any_csv(input_csv)

    with open(params_json, "r") as f:
        params = json.load(f)

    df_std = apply_norm_centered(df, params, ["cr_fiqa", "qcn"])

    out_df = df_std[[
        "rel_path",
        "cr_fiqa_norm_centered",
        "qcn_norm_centered",
    ]]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"[INFO] 새 CSV (표준화) 저장 → {output_csv}")

    split_quadrants_by_norm_centered(
        out_df,
        x_col="cr_fiqa_norm_centered",
        y_col="qcn_norm_centered",
        out_dir=quad_out_dir,
        prefix=prefix,
    )

    return out_df


# ================================
# 7. 산점도 유틸 (색상 매핑 포함)
# ================================
def _get_dataset_key_from_path(csv_path: str) -> str:
    """
    파일 이름에서 데이터셋 이름을 추출해 COLORS 키로 사용.
    - FaceForensics++_raw_merged(...)   -> FF_raw
    - FaceForensics++_c40_merged(...)   -> FF_c40
    - FaceForensics++_c23_merged(...)   -> FF_c23
    - Celeb-DF-v1_merged(...)           -> Celeb-DF-v1
    - DFDC_merged(...)                  -> DFDC
    - Realworld_Deepfake_merged(...)    -> Realworld_Deepfake
    등.
    """
    fname = os.path.basename(csv_path)
    base = fname.split("_merged")[0]  # 예: "Celeb-DF-v1", "FaceForensics++_raw"

    if base.startswith("FaceForensics++"):
        if "_raw" in base:
            return "FF_raw"
        elif "_c40" in base:
            return "FF_c40"
        elif "_c23" in base:
            return "FF_c23"
        else:
            return "FF_raw"

    return base


def scatter_from_csvs_with_norm(
    csv_list,
    params_json: str,
    out_path: str,
    x_col: str = "cr_fiqa",
    y_col: str = "qcn",
):
    """
    - 각 CSV를 동일한 참조(params_json) 기준으로 표준화한 뒤
      <x_col>_norm_centered vs <y_col>_norm_centered 산점도를 서로 다른 색으로 플롯.
    - x=0, y=0이 median 위치가 되며, 아래쪽 [-1,0], 위쪽 [0,1] 범위를 갖도록 스케일됨.
    """
    if isinstance(csv_list, str):
        csv_list = [csv_list]

    with open(params_json, "r") as f:
        params = json.load(f)

    plt.figure(figsize=(8, 6))

    all_x = []
    all_y = []

    dataset_key_for_title = None

    for csv_path in csv_list:
        df = read_any_csv(csv_path)
        df_std = apply_norm_centered(df, params, [x_col, y_col])

        x = df_std[f"{x_col}_norm_centered"].astype(float)
        y = df_std[f"{y_col}_norm_centered"].astype(float)

        all_x.append(x)
        all_y.append(y)

        dataset_key = _get_dataset_key_from_path(csv_path)
        dataset_key_for_title = dataset_key  # 마지막 것 기준
        color = COLORS.get(dataset_key, "#7f8c8d")  # fallback 회색
        label = dataset_key

        plt.scatter(
            x,
            y,
            s=6,
            alpha=0.5,
            label=label,
            color=color,
        )

    all_x = pd.concat(all_x)
    all_y = pd.concat(all_y)

    # 이론상 [-1,1] 안에 들어가지만, 실제 min/max 기준으로 약간 여유를 둠
    max_x = max(abs(all_x.min()), abs(all_x.max()))
    max_y = max(abs(all_y.min()), abs(all_y.max()))
    max_lim = max(max_x, max_y)
    max_lim = max(max_lim, 1.0)  # 최소 범위 확보

    plt.xlim(-max_lim, +max_lim)
    plt.ylim(-max_lim, +max_lim)

    plt.axvline(0, linestyle="--", linewidth=1, color="black")
    plt.axhline(0, linestyle="--", linewidth=1, color="black")

    plt.xlabel(f"{x_col}")
    plt.ylabel(f"{y_col}")
    if dataset_key_for_title is None:
        title_ds = "datasets"
    else:
        title_ds = "datasets" 
        
    plt.title(f"{dataset}_standardized_scatter")
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[INFO] 산점도 저장 완료 → {out_path}")


# ================================
# 8. 실행 예시
# ================================
if __name__ == "__main__":
        
    # ================================
    # 0. 전역 설정 (색상 팔레트)
    # ================================
    COLORS = {
        "FF_c40": "#bdc3c7",
        "FF_raw": "#bdc3c7",
        "FF_c23": "#bdc3c7",
        "Celeb-DF-v1": "#3498db",
        "Celeb-DF-v2": "#9b59b6",
        "Celeb-DF-v2_SR": "#e74c3c",
        "Celeb-DF-v1_SR": "#2ecc71",
        "UADFV": "#450693",
        "wilddeepfake": "#e67e22",   
        "DFDC": "#ff69b4",
        "Realworld_Deepfake": "#34495e",
        "kodf": "#2ecc71",
        "wilddeepfake_pro": "#9b59b6",
        "DF40": "#e74c3c",
        "Celeb-DF-v3": "#1abc9c",
        "DF-eval": "#f1c40f",
        "DFo": "#e74c3c",
        "FFIW": "#450693",
        "woo_preprocess(image)": "#3498db",
        "woo_preprocess(video)": "#9b59b6",
    }


    BASE_CSV_LIST = [
        "/home/compu/eunbi/iqa/0.module/analysis/qcn_crfiqa/FaceForensics++_raw_merged(cr_fiqa+qcn).csv",
        "/home/compu/eunbi/iqa/0.module/analysis/qcn_crfiqa/FaceForensics++_c23_merged(cr_fiqa+qcn).csv",
        "/home/compu/eunbi/iqa/0.module/analysis/qcn_crfiqa/FaceForensics++_c40_merged(cr_fiqa+qcn).csv",
    ]

    BASE_DIR = "/home/compu/eunbi/iqa/0.module/analysis/qcn_crfiqa/normalize_minmax_median0_v2/standardized_csv"
    PARAMS_JSON = os.path.join(BASE_DIR, "normalization_params.json")
    STD_CSV = os.path.join(BASE_DIR, "FaceForensics++_standardized.csv")
    QUAD_DIR = os.path.join(BASE_DIR, "quadrants_base")
    # dataset="FaceForensics++(raw,c23,c40)"
    # # 1) FF++ 기준 파라미터 학습 + 사분면 분리
    fit_on_base_csvs(BASE_CSV_LIST, PARAMS_JSON, STD_CSV, QUAD_DIR)

    # # 2) FF++만 global 기준으로 표준화해서 산점도
    # SCATTER_PNG = os.path.join(BASE_DIR, "FFpp_crfiqa_qcn_norm_scatter.png")
    # scatter_from_csvs_with_norm(
    #     BASE_CSV_LIST,
    #     params_json=PARAMS_JSON,
    #     out_path=SCATTER_PNG,
    #     x_col="cr_fiqa",
    #     y_col="qcn",
    # )

    # 3) 새로운 CSV 적용 예시 (데이터셋 이름만 바꿔가며 사용)
    dataset = "Celeb-DF-v1_black-box_8_255"  # Celeb-DF-v1, DFDC, wilddeepfake, wilddeepfake_pro
    new_csv_path = f"/home/compu/eunbi/iqa/0.module/analysis/qcn_crfiqa/{dataset}_merged(cr_fiqa+qcn).csv"
    
    normalize_new_csv_and_split(
        input_csv=new_csv_path,
        output_csv=os.path.join(BASE_DIR, f"{dataset}_standardized.csv"),
        params_json=PARAMS_JSON,
        quad_out_dir=os.path.join(BASE_DIR, f"quadrants_{dataset}"),
        prefix=f"{dataset}_",
    )
    
    # 4) FF++ + 새 데이터셋까지 포함해서 한 장에 산점도
    SCATTER_PNG_ALL = os.path.join(BASE_DIR, f"FFpp_{dataset}_crfiqa_qcn_norm_scatter.png")
    scatter_from_csvs_with_norm(
        BASE_CSV_LIST + [new_csv_path],
        params_json=PARAMS_JSON,
        out_path=SCATTER_PNG_ALL,
        x_col="cr_fiqa",
        y_col="qcn",
    )

    print("\n[모든 작업 완료]")
