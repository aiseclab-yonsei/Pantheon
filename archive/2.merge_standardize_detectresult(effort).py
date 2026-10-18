import os
import re
import pandas as pd
from typing import Dict, Tuple, Set, List, Optional

# =========================
# 1. 유틸 함수
# =========================

def norm_path(p: str) -> str:
    return os.path.normpath(str(p)).replace("\\", "/")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def quadrant_from_scores(cr: float, qcn: float) -> int:
    if cr >= 0 and qcn >= 0:
        return 1
    elif cr < 0 and qcn >= 0:
        return 2
    elif cr < 0 and qcn < 0:
        return 3
    else:
        return 4


def read_concat_csv(paths: List[str], tag: str) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] Missing ({tag}): {p}")
            continue
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"[WARN] Failed to read ({tag}): {p} ({e})")
            continue
    if len(dfs) == 0:
        raise FileNotFoundError(f"[ERROR] No readable CSV for {tag}")
    return pd.concat(dfs, ignore_index=True)


# =========================
# 2. key 생성 (핵심 수정)
# =========================

def make_key_generic(path: str, token: str) -> str:
    """
    path에서 token이 나오는 위치부터 끝까지 (case-insensitive).
    """
    p = norm_path(path)
    p_low = p.lower()
    token_low = token.lower()
    idx = p_low.find(token_low)
    if idx >= 0:
        return p[idx:]
    return p


_FFPP_SEG_RE = re.compile(r"(faceforensics\+\+)(_[^/]+)?", re.IGNORECASE)

def make_key_ffpp(path: str) -> str:
    """
    FaceForensics++ 전용:
      1) 경로에서 'FaceForensics++'가 시작되는 위치부터 자른다.
      2) 첫 세그먼트가 FaceForensics++_RAW / _c23 / _c40 / _raw 등이어도
         이를 항상 'FaceForensics++'로 정규화한다.

    예)
      .../FaceForensics++_RAW/original_sequences/... -> FaceForensics++/original_sequences/...
      .../FaceForensics++_c23/manipulated_sequences/... -> FaceForensics++/manipulated_sequences/...
    """
    p = norm_path(path)

    # 1) FaceForensics++ 시작 지점 찾기 (case-insensitive)
    m = re.search(r"faceforensics\+\+", p, flags=re.IGNORECASE)
    if not m:
        return p

    key = p[m.start():]  # FaceForensics++...부터 끝까지

    # 2) 첫 세그먼트의 변형(_RAW/_c23/...) 제거: FaceForensics++_XXX -> FaceForensics++
    #    key의 맨 앞에서만 1회 치환
    key = _FFPP_SEG_RE.sub(r"FaceForensics++", key, count=1)

    return key


def make_key(path: str, dataset: str, subdataname: str) -> str:
    """
    dataset에 따라 key 생성 방식을 선택.
    """
    if dataset == "FaceForensics++":
        return make_key_ffpp(path)
    return make_key_generic(path, subdataname)


# =========================
# 3. FaceForensics++ 설정
# =========================

FFPP_SUBSETS = ["FaceForensics++_c23", "FaceForensics++_c40", "FaceForensics++_raw"]

def get_pred_subdatasets(dataset: str) -> List[str]:
    return FFPP_SUBSETS if dataset == "FaceForensics++" else [dataset]

def get_qcn_dataset_name(dataset: str) -> str:
    return "FaceForensics++" if dataset == "FaceForensics++" else dataset


# =========================
# 4. 경로 생성
# =========================

def build_paths_for_dataset(dataset: str) -> Tuple[str, Dict[str, List[str]]]:
    qcn_name = get_qcn_dataset_name(dataset)
    qcn_csv = QCN_CSV_PATTERN.format(root=QCN_ROOT, dataset=qcn_name)

    pred_subdatasets = get_pred_subdatasets(dataset)

    model_csvs = {
         
        # "raw": [f"{EVAL_ROOT_GLOBAL}/{MODEL}_train_raw/{sd}/frame_predictions.csv" for sd in pred_subdatasets],
        "c23": [f"{EVAL_ROOT_GLOBAL}/{MODEL}_train_c23/{sd}/frame_predictions.csv" for sd in pred_subdatasets],
        # "c40": [f"{EVAL_ROOT_GLOBAL}/{MODEL}_train_c40/{sd}/frame_predictions.csv" for sd in pred_subdatasets],
        # "all": [f"{EVAL_ROOT_GLOBAL}/{MODEL}_train_all/{sd}/frame_predictions.csv" for sd in pred_subdatasets],

        "q1":  [f"{EVAL_ROOT_QUAD}/{MODEL}_train_Q1/{sd}/frame_predictions.csv" for sd in pred_subdatasets],
        "q2":  [f"{EVAL_ROOT_QUAD}/{MODEL}_train_Q2/{sd}/frame_predictions.csv" for sd in pred_subdatasets],
        "q3":  [f"{EVAL_ROOT_QUAD}/{MODEL}_train_Q3/{sd}/frame_predictions.csv" for sd in pred_subdatasets],
        "q4":  [f"{EVAL_ROOT_QUAD}/{MODEL}_train_Q4/{sd}/frame_predictions.csv" for sd in pred_subdatasets],
    }

    return qcn_csv, model_csvs


# =========================
# 5. 병합 로직
# =========================

def merge_for_dataset(dataset: str, subdataname: str) -> None:
    qcn_csv, model_csvs = build_paths_for_dataset(dataset)

    print(f"[INFO] Dataset: {dataset}")
    print(f"[INFO] QCN CSV: {qcn_csv}")

    if not os.path.exists(qcn_csv):
        raise FileNotFoundError(f"[ERROR] QCN CSV not found: {qcn_csv}")

    # ---- QCN 로딩
    qdf = pd.read_csv(qcn_csv)
    qdf.columns = [c.strip().lower() for c in qdf.columns]

    path_col = qdf.columns[0]
    cr_col = "cr_fiqa_norm_centered"
    qcn_col = "qcn_norm_centered"
    if cr_col not in qdf.columns or qcn_col not in qdf.columns:
        raise KeyError(f"[ERROR] QCN CSV must include '{cr_col}' and '{qcn_col}'.")

    qdf["key"] = qdf[path_col].apply(lambda p: make_key(p, dataset, subdataname))
    qdf["crfiqa"] = qdf[cr_col]
    qdf["qcn"] = qdf[qcn_col]
    qdf["quadrant"] = qdf.apply(lambda x: quadrant_from_scores(x["crfiqa"], x["qcn"]), axis=1)

    merge_df = qdf[["key", "crfiqa", "qcn", "quadrant"]].copy()

    q_keys: Set[str] = set(merge_df["key"].tolist())
    pred_keys_union: Set[str] = set()
    label_df: Optional[pd.DataFrame] = None

    # ---- 모델별 pred 로딩 & merge
    for model_name, csv_paths in model_csvs.items():
        print(f"[INFO] Loading '{model_name}' (concat {len(csv_paths)} files)")
        df = read_concat_csv(csv_paths, tag=f"{model_name}:{dataset}")
        df.columns = [c.strip().lower() for c in df.columns]

        # 포맷: dataset, path, pred, label
        if len(df.columns) < 4:
            raise ValueError(f"[ERROR] Unexpected CSV format for {model_name}:{dataset}")

        pcol = df.columns[1]
        if "pred" not in df.columns:
            raise KeyError(f"[ERROR] '{model_name}:{dataset}' CSV must include 'pred' column.")

        df["key"] = df[pcol].apply(lambda p: make_key(p, dataset, subdataname))

        if label_df is None and "label" in df.columns:
            label_df = df[["key", "label"]].copy()

        df_sub = df[["key", "pred"]].rename(columns={"pred": model_name})
        merge_df = merge_df.merge(df_sub, how="left", on="key")

        pred_keys_union.update(df["key"].tolist())

    # ---- label merge
    if label_df is not None:
        merge_df = merge_df.merge(label_df, how="left", on="key")
        merge_df["label"] = merge_df["label"].astype("Int64")
    else:
        merge_df["label"] = pd.NA

    # pred에 존재하는 key만 유지
    merge_df = merge_df[merge_df["key"].isin(pred_keys_union)].copy()

    # ---- 최종 컬럼 정리
    # for m in ["raw", "c23", "c40", "all", "q1", "q2", "q3", "q4"]:
    #     if m not in merge_df.columns:
    #         merge_df[m] = pd.NA

    # final_cols = [
    #     "key", "label", "crfiqa", "qcn", "quadrant",
    #     "raw", "c23", "c40", "all",
    #     "q1", "q2", "q3", "q4",
    # ]
    for m in ["c23", "q1", "q2", "q3", "q4"]:
        if m not in merge_df.columns:
            merge_df[m] = pd.NA

    final_cols = [
        "key", "label", "crfiqa", "qcn", "quadrant",
         "c23", 
        "q1", "q2", "q3", "q4",
    ]
    merge_df = merge_df[final_cols].copy()
    merge_df.rename(columns={"key": "path"}, inplace=True)

    # ---- mismatch 로그
    q_only = sorted(q_keys - pred_keys_union)
    pred_only = sorted(pred_keys_union - q_keys)

    ensure_dir(OUT_ROOT)
    out_csv = f"{OUT_ROOT}/merged_{dataset}.csv"
    q_only_log = f"{OUT_ROOT}/qcn_only_{dataset}.txt"
    pred_only_log = f"{OUT_ROOT}/pred_only_{dataset}.txt"

    merge_df.to_csv(out_csv, index=False)
    print(f"[DONE] Merged CSV saved to: {out_csv}")

    with open(q_only_log, "w", encoding="utf-8") as f:
        f.write("# key only in QCN\n")
        for k in q_only:
            f.write(k + "\n")
    print(f"[LOG] QCN-only keys: {len(q_only)} → {q_only_log}")

    with open(pred_only_log, "w", encoding="utf-8") as f:
        f.write("# key only in PRED\n")
        for k in pred_only:
            f.write(k + "\n")
    print(f"[LOG] Pred-only keys: {len(pred_only)} → {pred_only_log}")


# =========================
# 6. main
# =========================

def main():
    ensure_dir(OUT_ROOT)
    merge_for_dataset(dataset, subdataname)


if __name__ == "__main__":
    # ===== 하드코딩 =====
    # dataset = "FaceForensics++"  # 예: FaceForensics++, DFDC, Celeb-DF-v1 ...
    # datasets=[
    #     "UADFV", "Celeb-DF-v1", "Celeb-DF-v2", "Celeb-DF-v3",
    #     "DFDC","DFo","Wilddeepfake","FFIW","KoDF","DF40","DF-Eval"  ]
    # datasets=[
    #   "woo_preprocess(image)"  ]
    datasets = [
       "FaceForensics++","Celeb-DF-v1","Celeb-DF-v2","Celeb-DF-v3","DFDC","DFo","wilddeepfake","FFIW","KoDF","DF40","DF-Eval" 
    ]
    for dataset in datasets:
        
        if dataset == "KoDF":
            subdataname = "kodf_preprocess"
        elif dataset == "DF-eval":
            subdataname = "Deepfake-eval-2024"
        elif dataset == "DFo":
            subdataname = "DeeperForensics1.0"
        elif dataset == "KoDF_c23":
            subdataname = "kodf_preprocess_test_c23"
        elif dataset == "KoDF_c40":
            subdataname = "kodf_preprocess_test_c40"
        elif dataset == "woo_preprocess(image)":
            subdataname = "woo_preprocess"
        elif dataset == "woo_preprocess(video)":
            subdataname = "woo_preprocess"
        else:
            subdataname = dataset



        QCN_ROOT = "/home/compu/eunbi/iqa/0.module/analysis/qcn_crfiqa/normalize_minmax_median0_v2/standardized_csv"
        QCN_CSV_PATTERN = "{root}/{dataset}_standardized.csv"

        EVAL_ROOT_GLOBAL = "/srv/shared/evaluation/Effort-AIGI-Detection.bak/DeepfakeBench/cleanFF/eval"
        EVAL_ROOT_QUAD   = "/srv/shared/evaluation/Effort-AIGI-Detection.bak/DeepfakeBench/quadrant/eval"

        MODEL="effort"
        OUT_ROOT = f"/home/compu/eunbi/iqa/0.module/analysis/qcn_crfiqa/normalize_minmax_median0_v2/merge/{MODEL}"

        main()
