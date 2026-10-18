#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[CSV 입력 버전 - UI 연동형]
CSV(첫 행 헤더, 첫 번째 열=절대경로) → 품질 점수 산출
"""

import os
import sys
import csv
import argparse
import warnings
from typing import List, Tuple
import numpy as np

# 💡 PyTorch FutureWarning 숨기기
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.QualityModel import QualityModel

# -------------------------
# 유틸 함수
# -------------------------
def _norm(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")

def _detect_delimiter(csv_path: str, default: str = ",") -> str:
    with open(csv_path, "r", newline="") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        return default

def _read_first_column_paths(csv_path: str) -> Tuple[List[str], str]:
    delim = _detect_delimiter(csv_path)
    paths: List[str] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV에 헤더가 없습니다: {csv_path}")
        for row in reader:
            if row and row[0].strip():
                paths.append(_norm(row[0].strip()))
    return paths, delim

def _progress(i_done: int, total: int) -> None:
    pct = (i_done / total * 100.0) if total > 0 else 100.0
    print(f"[{i_done:>6}/{total}] {pct:5.1f}% 완료", flush=True)

# -------------------------
# 실행부
# -------------------------
def main():
    # 💡 main_ui.py와의 통신을 위한 인자 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    INPUT_CSV = args.input_csv
    score_path = args.output_csv

    # ======== 설정 ========
    BATCH_SIZE     = 2048
    COLOR_CHANNEL  = "BGR"
    GPU_ID         = 0
    MODEL_PATH     = "/home/compu/mlflow_t3/_iqa/CR-FIQA/model/CR-FIQA(L)/iresnet100"
    MODEL_ID       = "181952"
    # ========================

    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f"입력 CSV를 찾을 수 없습니다: {INPUT_CSV}")

    paths, _ = _read_first_column_paths(INPUT_CSV)
    total = len(paths)
    print(f"[INFO] 입력 CSV: {INPUT_CSV}")
    print(f"[INFO] 총 이미지 수: {total}")

    if total == 0:
        print("[WARN] 경로가 비어 있습니다. 종료합니다.")
        return

    exist_paths = [p for p in paths if os.path.isfile(p)]
    if len(exist_paths) < total:
        print(f"[WARN] {total - len(exist_paths)}개의 파일이 존재하지 않아 제외됨.")
    paths = exist_paths
    total = len(paths)

    print("[INFO] 모델 초기화 중...")
    model = QualityModel(MODEL_PATH, MODEL_ID, GPU_ID)

    # 💡 요청하신 헤더명으로 출력 파일 초기화
    with open(score_path, "w", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(["image_path", "predicted_score"])

    done = 0
    while done < total:
        batch = paths[done: done + BATCH_SIZE]
        try:
            _, qualities = model.get_batch_feature(batch, batch_size=BATCH_SIZE, color=COLOR_CHANNEL)
        except Exception as e:
            print(f"[ERROR] 배치 실패 ({done}~{done+len(batch)}): {e}")
            qualities = []
            for p in batch:
                try:
                    _, q = model.get_batch_feature([p], batch_size=1, color=COLOR_CHANNEL)
                    qualities.append(q[0])
                except Exception as e1:
                    print(f"[ERROR] 개별 파일 실패: {p} ({e1})")
                    qualities.append([float("nan")])

        with open(score_path, "a", newline="") as fw:
            writer = csv.writer(fw)
            for p, q in zip(batch, qualities):
                if isinstance(q, (list, tuple, np.ndarray)):
                    val = float(q[0])
                else:
                    val = float(q)
                writer.writerow([p, val])

        done += len(batch)
        _progress(done, total)

    print(f"[OK] 품질 점수 저장 완료 → {score_path} (총 {total}개)", flush=True)
    
    # 💡 작업 종료 시 프로세스 강제 종료 (UI 무한 대기 현상 방지)
    os._exit(0)

if __name__ == "__main__":
    main()