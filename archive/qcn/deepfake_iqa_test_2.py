#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import math
from datetime import datetime
import argparse # 💡 argparse 추가

import numpy as np
import pandas as pd
import torch
import torchvision
from einops import rearrange

from network.network_utils import build_model
from network.optimizer_utils import get_optimizer, get_scheduler
from dataloaders import dataloader_gen

from utils.loss_utils import compute_center_loss_v2, compute_rpt_direction_v1, compute_mae_loss_v3, compute_metric_loss_v1
from utils.train_utils import get_batches_v7_mask_single, Maxlloyd, get_left_right_idxs, get_pairs_equally_fast_v2
from utils.test_utils import compute_score_v1
from utils.util import load_model, save_model
from utils.util import set_wandb, tensor2np, write_log

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 1번 카드만 보이도록

# =========================
# 핵심 유틸
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def append_rows_csv(csv_path: str, rows: list, header: list):
    """csv_path 없으면 헤더 쓰고, 있으면 헤더 없이 rows만 append."""
    file_exists = os.path.exists(csv_path)
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(rows)
        f.flush()  # 즉시 디스크 반영

# =========================
# 평가 루프 (샘플마다 바로 저장)
# =========================
@torch.no_grad()
def evaluation_and_append(cfg, net, data_loader, out_csv_path: str):
    net.eval()

    header = ["image_path", "predicted_score"] # 💡 UI와 통일된 헤더
    file_exists = os.path.exists(out_csv_path)
    if not file_exists:
        append_rows_csv(out_csv_path, [], header)

    use_cuda = torch.cuda.is_available()
    use_autocast = use_cuda  

    for idx, sample in enumerate(data_loader):
        sys.stdout.write(f'\r🔍 Evaluating [{idx + 1}/{len(data_loader)}]')
        sys.stdout.flush()

        if use_autocast:
            cm = torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            class _Noop:
                def __enter__(self): return None
                def __exit__(self, *a): return False
            cm = _Noop()

        with cm:
            image = sample['img'].cuda() if use_cuda else sample['img']
            img_path = sample['img_path'][0]

            f = net('extraction', {'img': image})
            f_hflip = net('extraction', {'img': torchvision.transforms.functional.hflip(image)})
            f = torch.cat([f, f_hflip], dim=0).mean(0, keepdim=True).squeeze()  

            f = f.unsqueeze(0).unsqueeze(-1).repeat(1, 1, cfg.reference_point_num)
            f_clustered, score_pivots = net('get_cluster', {'f': f})

            dummy_emb_scores = (torch.zeros(f_clustered.shape[0], 1)
                                .to(image.device)
                                .float())

            pred = compute_score_v1(
                embs=rearrange(f_clustered, 'b c l -> b l c')[:, -1:],  
                spv=rearrange(score_pivots, 'b c l -> b l c'),
                emb_scores=dummy_emb_scores,
                spv_scores=cfg.score_pivot_score
            )

            append_rows_csv(out_csv_path, [[img_path, float(pred.item())]], header)

    print("\n[✓] Evaluation & save completed.")

# =========================
# 메인
# =========================
def main(cfg, out_csv): # 💡 out_csv 파라미터 추가
    # 데이터로더 & 모델
    test_loader = dataloader_gen.gen_dataloader(cfg)
    net = build_model(cfg)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(0)
        print("Available GPUs:", torch.cuda.device_count())
        print("Current CUDA Device:", torch.cuda.current_device())
        print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        net = net.cuda()

    optimizer = get_optimizer(cfg, net)

    if cfg.load:
        load_model(cfg, net, optimizer=optimizer, load_optim_params=False)

    net.eval()

    print(f"[INFO] Output CSV → {out_csv}")
    print(f"[INFO] Total samples in dataloader: {len(test_loader)}")

    # 평가하면서 즉시 저장
    try:
        evaluation_and_append(cfg, net, test_loader, out_csv)
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Partial results kept at:", out_csv)
    finally:
        print("[DONE] Process finished.")
        os._exit(0) # 💡 UI 무한대기 방지용 프로세스 강제 종료


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    
    # 💡 UI로부터 추가로 전달받을 경로 인자들
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--dataset_root", type=str, default="")
    parser.add_argument("--datasplit_root", type=str, default="")
    args = parser.parse_args()

    from configs.config_v1 import ConfigV1 as Config
    cfg = Config()
    
    # 💡 인자로 들어온 값이 있으면 ConfigV1의 하드코딩 값을 덮어씁니다!
    cfg.input_csv = args.input_csv
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    if args.dataset_root:
        cfg.dataset_root = args.dataset_root
    if args.datasplit_root:
        cfg.datasplit_root = args.datasplit_root

    ckpt_file = './ckpt/SPAQ/Split_1/SRCC_Epoch_94_SRCC_0.9246_PCC_0.9265_MAE_6.1870.pth'
    cfg.init_model = ckpt_file
    main(cfg, args.output_csv)