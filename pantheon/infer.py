"""Run QCN, CR-FIQA(L), five Effort models and Pantheon on face crops."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from .pipeline import ROOT, calibrate, load_bundle, score_frames
from .quality import normalize_quality


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quality_scores(paths, device, amp=True):
    sys.path.insert(0, str(ROOT / 'vendor/qcn'))
    from network.model_ct_v29 import CTV29
    cfg = SimpleNamespace(backbone='resnet50', reduced_dim=256,
                          dec_layers=[1, 1, 1], reference_point_num=101)
    net = CTV29(cfg)
    state = torch.load(ROOT / 'weights/qcn_spaq.pth', map_location='cpu', weights_only=False)
    net.load_state_dict(state['model_state_dict'], strict=True)
    net.to(device).eval()
    scoring = load_module('qcn_scoring', ROOT / 'vendor/qcn/score.py')
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize(384),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
    qcn = []
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as im:
                image = transform(im.convert('RGB')).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp and device.type == 'cuda'):
                f = net('extraction', {'img': image})
                flipped = net('extraction', {'img': image.flip(-1)})
                f = torch.cat([f, flipped]).mean(0, keepdim=True).squeeze()
                f = f.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 101)
                features, pivots = net('get_cluster', {'f': f})
                value = scoring.compute_score_v1(
                    features.transpose(1, 2)[:, -1:], pivots.transpose(1, 2),
                    torch.zeros(features.shape[0], 1, device=device), list(range(101)))
                qcn.append(float(value.item()))
    del net, state
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    backbone = load_module('crfiqa_iresnet', ROOT / 'vendor/crfiqa/backbones/iresnet.py')
    net = backbone.iresnet100(num_features=512, qs=1, use_se=False)
    net.load_state_dict(torch.load(ROOT / 'weights/crfiqa_l.pth', map_location='cpu',
                                  weights_only=True), strict=True)
    net.to(device).eval()
    crfiqa = []
    with torch.inference_mode():
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f'Cannot read image: {path}')
            image = cv2.resize(image, (112, 112))
            tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().unsqueeze(0)
            tensor = tensor.to(device).div(255).sub(.5).div(.5)
            _, quality = net(tensor)
            crfiqa.append(float(quality.item()))
    del net
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.asarray(crfiqa), np.asarray(qcn)




def effort_model(checkpoint, device):
    """Fold saved SVD residuals into dense layers; no pretrained download/SVD needed."""
    from transformers import CLIPVisionConfig, CLIPVisionModel
    cfg = json.loads((ROOT / 'reference/clip_config.json').read_text())['vision_config']
    model = CLIPVisionModel(CLIPVisionConfig(**cfg))
    model.set_attn_implementation('eager')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = state.get('state_dict', state)
    dense = {}
    for key, value in state.items():
        if not key.startswith('backbone.'):
            continue
        key_out = 'vision_model.' + key[len('backbone.'):]
        if key.endswith('.weight_main'):
            prefix = key.removesuffix('weight_main')
            value = value + (state[prefix + 'U_residual'] *
                             state[prefix + 'S_residual'].unsqueeze(0)) @ state[prefix + 'V_residual']
            key_out = key_out.removesuffix('weight_main') + 'weight'
        elif key.rsplit('.', 1)[-1] in {'S_residual', 'U_residual', 'V_residual', 'S_r', 'U_r', 'V_r'}:
            continue
        dense[key_out] = value
    model.load_state_dict(dense, strict=True)
    head = torch.nn.Linear(1024, 2)
    head.load_state_dict({key: state['head.' + key] for key in ['weight', 'bias']})
    return model.to(device).eval(), head.to(device).eval()


def effort_scores(paths, device, keys):
    result = {}
    mean = torch.tensor([.48145466, .4578275, .40821073], device=device)[:, None, None]
    std = torch.tensor([.26862954, .26130258, .27577711], device=device)[:, None, None]
    images = []
    for path in paths:
        im = cv2.imread(str(path))
        if im is None:
            raise ValueError(f'Cannot read image: {path}')
        im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_CUBIC)
        tensor = torch.from_numpy(im.transpose(2, 0, 1).copy()).float().to(device) / 255
        images.append((tensor - mean) / std)
    with torch.inference_mode():
        for key in keys:
            print(f'Effort {key}', flush=True)
            model, head = effort_model(ROOT / 'weights/effort' / f'{key}.pth', device)
            values = []
            for tensor in images:
                feature = model(tensor.unsqueeze(0)).pooler_output
                values.append(float(head(feature).softmax(1)[0, 1].item()))
            result[key] = values
            del model, head
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'demo/input.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/image_demo')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--calibration', type=Path, default=ROOT / 'outputs/reproduction/calibration/pantheon.joblib')
    parser.add_argument('--all-models', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    df = pd.read_csv(args.input)
    paths = [(args.input.resolve().parent / p).resolve() if not Path(p).is_absolute()
             else Path(p) for p in df.path]
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError('Input must contain valid image paths (relative to input CSV).')
    if 'label' not in df:
        df['label'] = 0
    df['label'] = df.label.astype(int)
    if not df.label.isin([0, 1]).all():
        raise ValueError('Labels must be 0 (real) or 1 (fake).')
    args.output.mkdir(parents=True, exist_ok=True)
    crfiqa, qcn = quality_scores(paths, device, amp=not args.no_amp)
    raw = pd.DataFrame({'path': df.path, 'cr_fiqa': crfiqa, 'qcn': qcn})
    raw.to_csv(args.output / 'quality_raw.csv', index=False)
    df['crfiqa'], df['qcn'] = normalize_quality(crfiqa, qcn)
    df['quadrant'] = np.select([(df.crfiqa >= 0) & (df.qcn >= 0),
                                (df.crfiqa < 0) & (df.qcn >= 0),
                                (df.crfiqa < 0) & (df.qcn < 0)], [1, 2, 3], default=4)
    keys = ['c23', 'q1', 'q2', 'q3', 'q4']
    if args.all_models:
        keys += ['raw', 'c40', 'all']
    for key, values in effort_scores(paths, device, keys).items():
        df[key] = values
    df.to_csv(args.output / 'merged_predictions.csv', index=False)
    bundle = load_bundle(args.calibration) if args.calibration.is_file() else calibrate(
        output=args.output / 'calibration')
    result = score_frames(df, bundle)
    result['pantheon_pred'] = (result.ens_adapt >= .5).astype(int)
    result.to_csv(args.output / 'frame_predictions.csv', index=False)
    if 'video_id' in result:
        result.groupby('video_id', as_index=False).agg(
            pantheon_score=('ens_adapt', 'mean'), frames=('ens_adapt', 'count')).to_csv(
                args.output / 'video_predictions.csv', index=False)
    print(result[['path', 'crfiqa', 'qcn', 'c23', 'ens_adapt']].to_string(index=False))


if __name__ == '__main__':
    main()
