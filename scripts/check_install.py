"""Verify numerical libraries, inference imports and a real device operation."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--backend', default='cpu')
    args = parser.parse_args()
    versions = {}
    for name in ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'joblib']:
        module = importlib.import_module(name)
        versions[name] = module.__version__
    from sklearn.ensemble import HistGradientBoostingClassifier
    import numpy as np
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        model = HistGradientBoostingClassifier(max_iter=2, min_samples_leaf=1).fit(
            np.array([[0.], [1.], [2.], [3.]]), [0, 0, 1, 1])
        assert np.isfinite(model.predict_proba([[1.5]])).all()
    import pantheon.core
    if args.mode == 'gpu':
        import torch
        import torchvision
        import timm
        import transformers
        import cv2
        from transformers import CLIPVisionConfig, CLIPVisionModel
        import pantheon.infer
        sys.path.insert(0, str(ROOT / 'vendor/qcn'))
        from network.model_ct_v29 import CTV29
        device = 'cuda:0' if args.backend != 'cpu' else 'cpu'
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA wheel installed, but CUDA is unavailable. Check the NVIDIA driver '
                               'or use bash setup.sh inference-cpu.')
        tensor = torch.eye(4, device=device)
        assert torch.equal(tensor @ tensor, tensor)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        versions.update(torch=torch.__version__, torchvision=torchvision.__version__,
                        timm=timm.__version__, transformers=transformers.__version__,
                        opencv=cv2.__version__, device=device)
        if device.startswith('cuda'):
            versions['gpu'] = torch.cuda.get_device_name(0)
    print(json.dumps(versions, indent=2), flush=True)
    print('Package imports and runtime check: PASS', flush=True)
    required = [ROOT / 'data/merged/merged_FaceForensics++.csv',
                ROOT / 'data/standardized/FaceForensics++_standardized.csv']
    if args.mode == 'gpu':
        required += [ROOT / 'weights/qcn_spaq.pth', ROOT / 'weights/crfiqa_l.pth']
        required += [ROOT / 'weights/effort' / f'{key}.pth' for key in ['c23', 'q1', 'q2', 'q3', 'q4']]
    missing = [str(file.relative_to(ROOT)) for file in required if not file.is_file()]
    if missing:
        print('Environment is ready. Separate data/weights are still needed:\n  ' + '\n  '.join(missing))


if __name__ == '__main__':
    main()
