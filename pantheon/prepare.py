"""Normalize raw quality scores and join frame predictions by explicit path keys."""
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .quality import normalize_quality


def key(path):
    path = str(path).replace('\\', '/')
    match = re.search(r'faceforensics\+\+', path, re.I)
    if match:
        path = path[match.start():]
        return re.sub(r'^faceforensics\+\+(_[^/]+)?', 'FaceForensics++', path, count=1, flags=re.I)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality', type=Path, required=True,
                        help='CSV with path,cr_fiqa,qcn (raw scores)')
    parser.add_argument('--predictions', type=Path, required=True,
                        help='CSV with path,label,c23,q1,q2,q3,q4; raw,c40,all optional')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    quality = pd.read_csv(args.quality)
    pred = pd.read_csv(args.predictions)
    quality = quality.rename(columns={c: 'path' for c in ['rel_path', 'image_path']
                                      if c in quality and 'path' not in quality})
    pred = pred.drop(columns=['crfiqa', 'qcn', 'quadrant'], errors='ignore')
    for df, cols in [(quality, ['path', 'cr_fiqa', 'qcn']),
                     (pred, ['path', 'label', 'c23', 'q1', 'q2', 'q3', 'q4'])]:
        if not set(cols).issubset(df.columns):
            raise ValueError(f'Required columns: {cols}')
        df['key'] = df.path.map(key)
        if df.key.duplicated().any():
            raise ValueError('Duplicate frame keys; resolve duplicates before merging.')
    cr, qcn = normalize_quality(quality.cr_fiqa.to_numpy(float), quality.qcn.to_numpy(float))
    qdf = pd.DataFrame({'key': quality.key, 'crfiqa': cr, 'qcn': qcn})
    merged = pred.merge(qdf, on='key', how='left', validate='one_to_one')
    if merged[['crfiqa', 'qcn']].isna().any().any():
        raise ValueError('Some prediction paths have no matching quality score.')
    required = ['crfiqa', 'qcn', 'c23', 'q1', 'q2', 'q3', 'q4']
    if not np.isfinite(merged[required].to_numpy(float)).all():
        raise ValueError('Quality scores and predictions must be finite.')
    if not merged.label.isin([0, 1]).all():
        raise ValueError('Expected binary labels (0=real, 1=fake).')
    merged['quadrant'] = np.select([(merged.crfiqa >= 0) & (merged.qcn >= 0),
                                    (merged.crfiqa < 0) & (merged.qcn >= 0),
                                    (merged.crfiqa < 0) & (merged.qcn < 0)], [1, 2, 3], default=4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.drop(columns='key').to_csv(args.output, index=False)
    print(f'Saved {len(merged)} frames to {args.output}')


if __name__ == '__main__':
    main()
