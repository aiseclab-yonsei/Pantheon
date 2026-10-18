"""Apply a calibrated Pantheon gate to an arbitrary merged prediction CSV."""
import argparse
from pathlib import Path
import pandas as pd
from .pipeline import ROOT, calibrate, load_bundle, score_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration', type=Path,
                        default=ROOT / 'outputs/reproduction/calibration/pantheon.joblib')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    required = {'path', 'crfiqa', 'qcn', 'quadrant', 'c23', 'q1', 'q2', 'q3', 'q4'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(df.columns))}')
    if 'label' not in df:
        df['label'] = 0
    bundle = load_bundle(args.calibration) if args.calibration.is_file() else calibrate(
        output=args.output.parent / 'calibration')
    result = score_frames(df, bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Saved {len(result)} frame predictions to {args.output}')


if __name__ == '__main__':
    main()
