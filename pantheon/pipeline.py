"""Portable Pantheon calibration, scoring and benchmark reproduction."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from . import core

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['FaceForensics++', 'Celeb-DF-v1', 'Celeb-DF-v2', 'Celeb-DF-v3',
            'DFDC', 'DFo', 'Wilddeepfake', 'FFIW', 'KoDF', 'DF40', 'DF-eval']
FEATURES = ['conf', 'q_std', 'q_margin', 'q_range']


def calibrate(data_dir=ROOT / 'data', output=ROOT / 'outputs/calibration'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    core.RBF_CENTERS = core.load_ffpp_rbf_centers_from_train(
        str(Path(data_dir) / 'standardized/FaceForensics++_standardized.csv'))
    df = pd.read_csv(Path(data_dir) / 'merged/merged_FaceForensics++.csv')
    df = core.build_df_common(core.filter_ffpp_c40_only(df), 'FaceForensics++')
    score = df.c23.to_numpy(float)
    df = df[~((score > .05) & (score < .95))]
    gate = core.fit_gate_hgb(df, FEATURES, seed=0)
    bundle = {'gate': gate, 'centers': core.RBF_CENTERS, 'features': FEATURES}
    joblib.dump(bundle, output / 'pantheon.joblib')
    return bundle


def load_bundle(path):
    bundle = joblib.load(path)
    core.RBF_CENTERS = bundle['centers']
    return bundle


def score_frames(df, bundle):
    core.RBF_CENTERS = bundle['centers']
    df = core.build_df_common(df, 'inference')
    score = df.c23.to_numpy(float)
    low = (score > .05) & (score < .95)
    pc = core.gate_predict_p_correct(bundle['gate'], df[FEATURES].to_numpy(float))
    df['ens_mix_high'] = core.build_mix_static(score, df.s_soft.to_numpy(float), low, .8)
    df['ens_adapt'], lam = core.build_mix_adaptive(
        score, df.s_soft.to_numpy(float), low, pc, .8, .2, .05, .4)
    df['gate_p_correct'] = pc
    df['gate_lambda'] = lam
    df['low_mask'] = low.astype(int)
    return df


def evaluate(data_dir, out, datasets, bundle, all_models=False):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in datasets:
        print(f'Evaluating {ds}', flush=True)
        df = pd.read_csv(data_dir / 'merged' / f'merged_{ds}.csv')
        df = score_frames(df, bundle)
        df['video_id'] = core.get_video_id_series(df, ds)
        df = core.filter_df40_video_frames(df, ds)
        jobs = core.make_ffpp_splits(df) if ds == 'FaceForensics++' else [(ds, df)]
        for name, frames in jobs:
            if frames.empty:
                raise ValueError(f'Empty evaluation split: {name}')
            strict, info = core.strict_video_filter_once(
                frames, name, str(out), save_examples=False, save_all_mixed_paths_txt=False)
            models = ['c23', 'ens_adapt']
            if all_models:
                models = [c for c in ['raw', 'c23', 'c40', 'all', 'q_routed',
                                     's_soft', 'ens_mix_high', 'ens_adapt'] if c in frames]
            for model in models:
                fm = core.metric_frame_auc_acc(frames, model)
                vm = core.compute_video_metrics_strict_from_frames(strict, model)
                rows.append({'dataset': name, 'model': model,
                             'frame_auc': None if fm['auc'] is None else fm['auc'] * 100,
                             'video_auc': None if vm['auc'] is None else vm['auc'] * 100,
                             'video_acc': None if vm['acc'] is None else vm['acc'] * 100,
                             'n_frames': len(frames), 'n_videos': vm['n_valid_videos'],
                             'mixed_videos': info['mixed_videos']})
        del df
    result = pd.DataFrame(rows)
    result.to_csv(out / 'results.csv', index=False)
    pantheon = result[result.model == 'ens_adapt'].copy()
    pantheon.to_csv(out / 'pantheon_table5.csv', index=False)
    cross = pantheon[~pantheon.dataset.str.startswith('FaceForensics++')]
    summary = {'cross_dataset_video_auc_mean': float(cross.video_auc.mean()),
               'cross_dataset_count': len(cross), 'evaluated_datasets': datasets}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    ref = pd.read_csv(ROOT / 'reference/ALL_results_frame_video_auc_with_deltas.csv')
    check = result.merge(ref[['dataset', 'model', 'video_auc']], on=['dataset', 'model'],
                         suffixes=('', '_reference'), validate='one_to_one')
    check['delta'] = check.video_auc - check.video_auc_reference
    check.to_csv(out / 'reference_comparison.csv', index=False)
    if len(check) != len(result) or not np.allclose(check.delta, 0, atol=1e-8):
        raise RuntimeError('Reference comparison failed; see reference_comparison.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('off')
    table = ax.table(cellText=[[f'{v:.2f}' for v in pantheon.video_auc] +
                               [f'{cross.video_auc.mean():.2f}']],
                     colLabels=pantheon.dataset.tolist() + ['Cross Avg.'],
                     rowLabels=['Pantheon(Effort)'], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 2)
    fig.savefig(out / 'table5.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(pantheon[['dataset', 'video_auc']].to_string(index=False))
    print(json.dumps(summary, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/reproduction')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--all-models', action='store_true')
    args = parser.parse_args()
    bundle = calibrate(args.data_dir, args.output / 'calibration')
    evaluate(args.data_dir, args.output, args.datasets, bundle, args.all_models)


if __name__ == '__main__':
    main()
