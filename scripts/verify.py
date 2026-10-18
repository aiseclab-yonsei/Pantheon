"""Check packaged files, original-source metadata and reproduced Table 5 values."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hashes', action='store_true')
    parser.add_argument('--check-originals', action='store_true')
    args = parser.parse_args()
    report = {}
    manifest = json.loads((ROOT / 'reference/copy_manifest.json').read_text())
    for row in manifest:
        target = ROOT / row['destination']
        assert target.is_file(), target
        if args.check_originals:
            original = Path(row['source'])
            assert original.stat().st_size == row['bytes'], original
            assert original.stat().st_mtime_ns == row['source_mtime_ns'], original
        if args.hashes and row.get('sha256'):
            assert sha256(target) == row['sha256'], target
    report['packaged_manifest_files'] = len(manifest)
    report['original_size_and_mtime_unchanged'] = args.check_originals
    table = pd.read_csv(ROOT / 'outputs/reproduction/pantheon_table5.csv')
    ref = pd.read_csv(ROOT / 'reference/ALL_results_frame_video_auc_with_deltas.csv')
    ref = ref[ref.model == 'ens_adapt']
    check = table.merge(ref[['dataset', 'video_auc']], on='dataset', suffixes=('', '_reference'))
    assert len(check) == len(ref) == 13
    delta = float(abs(check.video_auc - check.video_auc_reference).max())
    assert delta < 1e-8, delta
    cross = table[~table.dataset.str.startswith('FaceForensics++')]
    assert len(cross) == 10
    assert np.isclose(cross.video_auc.mean(), 90.61374343684233, atol=1e-8)
    report['max_video_auc_delta_percentage_points'] = delta
    report['cross_dataset_video_auc_mean'] = float(cross.video_auc.mean())
    (ROOT / 'outputs/verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
