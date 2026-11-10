"""Generate DEMO-ONLY (not LLM) panels with known, targeted survey distortions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kernsignal.reference import make_demo, read_reference, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='data/demo_pair')
    ap.add_argument('--respondents', type=int, default=3200)
    ap.add_argument('--seed', type=int, default=41)
    args = ap.parse_args()
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    human_path, synthetic_path = folder/'human.csv', folder/'synthetic.csv'
    make_demo(human_path, n=args.respondents, seed=args.seed)
    rng = np.random.default_rng(args.seed + 100)
    synthetic = read_reference(human_path).copy()
    # Both panels simulated for the demonstration; these labels denote roles.
    synthetic['respondent_id'] = 'SIMULATED-LLM-' + synthetic['respondent_id']
    # Age-specific distortion deliberately hidden by the overall mean.
    elderly = synthetic['age_band'].eq('65+') & synthetic['question_id'].eq('Q2')
    younger = synthetic['age_band'].eq('18-29') & synthetic['question_id'].eq('Q2')
    synthetic.loc[elderly & (rng.random(len(synthetic)) < .36), 'answer'] = 5
    synthetic.loc[younger & (rng.random(len(synthetic)) < .36), 'answer'] = 1
    mask = synthetic.question_id.eq('Q3') & (rng.random(len(synthetic)) < .55)
    synthetic.loc[mask, 'answer'] = 3  # compressed variance
    synthetic.to_csv(synthetic_path, index=False)
    manifest = {'dataset_id': 'kernsignal-demo-pair', 'source_kind': 'simulated_human',
                'human_sha256': sha256(human_path),
                'synthetic_sha256': sha256(synthetic_path),
                'warning': 'Neither panel contains real respondents or outputs from an LLM.'}
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()