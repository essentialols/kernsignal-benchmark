"""Cross-study transfer validation (see docs/TRANSFER_GSS.md).

Fits the risk model on Twin-2K-500 only, freezes it, and scores GSS cells
without refitting. Inputs are local gitignored copies; the output JSON holds
aggregate metadata only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kernsignal import cross_study as cs
from kernsignal.paired_validation import SEED, THRESHOLD, prepare

TWIN_REVISION = 'f883165a3026fde855dfd448e0cd16443ab257b6'
TWIN_BASE = 'LLM_simulation_results/GPT4.1-mini-simulation-llm-vs-human/'


def twin2k_files(cache: Path) -> dict:
    return {'synthetic': cache / TWIN_BASE / 'responses_llm_imputed_formatted.csv',
            'human_wave4': cache / TWIN_BASE / 'responses_wave4_formatted.csv',
            'human_wave1_3': cache / TWIN_BASE / 'responses_wave1_3_formatted.csv',
            'mapping': cache / 'LLM_simulation_results/wave4_formatted_to_catalog_mapping.json'}


def run(twin_dir: Path, gss_dir: Path) -> dict:
    files = twin2k_files(twin_dir)
    train, twin_meta = prepare({k: files[k] for k in ('synthetic', 'human_wave4', 'human_wave1_3')},
                               files['mapping'], seed=SEED, audit_fraction=cs.AUDIT_FRACTION)
    frozen = cs.freeze_twin2k(train)
    report = {
        'status': 'evaluated', 'commercially_approved': False, 'model_registered': False,
        'target_tvd_threshold': THRESHOLD, 'seed': SEED, 'bootstrap_draws': cs.BOOTSTRAP_DRAWS,
        'training': {'dataset': 'LLM-Digital-Twin/Twin-2K-500', 'revision': TWIN_REVISION,
                     'cells': frozen['n_cells'], 'families': frozen['n_families'],
                     'constant_prevalence': frozen['prior'], 'exclusions': twin_meta['exclusions'],
                     'source_file_sha256': {k: cs.sha256(v) for k, v in files.items()},
                     'frozen_models': cs.frozen_summary(frozen)},
        'targets': {}}

    glong, gprov = cs.load_gss(gss_dir)
    gcells, gexcl = cs.build_cells(glong, {'all': frozenset(glong.respondent_id)}, None, cs.ELIGIBILITY['gss'])
    report['targets']['gss'] = {
        'provenance': gprov, 'eligibility': cs.ELIGIBILITY['gss'],
        'respondents': int(glong.respondent_id.nunique()), 'items': sorted(glong['item'].unique()),
        'models': sorted(glong.model.unique()), 'conditions': sorted(glong.condition.unique()),
        'cells_considered': int(len(gcells) + sum(gexcl.values())), 'exclusions': dict(gexcl),
        'excluded_cells': sorted(set(map(tuple, glong[['model', 'condition']].drop_duplicates().to_numpy()))
                                 - set(map(tuple, gcells[['model', 'condition']].drop_duplicates().to_numpy()))),
        'audit_arm': 'not evaluated: 10% of about 99 people per item is below the 25-person audit minimum',
        'transfer': cs.transfer(gcells, frozen, ['synthetic_only', 'S1_shape_only_synthetic']),
        'within_study': {'design': '5-fold question-family GroupKFold, no respondent split (about 99 people per item)',
                         **cs.within_study(gcells, gcells, ['synthetic_only'])},
        'human_noise_floor': cs.gss_noise_floor(gss_dir, glong)}
    report['excluded_studies'] = {'WVS (same GSS repository)': gprov['excluded']['WVS']}
    code = [Path(cs.__file__), Path(cs.__file__).with_name('paired_validation.py'), Path(__file__)]
    report['code_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    report['raw_respondent_data_in_report'] = False
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--twin2k-dir', type=Path, default=Path('data/twin2k'))
    ap.add_argument('--gss-dir', type=Path, default=Path('data/gss-when-synthetic-users-fail'))
    ap.add_argument('--output', type=Path, default=Path('artifacts/cross-study/gss_transfer.json'))
    args = ap.parse_args()
    report = run(args.twin2k_dir, args.gss_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    for name, t in report['targets'].items():
        p = t['transfer']['arms']['synthetic_only']
        print(name, t['transfer']['n_cells'], t['transfer']['primary_decision'],
              round(p['family_balanced_brier_gain_vs_constant'], 4), p['bootstrap']['family_balanced_brier_gain_95ci'],
              p['roc_auc'], p['bootstrap']['roc_auc_95ci'])


if __name__ == '__main__':
    main()
