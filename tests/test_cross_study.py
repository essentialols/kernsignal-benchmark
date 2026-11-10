"""Tiny-fixture tests for the GSS adapter and frozen transfer.

Fixtures are invented and NOT evidence about GSS or any LLM.
"""
import json

import numpy as np
import pandas as pd
import pytest

from kernsignal import cross_study as cs
from kernsignal import paired_validation as pv

def gss_fixture(tmp_path):
    build = tmp_path / 'experiment' / 'datasets' / 'GSS' / 'build'
    (build / 'llm_runs').mkdir(parents=True)
    (build / 'robustness_runs').mkdir()
    meta = {'questions': {'happy': {'options': {'1': 'a', '2': 'b', '3': 'c'}}}}
    (build / 'gss_pilot_meta.json').write_text(json.dumps(meta))
    pilot = pd.DataFrame({'respondent_id': ['2016_1', '2018_2'], 'question_id': ['happy'] * 2,
                          'answer_code': [1, 3]})
    pilot.to_parquet(build / 'pilot_sample.parquet', index=False)
    truth = {'2016_1': 1, '2018_2': 3}
    for model in cs.GSS_MODELS:
        recs = []
        for seed in (0, 1):
            for rid, h in truth.items():
                recs.append({'model_name': model, 'style': 'A', 'seed': seed, 'respondent_id': rid,
                             'question_id': 'happy', 'human_answer': h, 'valid': True,
                             'pred': 9 if rid == '2018_2' else 2, 'dist': None})
                recs.append({'model_name': model, 'style': 'C', 'seed': seed, 'respondent_id': rid,
                             'question_id': 'happy', 'human_answer': h, 'valid': rid == '2016_1',
                             'pred': None, 'dist': {'1': .2, '3': .7, '2': .1}})
        (build / 'llm_runs' / f'{model}.jsonl').write_text('\n'.join(map(json.dumps, recs)))
    for model in cs.GSS_ROBUSTNESS_MODELS:
        recs = [{'model_name': model, 'variant': v, 'style': s, 'reverse': v == 'order', 'seed': 0,
                 'respondent_id': rid, 'question_id': 'happy', 'human_answer': h, 'valid': True, 'pred': 1}
                for v, s in (('order', 'A'), ('persona', 'B')) for rid, h in truth.items()]
        (build / 'robustness_runs' / f'{model}.jsonl').write_text('\n'.join(map(json.dumps, recs)))
    return tmp_path, build


def test_gss_adapter_rules(tmp_path):
    root, build = gss_fixture(tmp_path)
    long, prov = cs.load_gss(root)
    assert len(long) == 2 * (4 * 2 + 2 * 2)                  # seed 1 excluded
    haiku = long.loc[long.model.eq('claude-haiku-4.5')].set_index(['condition', 'respondent_id'])
    assert haiku.loc[('A', '2016_1'), 'synthetic'] == '2'
    assert haiku.loc[('A', '2018_2'), 'synthetic'] is None   # code 9 outside documented options
    assert haiku.loc[('C', '2016_1'), 'synthetic'] == '3'    # argmax of dist, source rule
    assert haiku.loc[('C', '2018_2'), 'synthetic'] is None   # invalid output
    assert set(long.condition) == {'A', 'C', 'A_options_reversed', 'B_persona'}
    assert 'WVS' in prov['excluded']
    # A cached human answer that disagrees with the pilot table must fail closed.
    path = build / 'llm_runs' / 'llama3.1-8b.jsonl'
    path.write_text(path.read_text().replace('"human_answer": 3', '"human_answer": 2', 1))
    with pytest.raises(ValueError):
        cs.load_gss(root)


def _twin_fixture(tmp_path):
    from tests.test_paired_validation import fixture_files
    files = fixture_files(tmp_path, n=1200, groups=12)
    frames, _ = pv._frames(files)
    return frames


def test_cell_features_identical_to_twin2k_recipe(tmp_path):
    frames = _twin_fixture(tmp_path)
    split = pv._split_person_ids(list(frames['synthetic'].index), seed=41, audit_fraction=.1)
    all_ids, audit_ids, truth_ids = split['test']
    for col in ('Q001', 'Q002', 'Q007'):
        obs, _ = pv._cell(col, 'F', frames, split['test'], min_truth_n=100)
        mine, _ = cs.cell_features(frames['synthetic'].loc[all_ids, col].tolist(),
                                   frames['human_wave4'].loc[audit_ids, col].tolist(),
                                   frames['human_wave4'].loc[truth_ids, col].tolist(),
                                   min_synthetic=100, min_truth=100, min_audit=25)
        assert mine['base'] == obs.base_features
        assert mine['audit'] == obs.audit_features
        assert mine['label'] == obs.label and mine['tvd'] == obs.discrepancy


def _cells(n_fam=12, per=6, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fam):
        for j in range(per):
            label = int(rng.random() < .5)
            row = {'family': f'F{f}', 'label': label, 'tvd': .3 if label else .05, 'flip': label,
                   'model': f'm{j % 2}', 'condition': f'c{j % 3}', 'subgroup': 'all'}
            row.update({f'base_{i}': float(rng.normal() + label * (i == 0)) for i in range(7)})
            row.update({f'audit_{i}': float(rng.normal() + label * (i == 7)) for i in range(12)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_frozen_transfer_never_refits():
    frozen = cs.freeze_twin2k(_cells(seed=1))
    before = json.dumps(cs.frozen_summary(frozen))
    target = _cells(seed=2)
    out = cs.transfer(target, frozen, ['synthetic_only', 'S1_shape_only_synthetic'])
    assert json.dumps(cs.frozen_summary(frozen)) == before
    assert out['twin2k_constant'] == frozen['prior']
    p = out['arms']['synthetic_only']
    assert p['bootstrap']['n_clusters'] == 12 and len(p['bootstrap']['family_balanced_brier_gain_95ci']) == 2
    # Changing only target labels must not change predictions (no recalibration).
    flipped = target.assign(label=1 - target.label)
    out2 = cs.transfer(flipped, frozen, ['synthetic_only'])
    assert out2['arms']['synthetic_only']['mean_predicted'] == p['mean_predicted']
    assert out['primary_decision'] in {'transfer_supported', 'inconclusive', 'transfer_worse_than_frozen_constant'}


def test_bootstrap_counts_duplicate_families_and_decision_rule():
    y = np.array([1, 0, 1, 1, 0, 0])
    fam = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
    boot = cs.family_bootstrap(y, np.full(6, .5), np.full(6, .5), fam, draws=50)
    assert boot['family_balanced_brier_gain_95ci'] == [0.0, 0.0]
    assert cs.decision([.01, .2]) == 'transfer_supported'
    assert cs.decision([-.2, -.01]) == 'transfer_worse_than_frozen_constant'
    assert cs.decision([-.1, .1]) == 'inconclusive'
    value, table = cs.ece([0, 1], [.05, .95])
    assert abs(value - .05) < 1e-12 and len(table) == 2


def test_within_study_family_disjoint():
    cells = _cells()
    out = cs.within_study(cells, cells, ['synthetic_only'])
    assert out['status'] == 'evaluated' and out['holdout_families'] == 12
    assert set(out['arms']) == {'synthetic_only'}
