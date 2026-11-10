"""Offline tests for the scoring primitives and the demo scorecard."""
import json

import pandas as pd
import pytest

from kernsignal.reference import make_demo, read_reference, validate_reference
from kernsignal.scoring import InvalidCell, score_counts, weighted_counts, FEATURES


def test_distribution_metrics_have_expected_bounds_and_decisions():
    score = score_counts([50, 50, 0], [0, 50, 50], 70, 80)
    assert score.status == 'scored'
    assert score.tvd == pytest.approx(.5)
    assert 0 < score.js_divergence <= 1
    assert score.mean_gap == pytest.approx(.5)
    assert score.audit_choice == 1  # lowest-index tie-break
    assert score.synthetic_choice == 2
    assert score.winner_disagreement
    assert len(score.features()) == len(FEATURES)


def test_low_effective_n_abstains_even_with_large_weighted_counts():
    score = score_counts([5000, 1000], [1000, 5000], 10, 15)
    assert score.status == 'insufficient_evidence'
    with pytest.raises(InvalidCell):
        score_counts([-1, 1], [1, 0])
    with pytest.raises(InvalidCell):
        score_counts([1, 2], [1, 2, 3])
    with pytest.raises(InvalidCell):
        score_counts([float('nan'), 1], [1, 2])


def test_weighted_kish_effective_n():
    frame = pd.DataFrame({'answer': [1, 2, 2], 'weight': [1., 2., 3.]})
    counts, neff = weighted_counts(frame, k=2)
    assert counts.tolist() == [1., 5.]
    assert neff == pytest.approx(36/14)


def test_duplicate_response_detection(tmp_path):
    p = tmp_path / 'human.csv'
    make_demo(p, n=650)
    df = read_reference(p)
    bad = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match='Duplicate respondent'):
        validate_reference(bad)


def test_scorecard_detects_targeted_subgroup_failure(tmp_path):
    from kernsignal.validation import score_panels
    human_path = tmp_path/'human.csv'
    synth_path = tmp_path/'synthetic.csv'
    make_demo(human_path, n=1200)
    synth = read_reference(human_path).copy()
    synth['respondent_id'] = 's-' + synth.respondent_id
    affected = (synth['age_band'].eq('65+')) & synth['question_id'].eq('Q2')
    synth.loc[affected, 'answer'] = 5
    synth.to_csv(synth_path, index=False)
    result = score_panels(human_path, synth_path, tmp_path/'score')
    report = json.loads((tmp_path/'score'/'scorecard.json').read_text())
    assert result['cells_scored'] >= 20
    assert result['distribution_divergence'] >= 1
    assert any(r['question_id'] == 'Q2' and r['subgroup'] == '65+'
               and r['tvd'] > .2 and 'distribution_divergence' in r['flags']
               for r in report['cells'] if r['status'] == 'scored')
    assert 'limitations' in report
