"""Study-disjoint (cross-study) validation of the discrepancy-risk model.

The synthetic-only risk model of ``paired_validation`` is fitted on
Twin-2K-500 only, frozen, and used to score cells of an independent published
human-vs-LLM study (GSS attitude items) without refitting or recalibrating.
The analysis plan was fixed before any target outcome was computed.

Adapters turn each source into one paired long format:
``respondent_id, item, family, model, condition, synthetic, human`` where
``synthetic`` is the model's categorical answer for that respondent (None when
the model output was invalid) and ``human`` is the same respondent's real
answer. Only aggregate metadata ever leaves this module.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kernsignal.paired_validation import (
    SEED, THRESHOLD, _choice, _counts, _entropy, _hash_fraction, _margin,
    _metrics, _probabilities, _tvd, _weighted_brier,
)

LONG_COLUMNS = ('respondent_id', 'item', 'family', 'model', 'condition', 'synthetic', 'human')
AUDIT_FRACTION = 0.10
BOOTSTRAP_DRAWS = 2000
ECE_BINS = 10
# Column indices into the Twin-2K feature vectors (paired_validation._cell order).
BASE_ALL = tuple(range(7))
AUDIT_ALL = tuple(range(12))
SHAPE_BASE = (0, 1, 2, 3, 4)                 # entropy, margin, top share, Simpson, k
SHAPE_AUDIT = SHAPE_BASE + (7, 8, 9, 10)      # + audit TVD, modal flip, margin, entropy gap

GSS_COMMIT = '76aa0376835b60c5dd54851a6fc4e77e32753465'
GSS_MODELS = ('claude-haiku-4.5', 'claude-sonnet-4.6', 'llama3.1-8b', 'llama3.3-70b')
GSS_ROBUSTNESS_MODELS = ('claude-haiku-4.5', 'claude-sonnet-4.6')
ELIGIBILITY = {'gss': {'min_synthetic': 80, 'min_truth': 80, 'min_audit': None}}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shallow_commit(repo: str | Path) -> str | None:
    """Commit of a shallow clone, read as text (never runs git in untrusted repos)."""
    f = Path(repo) / '.git' / 'shallow'
    return f.read_text().split()[0] if f.exists() else None


# ----------------------------------------------------------------- GSS adapter
def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def gss_pred_code(rec: dict):
    """Mirror of the source's metrics._pred_code (argmax of dist for style C)."""
    if rec['style'] == 'C':
        return int(max(rec['dist'], key=rec['dist'].get)) if rec.get('dist') else None
    return rec.get('pred')


def load_gss(root: str | Path) -> tuple[pd.DataFrame, dict]:
    root = Path(root)
    build = root / 'experiment' / 'datasets' / 'GSS' / 'build'
    pilot = pd.read_parquet(build / 'pilot_sample.parquet')
    meta = json.loads((build / 'gss_pilot_meta.json').read_text())
    options = {q: {int(k) for k in v['options']} for q, v in meta['questions'].items()}
    pilot['respondent_id'] = pilot.respondent_id.astype(str)
    if pilot.duplicated(['respondent_id', 'question_id']).any() or not pilot.question_id.isin(options).all():
        raise ValueError('GSS pilot sample contract changed')
    if not all(int(a) in options[q] for q, a in zip(pilot.question_id, pilot.answer_code)):
        raise ValueError('GSS human answer outside documented options')
    truth = {(r, q): int(a) for r, q, a in zip(pilot.respondent_id, pilot.question_id, pilot.answer_code)}
    specs = [(m, f'llm_runs/{m}.jsonl', {'A': {'style': 'A'}, 'C': {'style': 'C'}}) for m in GSS_MODELS]
    specs += [(m, f'robustness_runs/{m}.jsonl', {'A_options_reversed': {'variant': 'order'},
                                                  'B_persona': {'variant': 'persona'}})
              for m in GSS_ROBUSTNESS_MODELS]
    rows, files = [], {}
    for model, rel, conditions in specs:
        path = build / rel
        files[rel] = path
        records = _read_jsonl(path)
        for condition, match in conditions.items():
            chosen = [r for r in records if r.get('seed') == 0 and r.get('model_name') == model
                      and all(r.get(k) == v for k, v in match.items())]
            seen = set()
            for rec in chosen:
                key = (str(rec['respondent_id']), rec['question_id'])
                if key in seen or key not in truth:
                    raise ValueError(f'GSS run not uniquely joinable to pilot: {model} {condition}')
                seen.add(key)
                if int(rec['human_answer']) != truth[key]:
                    raise ValueError('GSS cached human_answer disagrees with pilot answer_code')
                code = gss_pred_code(rec) if rec.get('valid') else None
                code = code if code is not None and int(code) in options[key[1]] else None
                rows.append({'respondent_id': key[0], 'item': key[1], 'family': key[1], 'model': model,
                             'condition': condition, 'synthetic': None if code is None else str(int(code)),
                             'human': str(truth[key])})
            if len(seen) != len(truth):
                raise ValueError(f'GSS {model} {condition} does not cover the pilot sample')
    files['pilot_sample.parquet'] = build / 'pilot_sample.parquet'
    files['gss_pilot_meta.json'] = build / 'gss_pilot_meta.json'
    prov = {'source': 'github ZihanChen1995/when-synthetic-users-fail-a-cross-domain-benchmark-of-llm-simulated-human-survey-responses', 'pinned_commit': GSS_COMMIT,
            'shallow_commit_on_disk': shallow_commit(root),
            'source_file_sha256': {k: sha256(v) for k, v in sorted(files.items())},
            'license': 'GSS-derived tables and model outputs CC BY 4.0 (LICENSE-DATA); Llama outputs under Llama 3.x community licenses',
            'included': 'GSS pilot_sample (eval fold), seed 0; styles A and C (all 4 models), order/persona variants (Claude models)',
            'excluded': {'WVS': 'LICENSE-DATA: WVS-derived build files incl. model outputs are WVSA non-profit replication only, no redistribution',
                         'seed 1': 'second draw of the same respondents',
                         'subgroups': 'about 99 people per item cannot be split and keep n>=80'}}
    return pd.DataFrame(rows, columns=list(LONG_COLUMNS)), prov


# ----------------------------------------------------------------- cell builder
def cell_features(synth_raw: list, audit_raw: list | None, truth_raw: list, *, min_synthetic: int,
                  min_truth: int, min_audit: int | None):
    """Byte-for-byte the Twin-2K ``paired_validation._cell`` feature recipe."""
    synthetic = _counts(pd.Series(synth_raw, dtype=object))
    truth = _counts(pd.Series(truth_raw, dtype=object))
    audit = _counts(pd.Series(audit_raw, dtype=object)) if audit_raw is not None else None
    if not (1 <= len(synthetic) <= 10) or sum(synthetic.values()) < min_synthetic:
        return None, 'ineligible_synthetic_type_or_n'
    if sum(truth.values()) < min_truth or (min_audit is not None and sum(audit.values()) < min_audit):
        return None, 'low_human_reference_n'
    if len(set(synthetic) | set(truth)) > 12:
        return None, 'uncomparable_response_domain'
    ps = _probabilities(synthetic, sorted(synthetic))
    n_s = sum(synthetic.values())
    base = (float(_entropy(ps) / math.log2(max(2, len(ps)))), _margin(synthetic), float(ps.max()),
            float((ps * ps).sum()), float(len(ps)), 1.0 - float(n_s) / len(synth_raw), math.log1p(n_s))
    audit_add = None
    if min_audit is not None:
        p_a = _probabilities(audit, sorted(audit))
        audit_add = (float(_tvd(synthetic, audit)), float(_choice(synthetic) != _choice(audit)),
                     _margin(audit), abs(float(_entropy(ps) - _entropy(p_a))), math.log1p(sum(audit.values())))
    dist = _tvd(synthetic, truth)
    return {'label': int(dist >= THRESHOLD), 'tvd': dist, 'flip': int(_choice(synthetic) != _choice(truth)),
            'n_truth': sum(truth.values()), 'n_synthetic': n_s, 'base': base,
            'audit': None if audit_add is None else base + audit_add}, None


def build_cells(long: pd.DataFrame, subgroups: dict[str, frozenset], audit_ids: frozenset | None,
                eligibility: dict, partition: str = 'all') -> tuple[pd.DataFrame, Counter]:
    rows, exclusions = [], Counter()
    use_audit = eligibility['min_audit'] is not None
    for (item, family, model, condition), g in long.groupby(['item', 'family', 'model', 'condition'], sort=True):
        ids = g.respondent_id.to_numpy()
        in_audit = np.isin(ids, list(audit_ids)) if use_audit else np.zeros(len(g), bool)
        for sub, members in subgroups.items():
            m = np.isin(ids, list(members))
            if not m.any():
                continue
            obs, reason = cell_features(
                g.synthetic.to_numpy()[m].tolist(),
                g.human.to_numpy()[m & in_audit].tolist() if use_audit else None,
                g.human.to_numpy()[m & ~in_audit].tolist(), **eligibility)
            if reason:
                exclusions[f'{partition}:{reason}'] += 1
                continue
            row = {'partition': partition, 'item': item, 'family': family, 'model': model,
                   'condition': condition, 'subgroup': sub,
                   **{k: obs[k] for k in ('label', 'tvd', 'flip', 'n_truth', 'n_synthetic')},
                   **{f'base_{i}': x for i, x in enumerate(obs['base'])}}
            if obs['audit'] is not None:
                row.update({f'audit_{i}': x for i, x in enumerate(obs['audit'])})
            rows.append(row)
    return pd.DataFrame(rows), exclusions


def audit_split(ids, seed: int = SEED, fraction: float = AUDIT_FRACTION) -> frozenset:
    """Same hash and prefix as the Twin-2K contemporary audit split."""
    return frozenset(i for i in ids if _hash_fraction(seed, 'contemporary_audit', i) < fraction)


# --------------------------------------------------------------- model + metrics
def feature_columns(prefix: str, idx) -> list[str]:
    return [f'{prefix}_{i}' for i in idx]


def fit_risk_model(frame: pd.DataFrame, columns: list[str], seed: int = SEED):
    """Equal-family-weighted logistic regression, the paired_validation recipe."""
    weights = (1 / frame.family.map(frame.family.value_counts())).to_numpy()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=.25, max_iter=1000, random_state=seed))
    pipe.fit(frame[columns].to_numpy(dtype=float), frame.label.to_numpy(dtype=int),
             logisticregression__sample_weight=weights)
    return pipe


def ece(y, p, bins: int = ECE_BINS) -> tuple[float, list[dict]]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    table, total = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        gap = abs(p[m].mean() - y[m].mean())
        total += m.mean() * gap
        table.append({'bin': [b / bins, (b + 1) / bins], 'n_cells': int(m.sum()),
                      'mean_predicted': float(p[m].mean()), 'observed_rate': float(y[m].mean())})
    return float(total), table


def _lift(y, p):
    k = max(1, math.ceil(.2 * len(y)))
    top = y[np.argsort(-p, kind='stable')[:k]].mean()
    return float(top / y.mean()) if y.mean() else None


def family_bootstrap(y, p, p_const, families, draws: int = BOOTSTRAP_DRAWS, seed: int = SEED) -> dict:
    """Percentile CIs, resampling question families (a family drawn twice counts twice)."""
    y, p, p_const = np.asarray(y, int), np.asarray(p, float), np.asarray(p_const, float)
    families = np.asarray(families)
    index = {f: np.flatnonzero(families == f) for f in sorted(set(families))}
    names = sorted(index)
    rng = np.random.default_rng(seed)
    stats = {k: [] for k in ('family_balanced_brier_gain', 'cell_brier_gain', 'roc_auc',
                             'average_precision', 'lift_top_20pct', 'ece')}
    skipped = 0
    for _ in range(draws):
        pick = rng.choice(len(names), size=len(names), replace=True)
        idx = np.concatenate([index[names[j]] for j in pick])
        groups = np.concatenate([[f'{slot}'] * len(index[names[j]]) for slot, j in enumerate(pick)])
        yy, pp, cc = y[idx], p[idx], p_const[idx]
        stats['family_balanced_brier_gain'].append(_weighted_brier(yy, cc, groups) - _weighted_brier(yy, pp, groups))
        stats['cell_brier_gain'].append(float(np.mean((yy - cc) ** 2 - (yy - pp) ** 2)))
        stats['ece'].append(ece(yy, pp)[0])
        if len(set(yy)) == 2:
            stats['roc_auc'].append(float(roc_auc_score(yy, pp)))
            stats['average_precision'].append(float(average_precision_score(yy, pp)))
            stats['lift_top_20pct'].append(_lift(yy, pp))
        else:
            skipped += 1
    out = {f'{k}_95ci': [float(x) for x in np.quantile(v, [.025, .975])] if v else None
           for k, v in stats.items()}
    out.update({'draws': draws, 'draws_without_both_classes': skipped, 'cluster_unit': 'question_family',
                'n_clusters': len(names)})
    return out


def score_arm(y, p, families, p_const, oracle: float | None = None, bootstrap: bool = True) -> dict:
    y, p = np.asarray(y, int), np.asarray(p, float)
    out = _metrics(y, p, list(families))
    const = _metrics(y, np.asarray(p_const, float), list(families))
    out['constant_family_balanced_brier'] = const['family_balanced_brier']
    out['family_balanced_brier_gain_vs_constant'] = const['family_balanced_brier'] - out['family_balanced_brier']
    out['cell_brier_gain_vs_constant'] = const['cell_brier'] - out['cell_brier']
    out['ece_10'], out['reliability'] = ece(y, p)
    out['mean_predicted'] = float(p.mean())
    if oracle is not None:
        out['oracle_constant_family_balanced_brier'] = _weighted_brier(y, np.full(len(y), oracle), list(families))
    if bootstrap and len(set(families)) > 1:
        out['bootstrap'] = family_bootstrap(y, p, p_const, families)
    return out


def decision(ci: list[float] | None) -> str:
    if not ci:
        return 'not_evaluable'
    return 'transfer_supported' if ci[0] > 0 else 'transfer_worse_than_frozen_constant' if ci[1] < 0 else 'inconclusive'


# ----------------------------------------------------------------- analyses
ARMS = {'synthetic_only': ('base', BASE_ALL), 'audit_10pct': ('audit', AUDIT_ALL),
        'S1_shape_only_synthetic': ('base', SHAPE_BASE), 'S1_shape_only_audit': ('audit', SHAPE_AUDIT)}


def freeze_twin2k(train: pd.DataFrame, seed: int = SEED) -> dict:
    """Fit every arm once on Twin-2K cells. The result is never refitted."""
    frozen = {'prior': float(train.label.mean()), 'n_cells': len(train),
              'n_families': int(train.family.nunique()), 'models': {}, 'columns': {}}
    for arm, (prefix, idx) in ARMS.items():
        cols = feature_columns(prefix, idx)
        frozen['models'][arm] = fit_risk_model(train, cols, seed)
        frozen['columns'][arm] = cols
    return frozen


def frozen_summary(frozen: dict) -> dict:
    out = {}
    for arm, pipe in frozen['models'].items():
        sc, lr = pipe.named_steps['standardscaler'], pipe.named_steps['logisticregression']
        out[arm] = {'features': frozen['columns'][arm], 'scaler_mean': sc.mean_.round(6).tolist(),
                    'scaler_scale': sc.scale_.round(6).tolist(), 'coef': lr.coef_[0].round(6).tolist(),
                    'intercept': round(float(lr.intercept_[0]), 6)}
    return out


def transfer(cells: pd.DataFrame, frozen: dict, arms: list[str]) -> dict:
    y, fam = cells.label.to_numpy(int), cells.family.to_numpy()
    const = np.full(len(cells), frozen['prior'])
    oracle = float(y.mean())
    result = {'n_cells': len(cells), 'n_families': int(cells.family.nunique()), 'target_prevalence': oracle,
              'twin2k_constant': frozen['prior'], 'mean_tvd': float(cells.tvd.mean()),
              'modal_flip_rate': float(cells.flip.mean()), 'arms': {}, 'breakdowns': {}}
    preds = {}
    for arm in arms:
        preds[arm] = frozen['models'][arm].predict_proba(cells[frozen['columns'][arm]].to_numpy(dtype=float))[:, 1]
        result['arms'][arm] = score_arm(y, preds[arm], fam, const, oracle=oracle)
    primary = result['arms']['synthetic_only']
    result['primary_decision'] = decision(primary['bootstrap']['family_balanced_brier_gain_95ci'])
    auc_ci = primary['bootstrap']['roc_auc_95ci']
    result['discrimination_decision'] = ('auc_ci_above_0.5' if auc_ci and auc_ci[0] > .5 else
                                         'auc_ci_below_0.5' if auc_ci and auc_ci[1] < .5 else 'auc_inconclusive')
    for by in ('model', 'condition', 'subgroup'):
        if cells[by].nunique() < 2:
            continue
        result['breakdowns'][by] = {}
        for level, ix in cells.groupby(by).indices.items():
            yy = y[ix]
            entry = {'n_cells': int(len(ix)), 'prevalence': float(yy.mean()),
                     'mean_tvd': float(cells.tvd.to_numpy()[ix].mean())}
            for arm in arms:
                s = score_arm(yy, preds[arm][ix], fam[ix], const[ix], oracle=float(yy.mean()),
                              bootstrap=(by != 'subgroup'))
                s.pop('reliability')
                entry[arm] = s
            result['breakdowns'][by][str(level)] = entry
    return result


def within_study(train: pd.DataFrame, test: pd.DataFrame, arms: list[str], seed: int = SEED) -> dict:
    """Question-family GroupKFold (as paired_validation.evaluate); train/test may be respondent-disjoint."""
    families = sorted(set(train.family) & set(test.family))
    if len(families) < 5:
        return {'status': 'insufficient_evidence', 'question_families': len(families)}
    train, test = train.loc[train.family.isin(families)], test.loc[test.family.isin(families)]
    pred = {k: [] for k in ['constant', *arms]}
    ys, fams = [], []
    for fold, (_, ti) in enumerate(GroupKFold(n_splits=min(5, len(families))).split(test, groups=test.family)):
        held = set(test.iloc[ti].family)
        tr, te = train.loc[~train.family.isin(held)], test.loc[test.family.isin(held)]
        prior = float(tr.label.mean())
        if not 0 < prior < 1:
            return {'status': 'insufficient_evidence', 'reason': 'training fold lacks both classes'}
        ys += te.label.astype(int).tolist()
        fams += te.family.tolist()
        pred['constant'] += [prior] * len(te)
        for arm in arms:
            prefix, idx = ARMS[arm]
            cols = feature_columns(prefix, idx)
            pipe = fit_risk_model(tr, cols, seed + fold)
            pred[arm] += pipe.predict_proba(te[cols].to_numpy(dtype=float))[:, 1].tolist()
    ys = np.asarray(ys)
    return {'status': 'evaluated', 'holdout_cells': len(ys), 'holdout_families': len(set(fams)),
            'prevalence': float(ys.mean()),
            'arms': {arm: score_arm(ys, np.asarray(pred[arm]), fams, np.asarray(pred['constant']))
                     for arm in arms}}


def tvd_values(a: list, b: list) -> float:
    return _tvd(Counter(a), Counter(b))


def gss_noise_floor(root: str | Path, long: pd.DataFrame) -> dict:
    """Paired ~99 humans vs the remaining eval-fold humans answering the same item."""
    build = Path(root) / 'experiment' / 'datasets' / 'GSS' / 'build'
    full = pd.read_parquet(build / 'gss_pilot_long.parquet', columns=['respondent_id', 'question_id', 'answer_code'])
    splits = pd.read_parquet(build / 'splits.parquet')
    full['respondent_id'] = full.respondent_id.astype(str)
    evalids = set(splits.loc[splits.fold.eq('eval'), 'respondent_id'].astype(str))
    pilot = long.drop_duplicates(['respondent_id', 'item'])
    out = {}
    for item, g in pilot.groupby('item'):
        rest = full.loc[full.question_id.eq(item) & full.respondent_id.isin(evalids)
                        & ~full.respondent_id.isin(set(g.respondent_id))]
        out[item] = {'tvd': tvd_values(g.human.tolist(), rest.answer_code.astype(int).astype(str).tolist()),
                     'n_pilot': len(g), 'n_rest': len(rest)}
    v = np.array([x['tvd'] for x in out.values()])
    return {'kind': 'paired pilot humans vs other eval-fold humans (sampling floor)', 'per_item': out,
            'mean_tvd': float(v.mean()), 'max_tvd': float(v.max()), 'share_ge_threshold': float((v >= THRESHOLD).mean()),
            'source_file_sha256': {'gss_pilot_long.parquet': sha256(build / 'gss_pilot_long.parquet'),
                                   'splits.parquet': sha256(build / 'splits.parquet')}}
