"""Out-of-sample validation of *actual* LLM respondents against actual people.

This research module answers an explicit product question: How much predictive
value comes from synthetic-only diagnostics, historical human evidence, or a
small contemporary human audit?  The labels are previously hidden Wave 4
human answers, NOT injected fake distortions. No model is commercially approved.

An LLM's entire set of synthetic answers may be public to the auditor. For the
zero-human arm, absolutely no human answer counts are features. For the 10%
audit arm, the remaining 90% of Wave 4 humans are held out as truth.

Two leakage defenses are intentional: respondent-disjoint TRAIN/TEST and
question-family-disjoint GroupKFold. This is within ONE public study; it is NOT
cross-study/vendor validation. Clinical, causal, or respondent-level claims
are outside scope.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NULLS = frozenset({'', 'nan', 'none', 'null', 'n/a', 'na', 'not applicable'})
THRESHOLD = 0.15  # fixed before observing outcomes. Descriptive, not economic loss.
SEED = 20261009
BASE_FEATURES = ('synth_entropy', 'synth_margin', 'synth_top_share',
                 'synth_simpson', 'synth_k', 'synth_missing_fraction', 'synth_log_n')
AUDIT_FEATURES = BASE_FEATURES + ('audit_tvd', 'audit_modal_disagree',
                                  'audit_margin', 'audit_entropy_gap', 'audit_log_n')
HISTORY_FEATURES = BASE_FEATURES + ('previous_tvd', 'previous_modal_disagree',
                                    'previous_log_n')


def _hash_fraction(seed: int, prefix: str, label: str) -> float:
    h = hashlib.sha256(f'{seed}:{prefix}:{label}'.encode()).digest()
    return int.from_bytes(h[:8], 'big') / 2**64


def clean_value(v) -> str | None:
    """Canonicalize binary-compatibly, without inventing unseen code mappings."""
    if pd.isna(v):
        return None
    val = str(v).strip()
    if val.lower() in NULLS:
        return None
    try:
        f = float(val)
        if math.isfinite(f) and f.is_integer() and len(val) < 24:
            return str(int(f))
    except ValueError:
        pass
    return val if len(val) < 100 else None  # exclude open text: no NLP semantics assumed


def _counts(series: pd.Series) -> Counter:
    return Counter(x for x in series.map(clean_value).tolist() if x is not None)


def _probabilities(counts: Counter, categories: list[str]) -> np.ndarray:
    c = np.array([counts.get(x, 0) for x in categories], dtype=float)
    return c / c.sum() if c.sum() else np.zeros(len(categories))


def _entropy(p: np.ndarray) -> float:
    nonzero = p[p > 0]
    return float(-(nonzero * np.log2(nonzero)).sum()) if len(nonzero) else 0.0


def _tvd(a: Counter, b: Counter) -> float:
    cats = sorted(set(a) | set(b))
    return float(0.5 * np.abs(_probabilities(a, cats) - _probabilities(b, cats)).sum())


def _choice(a: Counter) -> str:
    return min(a.keys(), key=lambda k: (-a[k], k))


def _margin(a: Counter) -> float:
    p = np.sort(_probabilities(a, sorted(a)))
    return float(p[-1] - p[-2]) if len(p) > 1 else 1.0


def _candidate_fields(mapping: list[dict], frames: Mapping[str, pd.DataFrame]):
    unique = {}
    required = set(frames)
    for item in mapping:
        col, group = item.get('formatted_column'), item.get('QuestionID')
        if not isinstance(col, str) or not isinstance(group, str):
            continue
        if not all(col in frames[k].columns for k in required):
            continue
        if col in unique and unique[col] != group:
            raise ValueError(f'Ambiguous question mapping for {col}')
        unique[col] = group
    return sorted(unique.items())


def _frames(files: Mapping[str, str | Path]):
    frames = {key: pd.read_csv(files[key], low_memory=False, dtype=str)
              for key in ('synthetic', 'human_wave4', 'human_wave1_3')}
    for label, df in frames.items():
        if 'TWIN_ID' not in df.columns:
            raise ValueError(f'No TWIN_ID in {label}: raw header contract changed')
        # Export contains exactly one Qualtrics metadata row under some versions.
        df.drop(df.loc[df['TWIN_ID'].eq('TWIN_ID')].index, inplace=True)
        df['TWIN_ID'] = df['TWIN_ID'].str.strip()
        if df['TWIN_ID'].isna().any() or (df['TWIN_ID'] == '').any() or df['TWIN_ID'].duplicated().any():
            raise ValueError(f'Invalid respondent IDs in {label}')
        frames[label] = df.set_index('TWIN_ID', drop=True)
    common = sorted(set.intersection(*(set(frame.index) for frame in frames.values())))
    if len(common) < 100:
        raise ValueError(f'Insufficient aligned participants: {len(common)}')
    return {key: df.loc[common].copy() for key, df in frames.items()}, len(common)


@dataclass(frozen=True)
class Observation:
    question: str
    family: str
    label: int
    flip: bool
    discrepancy: float
    base_features: tuple[float, ...]
    audit_features: tuple[float, ...]
    history_features: tuple[float, ...]
    n_truth: int
    historical_human_tvd: float


def _split_person_ids(ids: list[str], seed: int, audit_fraction: float):
    if not 0.05 <= audit_fraction <= 0.30:
        raise ValueError('Audit fraction must be 5%..30%')
    train = {id for id in ids if _hash_fraction(seed, 'respondent_partition', id) < 0.60}
    test = set(ids) - train
    def audit_truth(group):
        audit = {id for id in group if _hash_fraction(seed, 'contemporary_audit', id) < audit_fraction}
        return sorted(audit), sorted(set(group) - audit)
    tr_a, tr_t = audit_truth(train)
    te_a, te_t = audit_truth(test)
    if not train.isdisjoint(test) or any(set(a) & set(b) for a,b in [(tr_a,tr_t),(te_a,te_t)]):
        raise AssertionError('Respondent leakage')
    return {'train': (sorted(train), tr_a, tr_t), 'test': (sorted(test), te_a, te_t)}


def _cell(col: str, family: str, frames: Mapping[str,pd.DataFrame],
          ids: tuple[list[str],list[str],list[str]], min_truth_n: int=100):
    all_ids, audit_ids, truth_ids = ids
    raw_synth = frames['synthetic'].loc[all_ids, col]
    synthetic, audit = _counts(raw_synth), _counts(frames['human_wave4'].loc[audit_ids, col])
    truth = _counts(frames['human_wave4'].loc[truth_ids, col])
    previous = _counts(frames['human_wave1_3'].loc[all_ids, col])
    # Eligibility is determined without looking at the held-out human labels.
    # The extra union check is recorded as a posthoc exclusion, not covertly tuned.
    if not (1 <= len(synthetic) <= 10) or sum(synthetic.values()) < 100:
        return None, 'ineligible_synthetic_type_or_n'
    if sum(audit.values()) < 25 or sum(truth.values()) < min_truth_n:
        return None, 'low_human_reference_n'
    if len(set(synthetic) | set(truth)) > 12:
        return None, 'uncomparable_response_domain'
    synt_cats=sorted(synthetic)
    ps=_probabilities(synthetic, synt_cats)
    n_s=sum(synthetic.values())
    base=(float(_entropy(ps)/math.log2(max(2,len(ps)))),_margin(synthetic),float(ps.max()),
          float((ps*ps).sum()), float(len(ps)),
          1.0-float(n_s)/len(raw_synth), math.log1p(n_s))
    p_a=_probabilities(audit, sorted(audit))
    audit_add=(float(_tvd(synthetic,audit)),float(_choice(synthetic)!=_choice(audit)),
               _margin(audit), abs(float(_entropy(ps)-_entropy(p_a))), math.log1p(sum(audit.values())))
    if not previous:
        return None, 'no_historical_human'
    history_add=(_tvd(synthetic, previous), float(_choice(synthetic)!=_choice(previous)),
                 math.log1p(sum(previous.values())))
    dist=_tvd(synthetic,truth)
    return Observation(col, family, int(dist>=THRESHOLD), _choice(synthetic)!=_choice(truth),
                       dist, base, base+audit_add, base+history_add,sum(truth.values()),_tvd(previous,truth)),None


def prepare(files: Mapping[str, str | Path], mapping_path: str | Path,
            seed: int=SEED, audit_fraction: float=0.1) -> tuple[pd.DataFrame,dict]:
    frames, people=_frames(files)
    mapping=json.loads(Path(mapping_path).read_text())
    fields=_candidate_fields(mapping, frames)
    split=_split_person_ids(list(frames['synthetic'].index), seed, audit_fraction)
    rows=[]
    exclusions=Counter()
    for label, ids in split.items():
        for col, family in fields:
            obs,reason=_cell(col,family,frames,ids,min_truth_n=100)
            if reason:
                exclusions[f'{label}:{reason}']+=1
                continue
            rows.append({'partition':label,'question':obs.question,'family':obs.family,
                         'label':obs.label,'flip':int(obs.flip),'tvd':obs.discrepancy,
                         'n_truth':obs.n_truth, 'historical_human_tvd':obs.historical_human_tvd,
                         **{f'base_{i}':x for i,x in enumerate(obs.base_features)},
                         **{f'audit_{i}':x for i,x in enumerate(obs.audit_features)},
                         **{f'history_{i}':x for i,x in enumerate(obs.history_features)}})
    metadata={'dataset':'LLM-Digital-Twin/Twin-2K-500',
              'respondents_aligned':people,'mapped_question_columns':len(fields),
              'eligible_cells':len(rows),'exclusions':dict(exclusions),
              'seed':seed, 'audit_fraction':audit_fraction,
              'train_respondents':len(split['train'][0]),
              'test_respondents':len(split['test'][0]),
              'train_question_families':len({r['family'] for r in rows if r['partition']=='train'}),
              'test_question_families':len({r['family'] for r in rows if r['partition']=='test'}),
              'label':'actual GPT4.1-mini vs withheld actual human Wave4 responses',
              'source_kind':'unverified_paired_inputs'}
    result=pd.DataFrame(rows)
    if result.empty:
        raise ValueError('No comparable questions. Inspect codebook/response codes first.')
    return result,metadata


def _weighted_brier(y,p,groups):
    group_sizes=Counter(groups)
    weights=np.asarray([1/group_sizes[g] for g in groups])
    return float(np.average((y-p)**2,weights=weights))


def _metrics(y,p,groups):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    n=len(y); preval=float(y.mean())
    ix=np.argsort(-p,kind='stable');k=max(1,math.ceil(.2*n))
    out={'n_cells':n,'n_families':len(set(groups)),
         'positive_rate':preval,
         'family_balanced_brier':_weighted_brier(y,p,groups),
         'cell_brier':float(np.mean((y-p)**2)),
         'roc_auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
         'average_precision':float(average_precision_score(y,p)) if len(set(y))==2 else None,
         'precision_top_20pct':float(y[ix[:k]].mean()),
         'lift_top_20pct':float(y[ix[:k]].mean()/preval) if preval else None}
    return out


def evaluate(frame: pd.DataFrame, metadata: dict, seed: int=SEED) -> dict:
    """Five question-family-disjoint folds, using separate respondents for fitting and scoring."""
    train=frame.loc[frame.partition=='train']
    test=frame.loc[frame.partition=='test']
    families=sorted(set(train.family) & set(test.family))
    if len(families)<10:
        return {'status':'insufficient_evidence','reason':'fewer than 10 eligible question families',
                'question_families':len(families), 'metadata':metadata}
    train=train.loc[train.family.isin(families)].copy()
    test=test.loc[test.family.isin(families)].copy()
    test_y=[];pred={k:[] for k in ('constant','synthetic_only','audit_10pct','historical')};
    families_out=[];tvd_out=[];human_tvd=[];flips=[]
    folds=GroupKFold(n_splits=min(5,len(families)))
    for fold,(_,test_ind) in enumerate(folds.split(test, groups=test.family)):
        test_groups=set(test.iloc[test_ind].family)
        tr=train.loc[~train.family.isin(test_groups)]
        te=test.loc[test.family.isin(test_groups)]
        if len(tr)<12 or len(te)<2:
            raise ValueError('Not enough disjoint questions in one fold')
        # Family-balanced like the scoring, so the constant is not penalized by
        # matrix questions with many sub-items.
        prior=float(np.average(tr.label,weights=1/tr.family.map(tr.family.value_counts())))
        if not (0<prior<1):
            return {'status':'insufficient_evidence','reason':'training label lacks both classes',
                    'question_families':len(families),'metadata':metadata}
        test_y.extend(te.label.astype(int).tolist())
        tvd_out.extend(te.tvd.tolist());human_tvd.extend(te.historical_human_tvd.tolist());flips.extend(te.flip.astype(int).tolist())
        families_out.extend(te.family.tolist())
        pred['constant'].extend([prior]*len(te))
        for arm,prefix in [('synthetic_only','base_'),('audit_10pct','audit_'),('historical','history_')]:
            features=sorted([x for x in tr.columns if x.startswith(prefix)],key=lambda x:int(x.split('_')[-1]))
            pipe=make_pipeline(StandardScaler(),LogisticRegression(C=.25,max_iter=1000,random_state=seed+fold))
            family_counts=tr.family.value_counts()
            weights=(1/tr.family.map(family_counts)).to_numpy()
            pipe.fit(tr[features].to_numpy(dtype=float),tr.label.to_numpy(dtype=int),logisticregression__sample_weight=weights)
            pred[arm].extend(pipe.predict_proba(te[features].to_numpy(dtype=float))[:,1].tolist())
    ys=np.asarray(test_y)
    results={name:_metrics(ys,np.asarray(p),families_out) for name,p in pred.items()}
    baseline=results['constant']['family_balanced_brier']
    for x in results.values():x['gain_vs_constant_family_balanced']=baseline-x['family_balanced_brier']
    # Fold priors come from other question families and can run opposite to the
    # held-out ones; report the uninformative 0.5 forecast as a fixed reference too.
    half=_weighted_brier(ys,np.full(len(ys),.5),families_out)
    for x in results.values():x['gain_vs_half_family_balanced']=half-x['family_balanced_brier']
    # Cluster-bootstrap improvement vs a constant rate, accounting for matrix
    # subitems that share a question family (IID row CI would be spuriously tight).
    groups_by_name={name:np.flatnonzero(np.asarray(families_out)==name) for name in set(families_out)}
    rng=np.random.default_rng(seed+200)
    interval={}
    for name, probs in pred.items():
        if name=='constant': continue
        improvements=[]
        for _ in range(300):
            draws=rng.choice(sorted(groups_by_name),size=len(groups_by_name),replace=True)
            idx=np.concatenate([groups_by_name[q] for q in draws])
            pp=np.asarray(probs)[idx]
            p0=np.asarray(pred['constant'])[idx]
            improvements.append(float(np.mean((ys[idx]-p0)**2-(ys[idx]-pp)**2)))
        interval[name]=[float(v) for v in np.quantile(improvements,[.025,.975])]
        results[name]['cluster_bootstrap_cell_brier_gain_95ci']=interval[name]
    # Post-hoc sensitivity analysis: how much does a 10% human audit improve
    # probabilistic prioritization over the truly synthetic-only arm? Resample
    # question families (the independence unit), not individual matrix cells.
    # This comparison was added after the first empirical run, and is explicitly
    # exploratory rather than a prespecified confirmatory finding.
    group_differences=[]
    base_prob=np.asarray(pred['synthetic_only'], dtype=float)
    audit_prob=np.asarray(pred['audit_10pct'], dtype=float)
    for key in sorted(groups_by_name):
        ix=groups_by_name[key]
        group_differences.append(float(np.mean((ys[ix]-base_prob[ix])**2 -
                                               (ys[ix]-audit_prob[ix])**2)))
    group_differences=np.asarray(group_differences)
    paired_boot=np.random.default_rng(seed+300).choice(
        group_differences, size=(1000,len(group_differences)), replace=True).mean(axis=1)
    audit_increment={'metric':'family_balanced_brier_gain_from_10pct_audit',
                     'value':float(group_differences.mean()),
                     'cluster_bootstrap_95ci':list(map(float,np.quantile(paired_boot,[.025,.975]))),
                     'exploratory_after_first_empirical_run':True,
                     'cluster_unit':'question_family'}
    distinct=len(set(families_out))
    notes=['Feature eligibility is based primarily on observed synthetic categories; category definitions should be verified with the official codebook.',
           'This is one public study. Question family and respondent are held out; survey, vendor, and cultural generalization remain untested.',
           'A 10% human audit represents actual human evidence, not a free or human-independent test.',
           'Historical arm uses real Wave1-3 responses to the same items; it is not a zero-human method.',
           'The 0.15 TVD definition is a statistical discrepancy flag, not a demonstrated monetary loss or decision reversal.',
           'Cluster bootstrap resamples question families; uncertainty is still conditional on this study and fixed respondent partition.',
           'No threshold selection or model approval is performed using the test outcomes.']
    return {'status':'evaluated','commercially_approved':False,
            'evaluation':'real LLM outputs vs withheld human survey responses',
            'holdout_question_families':distinct,'holdout_cells':len(test_y),
            'actual_tvd_mean':float(np.mean(tvd_out)),
            'actual_modal_flip_rate':float(np.mean(flips)),
            'historical_human_vs_human_tvd_mean':float(np.mean(human_tvd)),
            'reference_half_family_balanced_brier':half,
            'arms':results,'audit_increment':audit_increment,'metadata':metadata,'limitations':notes}


def save_report(result: dict, path: str | Path, files: Mapping[str,str | Path], revision: str,
                source_path: str | Path | None=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    result=dict(result)
    result['provenance']={'revision':revision,
                          'source_file_sha256':{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in files.items()},
                          'code_sha256':hashlib.sha256(Path(source_path or __file__).read_bytes()).hexdigest(),
                          'license':'CC BY 4.0','raw_respondent_data_in_report':False}
    path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return result