"""Offline fixture tests, explicitly NOT independent evidence of real LLM validity."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from kernsignal.paired_validation import (
    clean_value, prepare, evaluate, _split_person_ids, _cell, THRESHOLD,
)


def fixture_files(tmp_path, n=1200, groups=36):
    ids=[str(i+1000) for i in range(n)]
    rng=np.random.default_rng(41)
    frames={typ:{'TWIN_ID':ids} for typ in ('synthetic','human_wave4','human_wave1_3')}
    maps=[]
    for q in range(groups):
        key=f'Q{q+1:03d}'
        maps.append({'formatted_column':key,'QuestionID':f'QID{q+1}'})
        human=rng.choice([1,2,3,4],size=n,p=[.15,.2,.5,.15])
        synth=human.copy()
        if q%2==0: synth=np.where(rng.random(n)<.55,1,synth)
        else: synth=np.where(rng.random(n)<.04,2,synth)  # below the 0.15 TVD flag, so both label classes exist
        frames['human_wave4'][key]=human.astype(str)
        frames['human_wave1_3'][key]=np.where(rng.random(n)<.1,3,human).astype(str)
        frames['synthetic'][key]=synth.astype(str)
    paths={}
    for kind,data in frames.items():
        p=tmp_path/(kind+'.csv');pd.DataFrame(data).to_csv(p,index=False);paths[kind]=p
    paths['mapping']=tmp_path/'mapping.json'
    paths['mapping'].write_text(json.dumps(maps))
    paths['catalog']=tmp_path/'catalog.json'
    paths['catalog'].write_text('{}')
    return paths


def test_two_axis_leakage_protection():
    ids=[str(i) for i in range(800)]
    split=_split_person_ids(ids,seed=31,audit_fraction=.1)
    train, test=split['train'],split['test']
    assert set(train[0]).isdisjoint(set(test[0]))
    for all_ids,audit,truth in [train,test]:
        assert not set(audit)&set(truth)
        assert set(audit)|set(truth)==set(all_ids)


def test_preprocess_and_evaluate(tmp_path):
    files=fixture_files(tmp_path)
    df, meta=prepare(files,files['mapping'],seed=41,audit_fraction=.1)
    assert meta['source_kind']=='unverified_paired_inputs'  # fixture NEVER attests publication
    assert not df.empty
    assert df['label'].between(0,1).all()
    assert not (df.partition=='train').sum()==0
    for part in ('train', 'test'):
        assert meta[f'{part}_question_families']==df.loc[df.partition==part,'family'].nunique()>0
    r=evaluate(df,meta,seed=41)
    assert r['status']=='evaluated', r.get('reason')
    if r['status']=='evaluated':
        assert r['commercially_approved'] is False
        assert r['reference_half_family_balanced_brier']==pytest.approx(.25)
        assert all('gain_vs_half_family_balanced' in a for a in r['arms'].values())
        assert set(r['arms'])=={'constant','synthetic_only','audit_10pct','historical'}
        assert r['holdout_question_families']>=10
        assert np.isfinite(r['arms']['constant']['family_balanced_brier'])


def test_synthetic_only_features_ignore_human_values(tmp_path):
    files=fixture_files(tmp_path,n=1200,groups=12)
    import kernsignal.paired_validation as p
    frames, _=p._frames(files)
    spl=p._split_person_ids(list(frames['synthetic'].index),seed=41,audit_fraction=.1)
    obs,_=p._cell('Q001','QID1',frames,spl['test'],min_truth_n=100)
    assert obs
    f0=obs.base_features
    # Alter ONLY the human comparison sample. It can change labels and audit
    # predictors but MUST NOT change the synthetic-only predictor inputs.
    changed={k:v.copy() for k,v in frames.items()}
    changed['human_wave4']['Q001']='4'
    obs2,_=p._cell('Q001','QID1',changed,spl['test'],min_truth_n=100)
    assert obs2.base_features == f0


def test_low_support_synthetic_not_excluded(tmp_path):
    files=fixture_files(tmp_path,n=1200,groups=12)
    import kernsignal.paired_validation as p
    frames,_=p._frames(files)
    frames['synthetic']['Q001']='2'
    sp=p._split_person_ids(list(frames['synthetic'].index),seed=41,audit_fraction=.1)
    o,reason=p._cell('Q001','QID1',frames,sp['train'],min_truth_n=100)
    assert o is not None,reason
    assert o.base_features[4] == 1


def test_threshold_and_value_cleaning_are_fixed():
    assert THRESHOLD==0.15
    assert clean_value('3.0')=='3'
    assert clean_value('nan') is None


def test_exploratory_audit_increment_has_family_cluster_ci(tmp_path):
    from kernsignal.paired_validation import evaluate
    import pandas as pd
    # Many families and enough rows per arm: exercise the new schema contract.
    rows=[]
    for i in range(20):
        for split in ('train','test'):
            label = int(i % 3 != 0)
            row={'partition':split, 'question':f'Q{i}', 'family':f'F{i}',
                 'label':label,'tvd':.26 if label else .05,
                 'historical_human_tvd':.03, 'flip':int(i % 4 == 0)}
            row.update({f'base_{j}':float(i%5+j*.1) for j in range(3)})
            row.update({f'audit_{j}':float(i%5+j*.1+(.04 if label else 0)) for j in range(3)})
            row.update({f'history_{j}':float(i%5+j*.1+(.09 if label else 0)) for j in range(3)})
            rows.append(row)
    output=evaluate(pd.DataFrame(rows), {'dataset':'fixture'},seed=44)
    assert output['status']=='evaluated'
    x=output['audit_increment']
    assert x['exploratory_after_first_empirical_run'] is True
    assert x['cluster_unit']=='question_family'
    assert len(x['cluster_bootstrap_95ci'])==2
    assert abs(x['value']-(output['arms']['synthetic_only']['family_balanced_brier']-
                           output['arms']['audit_10pct']['family_balanced_brier']))<1e-10