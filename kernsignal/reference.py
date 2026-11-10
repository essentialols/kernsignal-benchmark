"""Reference-panel validation and the simulated demo panel generator."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ('respondent_id', 'question_id', 'answer', 'age_band', 'region', 'weight')
DEMO = 'simulated_human'


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_reference(df: pd.DataFrame, min_respondents: int = 100) -> None:
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if df.empty or df['respondent_id'].nunique() < min_respondents:
        raise ValueError(f'Need at least {min_respondents} distinct human respondents')
    for field in ['respondent_id', 'question_id', 'age_band', 'region']:
        if df[field].isna().any() or (df[field].astype(str).str.len() == 0).any():
            raise ValueError(f'{field} contains missing values')
    if df.duplicated(['respondent_id', 'question_id']).any():
        raise ValueError('Duplicate respondent × question rows (possible replay/data leakage)')
    answers = pd.to_numeric(df['answer'], errors='coerce')
    if answers.isna().any() or (answers < 1).any() or (answers > 10).any() or (answers % 1 != 0).any():
        raise ValueError('Answers must be integer categories 1..10')
    weights = pd.to_numeric(df['weight'], errors='coerce')
    if weights.isna().any() or ~np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Weights must be positive and finite')
    if df['question_id'].nunique() > 100:
        raise ValueError('Unexpected number of questions; check normalization mapping')


def read_reference(path: str | Path, min_respondents: int = 100) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'respondent_id': str, 'question_id': str, 'age_band': str, 'region': str})
    validate_reference(df, min_respondents=min_respondents)
    df['answer'] = df['answer'].astype(int)
    df['weight'] = df['weight'].astype(float)
    return df


def make_demo(path: str | Path, n: int = 3200, seed: int = 41) -> dict:
    """Simulation ONLY. Not a public human dataset or a validation claim."""
    if n < 600:
        raise ValueError('Demo needs at least 600 respondents for disjoint holdouts')
    rng = np.random.default_rng(seed)
    records = []
    ages = np.array(['18-29', '30-44', '45-64', '65+'])
    regions = np.array(['Northeast', 'Midwest', 'South', 'West'])
    for i in range(n):
        age = str(rng.choice(ages))
        region = str(rng.choice(regions))
        age_shift = {'18-29': -0.30, '30-44': -0.12, '45-64': 0.15, '65+': 0.38}[age]
        region_shift = {'Northeast': 0.0, 'Midwest': 0.15, 'South': -0.12, 'West': -0.05}[region]
        for q, mean in [('Q1', 2.9), ('Q2', 3.2), ('Q3', 2.6), ('Q4', 3.4)]:
            ans = int(np.clip(np.rint(rng.normal(mean + age_shift + region_shift, 1.2)), 1, 5))
            records.append((f'DEMO-{i:06d}', q, ans, age, region, 1.0))
    df = pd.DataFrame(records, columns=REQUIRED)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    manifest = {
        'dataset_id': 'kernsignal-simulated-human-v0', 'source_kind': DEMO,
        'source_url': None, 'release_date': None, 'simulation_seed': seed,
        'respondents': n, 'rows': len(df), 'sha256': sha256(path),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'warning': 'Simulated human-labelled demo, NOT evidence of performance on real human respondents.',
    }
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
