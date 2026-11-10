"""Inspect public Twin-2K-500 paired human/GPT results without printing raw rows.

Downloads only the published GPT-4.1-mini wave-4 triplet, mappings and
question catalog; pins every file to one Hugging Face commit SHA.
Data are ephemeral and gitignored. Never upload respondent-level CSVs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

REPO = 'LLM-Digital-Twin/Twin-2K-500'
BASE = 'LLM_simulation_results/GPT4.1-mini-simulation-llm-vs-human/'
FILES = {
    'synthetic': BASE + 'responses_llm_imputed_formatted.csv',
    'human_wave4': BASE + 'responses_wave4_formatted.csv',
    'human_wave1_3': BASE + 'responses_wave1_3_formatted.csv',
    'mapping': 'LLM_simulation_results/wave4_formatted_to_catalog_mapping.json',
    'catalog': 'question_catalog_and_human_response_csv/question_catalog.json',
}


def obtain(cache_dir: Path, revision: str | None = None, offline: bool = False) -> dict:
    """Return file paths and immutable repo revision; never trust mutable 'main'."""
    if offline:
        return {key: cache_dir / Path(name).name for key, name in FILES.items()}
    from huggingface_hub import HfApi, hf_hub_download
    revision = revision or HfApi().repo_info(REPO, repo_type='dataset').sha
    if not revision or len(revision) != 40:
        raise ValueError('Must resolve immutable Hugging Face dataset commit SHA')
    out = {}
    for key, file in FILES.items():
        out[key] = Path(hf_hub_download(REPO, filename=file, repo_type='dataset', revision=revision, local_dir=cache_dir))
    return out


def inspect(files: dict, revision: str, destination: Path) -> dict:
    import pandas as pd
    synth = pd.read_csv(files['synthetic'], low_memory=False)
    human = pd.read_csv(files['human_wave4'], low_memory=False)
    previous = pd.read_csv(files['human_wave1_3'], low_memory=False)
    mapping = json.loads(Path(files['mapping']).read_text())
    catalog = json.loads(Path(files['catalog']).read_text())
    if not isinstance(mapping, list) or not mapping or not all(isinstance(x, dict) for x in mapping):
        raise ValueError('Unexpected Twin mapping JSON structure')
    # Qualtrics-formatted exports may contain a first metadata/header row.
    # Only count participant IDs; NEVER print them, timestamps, or raw answers.
    for frame in (synth, human, previous):
        if 'TWIN_ID' not in frame.columns:
            raise ValueError('Expected Twin respondent key TWIN_ID missing')
        if frame['TWIN_ID'].astype(str).eq('TWIN_ID').any():
            frame.drop(index=frame.loc[frame['TWIN_ID'].astype(str).eq('TWIN_ID')].index, inplace=True)
        if frame['TWIN_ID'].isna().any() or frame['TWIN_ID'].duplicated().any():
            raise ValueError('Duplicate/missing participant IDs; fail closed')
    matched_questions = sorted({entry['formatted_column'] for entry in mapping
                                if entry.get('formatted_column') in synth.columns
                                and entry.get('formatted_column') in human.columns})
    if len(matched_questions) < 20:
        raise ValueError(f'Unexpectedly few mapped question columns: {len(matched_questions)}')
    # Aggregate-only summary of categorical fields. Deliberately no raw values,
    # even if a question unexpectedly contains identifiers or free text.
    preview = {}
    for label, frame in [('synthetic', synth), ('human_wave4', human), ('human_wave1_3', previous)]:
        preview[label] = {
            'respondent_count': int(len(frame)),
            'column_count': int(len(frame.columns)),
            'participant_id_key': 'TWIN_ID',
            'duplicate_ids': int(frame['TWIN_ID'].duplicated().sum()),
        }
    for col in matched_questions[:20]:
        sh, hh = synth[col], human[col]
        preview[f'question::{col}'] = {
            'synthetic_unique_nonnull': int(sh.nunique()),
            'human_unique_nonnull': int(hh.nunique()),
            'synthetic_nonnull': int(sh.notna().sum()),
            'human_nonnull': int(hh.notna().sum()),
            'numeric_fraction_synthetic': round(float(pd.to_numeric(sh, errors='coerce').notna().mean()),3),
            'numeric_fraction_human': round(float(pd.to_numeric(hh, errors='coerce').notna().mean()),3),
        }
    result = {
        'dataset': REPO,
        'revision_sha': revision,
        'license': 'CC BY 4.0',
        'files': {key: {'name': str(FILES[key]), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(), 'bytes': Path(path).stat().st_size} for key, path in files.items()},
        'schema': preview,
        'matched_columns': len(matched_questions),
        'mapping_entries': len(mapping),
        'mapping_first_five': mapping[:5],
        'catalog_type': type(catalog).__name__,
        'catalog_first_key_names': list(catalog)[:10] if isinstance(catalog, dict) else [],
        'raw_records_saved': False,
        'source': 'actual LLM outputs and actual human respondents, published paired dataset',
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-dir', default='data/twin2k')
    ap.add_argument('--output', default='artifacts/twin2k/inspection.json')
    ap.add_argument('--revision', default=None)
    ap.add_argument('--offline', action='store_true')
    args = ap.parse_args()
    from huggingface_hub import HfApi
    rev = args.revision or ('offline-fixture' if args.offline else HfApi().repo_info(REPO, repo_type='dataset').sha)
    files = obtain(Path(args.cache_dir), rev, offline=args.offline)
    data = inspect(files, rev, Path(args.output))
    print(json.dumps({k: data[k] for k in ('dataset','revision_sha','matched_columns','mapping_entries','schema')}, indent=2)[:8500])


if __name__ == '__main__':
    main()