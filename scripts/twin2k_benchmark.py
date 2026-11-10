"""Real human/LLM paired benchmark; raw downloads remain local and gitignored."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from kernsignal.paired_validation import SEED, prepare, evaluate, save_report
from scripts.inspect_twin2k import obtain, REPO


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-dir',default='data/twin2k')
    ap.add_argument('--output',default='artifacts/twin2k/validation.json')
    ap.add_argument('--seed',type=int,default=SEED)
    ap.add_argument('--audit-fraction',type=float,default=.10)
    ap.add_argument('--revision')
    ap.add_argument('--offline',action='store_true')
    args=ap.parse_args()
    if args.offline:
        revision=args.revision or 'offline-test-fixture'
    else:
        from huggingface_hub import HfApi
        revision=args.revision or HfApi().repo_info(REPO,repo_type='dataset').sha
    files=obtain(Path(args.cache_dir),revision,offline=args.offline)
    frame,metadata=prepare(files,files['mapping'],seed=args.seed,audit_fraction=args.audit_fraction)
    # A local CSV fixture must never be relabeled as public human evidence.
    if not args.offline and revision == 'f883165a3026fde855dfd448e0cd16443ab257b6':
        metadata['source_kind']='published_human_and_llm_outputs'
        metadata['publication_provenance']='Hugging Face immutable repo revision + file hashes, not a cryptographic publisher attestation'
    result=evaluate(frame,metadata,seed=args.seed)
    written=save_report(result,args.output,files,revision)
    safe={k:written.get(k) for k in ('status','evaluation','holdout_question_families','holdout_cells','actual_tvd_mean','actual_modal_flip_rate','arms','audit_increment','limitations')}
    print(json.dumps(safe,indent=2))
    if result['status']!='evaluated':
        raise SystemExit('Insufficient cross-question evidence. See report for exact reason.')

if __name__=='__main__':
    main()