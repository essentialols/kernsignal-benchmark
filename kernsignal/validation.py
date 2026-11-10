"""Commercially useful validation scorecard; no fitted model or LLM required.

This is the product's dependable core. Risk probability is optional; observable
subgroup divergence, effective sample size and decisions remain independently
interpretable and evidence-backed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from kernsignal.reference import read_reference, sha256
from kernsignal.scoring import score_counts, weighted_counts


def _load_panel(path: str | Path):
    panel = read_reference(path)
    return panel


def score_panels(human_path: str | Path, synthetic_path: str | Path,
                 output_dir: str | Path, *, min_effective_n: float = 30,
                 diverge_threshold: float = .15) -> dict:
    """Independent human-vs-synthetic audit, not model-based certification.

    Scoring a sample means no need to upload raw respondents into a service.
    Deployments should execute this in a tenant-isolated batch worker.
    """
    if min_effective_n < 1 or not 0 < diverge_threshold <= 1:
        raise ValueError('Invalid scorecard thresholds')
    human = _load_panel(human_path)
    synthetic = _load_panel(synthetic_path)
    if not set(human.question_id) & set(synthetic.question_id):
        raise ValueError('Human and synthetic panels have no shared questions')
    if set(human.respondent_id) & set(synthetic.respondent_id):
        raise ValueError('Respondent IDs overlap across panels; check source and join')
    rows = []
    for question in sorted(set(human.question_id) & set(synthetic.question_id)):
        hq = human[human.question_id == question]
        sq = synthetic[synthetic.question_id == question]
        k = int(max(hq.answer.max(), sq.answer.max()))
        if k < 2 or k > 10:
            continue
        groups = [('all', 'all')]
        for dim in ('age_band', 'region'):
            groups.extend((dim, group) for group in sorted(set(hq[dim]) | set(sq[dim])))
        for dim, group in groups:
            a = hq if dim == 'all' else hq[hq[dim] == group]
            b = sq if dim == 'all' else sq[sq[dim] == group]
            if a.empty or b.empty:
                rows.append({'question_id': question, 'dimension': dim,
                             'subgroup': group, 'status': 'missing_panel_cell',
                             'human_n': len(a), 'synthetic_n': len(b)})
                continue
            ac, ane = weighted_counts(a, k)
            bc, bne = weighted_counts(b, k)
            score = score_counts(ac, bc, ane, bne, min_effective_n=min_effective_n)
            flags = []
            if score.status != 'scored':
                flags.append('underpowered')
            else:
                if score.tvd >= diverge_threshold:
                    flags.append('distribution_divergence')
                if score.winner_disagreement:
                    flags.append('observed_decision_disagreement')
                if score.audit_margin < .05:
                    flags.append('reference_decision_near_tie')
            rows.append({
                'question_id': question, 'dimension': dim, 'subgroup': group,
                'human_n': len(a), 'synthetic_n': len(b), **score.as_dict(),
                'flags': flags,
            })
    if not rows:
        raise ValueError('No overlapping comparable cells')
    overall = {(r['question_id']): r for r in rows
               if r['dimension'] == 'all' and r['status'] == 'scored'}
    for r in rows:
        parent = overall.get(r['question_id'])
        # Aggregate agreement hides subgroup failure, a key review question.
        if (parent and r['dimension'] != 'all' and r['status'] == 'scored'
                and parent['tvd'] < diverge_threshold <= r['tvd']):
            r['flags'].append('aggregate_masks_subgroup_failure')
    scoreable = [r for r in rows if r['status'] == 'scored']
    counts = {
        'cells_total': len(rows), 'cells_scored': len(scoreable),
        'cells_underpowered': sum(r['status'] == 'insufficient_evidence' for r in rows),
        'cells_missing_panel': sum(r['status'] == 'missing_panel_cell' for r in rows),
        'distribution_divergence': sum('distribution_divergence' in r.get('flags', []) for r in rows),
        'observed_decision_disagreements': sum('observed_decision_disagreement' in r.get('flags', []) for r in rows),
        'masked_subgroup_failures': sum('aggregate_masks_subgroup_failure' in r.get('flags', []) for r in rows),
    }
    result = {
        'schema_version': 'scorecard-v1',
        'human_data_sha256': sha256(human_path),
        'synthetic_data_sha256': sha256(synthetic_path),
        'min_effective_n': min_effective_n,
        'diverge_threshold': diverge_threshold,
        'summary': counts, 'cells': rows,
        'limitations': [
            'Observed modal disagreement is not an estimated causal decision-change probability.',
            'TVD threshold is an operator-set review trigger, not a statistical test or certification.',
            'Kish effective sample size does not replace replicate weights or design-based confidence intervals.',
            'No information about the provenance of a purported human/synthetic panel is independently verified.',
        ],
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/'scorecard.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    worst = sorted((r for r in scoreable if r['dimension'] != 'all'),
                   key=lambda r: r['tvd'], reverse=True)[:12]
    md = [
        '# Kernsignal independent validation scorecard',
        f'Compared **{human.respondent_id.nunique():,} human** and **{synthetic.respondent_id.nunique():,} synthetic** respondents (as labelled by source files).',
        f'Scored {counts["cells_scored"]}/{counts["cells_total"]} question × subgroup cells.',
        f'- Distribution divergence flags (TVD ≥ {diverge_threshold:.2f}): {counts["distribution_divergence"]}.',
        f'- Observable disagreements in modal answer: {counts["observed_decision_disagreements"]}.',
        f'- Subgroup failures masked by aggregate score: {counts["masked_subgroup_failures"]}.',
        f'- Underpowered or missing cells: {counts["cells_underpowered"] + counts["cells_missing_panel"]}.',
        '## Highest-divergence subgroup cells',
        '| Question | Dimension | Subgroup | TVD | Mean gap | Audit n_eff | Synthetic n_eff | Flags |',
        '|---|---|---|---:|---:|---:|---:|---|',
    ]
    for row in worst:
        md.append(f'| {row["question_id"]} | {row["dimension"]} | {row["subgroup"]} | '
                  f'{row["tvd"]:.3f} | {row["mean_gap"]:+.3f} | '
                  f'{row["audit_effective_n"]:.0f} | {row["synthetic_effective_n"]:.0f} | '
                  f'{", ".join(row["flags"]) or "none"} |')
    md.extend(['## Evidence limits', *('- '+x for x in result['limitations'])])
    (out/'scorecard.md').write_text('\n'.join(md) + '\n')
    return counts