# Architecture

The repository answers one question: where do synthetic survey respondents
diverge from real people, and can that be predicted without asking more people?

```
Twin-2K-500 (Hugging Face, pinned revision)      GSS replication package (pinned commit)
  GPT-4.1-mini answers + human answers              LLM runs + human answers
          |                                                   |
 scripts/inspect_twin2k.py   download + schema check          |
          |                                                   |
 kernsignal/paired_validation.py                    kernsignal/cross_study.py
   cells = question x split, features from the        GSS adapter -> same cell features,
   synthetic answer distribution only                 Twin-2K model frozen, no refit
   respondent-disjoint + question-family-disjoint
   CV, equal-family-weight logistic regression
          |                                                   |
 scripts/twin2k_benchmark.py  -> validation.json    scripts/cross_study_benchmark.py -> gss_transfer.json

 kernsignal/scoring.py + validation.py   offline scorecard (TVD, JS, mean gap, Kish n_eff),
 scripts/make_demo_panels.py             run on simulated panels via kernsignal.cli
```

## Modules

| Module | Role |
|---|---|
| `kernsignal/scoring.py` | Distribution metrics for one question x subgroup cell: weighted TVD, Jensen-Shannon divergence, mean gap, modal-answer disagreement, Kish effective n, insufficient-evidence abstention. |
| `kernsignal/validation.py` | Human-versus-synthetic scorecard over all questions and subgroups. Needs no fitted model. |
| `kernsignal/reference.py` | Panel validation (columns, duplicates, weights) and the simulated demo panel generator. |
| `kernsignal/paired_validation.py` | The Twin-2K-500 benchmark: cell construction, deterministic hash splits, family-grouped cross-validation, baselines, Brier/AUC/lift metrics and family-cluster bootstrap. |
| `kernsignal/cross_study.py` | GSS adapter, frozen-model transfer, calibration (ECE), bootstrap and within-study refits. |
| `kernsignal/cli.py` | `demo-reference` and `score-panels` commands for the offline demo. |

## Design choices

1. **Two leakage defenses.** Participants are hashed into train and test, and
   questions are grouped into families that never straddle a fold. A model
   that has seen a respondent or a sibling question cannot be credited with
   predicting it.
2. **Synthetic-only features.** The headline arm uses only properties of the
   synthetic answer distribution (entropy, margin, top share, Simpson index,
   number of categories, missing share, sample size). Human answers to the
   held-out question never enter it. Audit and historical arms add human
   evidence and are labeled as such.
3. **Fixed target.** A cell is flagged when the TVD between the synthetic
   distribution and the withheld human distribution is at least 0.15. The
   threshold is fixed in advance, not tuned on outcomes.
4. **Aggregates only.** Reports hold counts, metrics and file hashes. Raw
   respondent files stay in the git-ignored `data/` directory.
5. **Deterministic and replayable.** Splits are hash-based and seeds are fixed,
   so the same data and code give the same report.
6. **Frozen transfer.** The cross-study run fits on Twin-2K-500 once and scores
   GSS cells without refitting, so calibration failures cannot be hidden by
   re-estimation.
