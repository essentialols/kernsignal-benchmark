# Cross-study transfer: Twin-2K-500 risk model frozen and scored on GSS

Exploratory. The risk model is fitted on Twin-2K-500 only, frozen, and used to
score cells of an independent published human-versus-LLM study without any
refitting or recalibration. The analysis plan (cell eligibility, features,
decision rule) was fixed before any GSS outcome was computed. No model is
approved or registered.

## Bottom line

| Target | Cells (families) | Prevalence of TVD >= 0.15 | Family-balanced Brier gain vs frozen Twin-2K constant (95% family-cluster CI) | ROC AUC (95% CI) |
|---|---:|---:|---|---|
| GSS 2016-24 attitude items | 110 (10) | 0.627 | +0.040 [-0.028, 0.106], **inconclusive** | **0.752 [0.554, 0.893]** |

- The frozen Twin-2K synthetic-only model **ranks** discrepancy risk in an
  independent study. The AUC interval is above 0.5, and reviewing the top 20%
  of cells by predicted risk finds 1.59x [1.22, 2.17] more divergent cells than
  reviewing at random.
- Its **probabilities do not transfer.** The Brier gain over the Twin-2K
  constant is not distinguishable from zero, and calibration is poor (ECE 0.178
  [0.152, 0.302]; predicted mean 0.53 versus observed 0.63).
- Calibrated cross-study risk probabilities are therefore **not demonstrated**.

## Data included and excluded

| | GSS (replication package, commit `76aa037`) |
|---|---|
| Humans | 919 respondents, 993 respondent x item rows, 10 items |
| Models | Claude Haiku 4.5, Claude Sonnet 4.6, Llama 3.1-8B, Llama 3.3-70B |
| Prompt conditions | A (single answer), C (distribution, argmax), A with options reversed and B persona (Claude only) |
| Subgroups | all respondents only |
| Response coding | native answer codes |
| Cells eligible / considered | 110 / 120 |
| Excluded | Llama 3.1-8B style C (10 cells, fewer than 80 valid answers; the source flags it as 57% invalid); seed 1 (repeat draw of the same respondents); the World Values Survey files in the same package (license: non-profit replication only, no redistribution) |
| Audit arm (10%) | not evaluated (10% of about 99 people is below the 25-person minimum) |

Training: Twin-2K-500 revision `f883165a`, all 212 eligible cells from 75
families, constant prevalence 0.6745. Source-file SHA256s for both studies are
written to the output report.

## Transfer, all arms (frozen, no refit)

| Arm | Family-bal. Brier | Twin-2K const. | Oracle const. | Gain [95% CI] | AUC [95% CI] | AP | Top-20% lift | ECE |
|---|---:|---:|---:|---|---|---:|---:|---:|
| GSS synthetic-only | 0.196 | 0.236 | 0.234 | 0.040 [-0.028, 0.106] | 0.752 [0.554, 0.893] | 0.881 | 1.59 | 0.178 |
| GSS S1 shape-only synthetic | 0.191 | 0.236 | 0.234 | 0.045 [-0.014, 0.097] | 0.750 [0.553, 0.890] | 0.880 | 1.59 | 0.142 |

"S1 shape-only" restricts the model to features of the synthetic answer
distribution's shape (entropy, margin, top share, Simpson index, number of
categories). It matches the full model, so the GSS finding does not depend on
features whose meaning differs between the two studies.

## Secondary results

Per-model and per-condition slices of the frozen synthetic-only arm (family-cluster CIs):

| GSS slice | Cells | Prevalence | Brier gain [CI] | AUC [CI] |
|---|---:|---:|---|---|
| Claude Haiku 4.5 | 40 | 0.575 | 0.105 [0.017, 0.199] | 0.887 [0.714, 1.000] |
| Claude Sonnet 4.6 | 40 | 0.650 | -0.009 [-0.157, 0.123] | 0.635 [0.290, 0.929] |
| Llama 3.3-70B | 20 | 0.650 | 0.038 [-0.035, 0.120] | 0.791 [0.573, 0.989] |
| Llama 3.1-8B | 10 | 0.700 | -0.016 [-0.153, 0.122] | 0.595 [0.200, 0.958] |
| Style A | 40 | 0.600 | 0.045 [-0.024, 0.126] | 0.723 [0.546, 0.840] |
| Style C (argmax) | 30 | 0.733 | -0.016 [-0.153, 0.108] | 0.716 [0.433, 0.956] |
| Options reversed | 20 | 0.550 | 0.099 [0.002, 0.199] | 0.838 [0.647, 1.000] |
| Persona (style B) | 20 | 0.600 | 0.055 [-0.031, 0.133] | 0.760 [0.500, 0.961] |

These slices are descriptive and many-way; no claim rests on any single one.

Within-study (model refit inside GSS, family-held-out 5-fold, no respondent
split): 110 holdout cells from 10 families, Brier gain versus the
train-prevalence constant 0.070 [0.040, 0.104], AUC 0.714 [0.525, 0.874]. The
within-study discrimination is similar to the frozen transfer (0.75),
consistent with the Twin-2K features carrying ranking signal into GSS.

Human-human noise floor: paired roughly 99 humans versus other evaluation-fold
humans answering the same item gives a mean TVD of 0.048 and a maximum of 0.137
(the pilot sample is stratified, so this is not a pure sampling floor); no item
reaches 0.15. The observed synthetic-human mean TVD is 0.224, well above it.

## Limitations

- One target with 10 question families: cluster-bootstrap intervals are wide,
  and percentile intervals with so few clusters are approximate.
- GSS cells use about 99 people each, so label noise from sampling is
  non-trivial (noise floor up to 0.137 against a 0.15 flag).
- Feature semantics differ across studies (`synth_missing_fraction` is display
  logic in Twin-2K and invalid-output share here; cell sizes lie outside the
  Twin-2K range). The shape-only results match the full model, so these
  differences do not drive the finding.
- All models, prompts and items come from published academic replication
  packages, not commercial vendor panels, and models may have seen these public
  survey items.

## Reproduce

The GSS replication package is third-party and is not redistributed here. Fetch
it at the pinned commit into the git-ignored `data/` directory (the World
Values Survey files it contains are not read):

```bash
mkdir -p data/gss-when-synthetic-users-fail && cd data/gss-when-synthetic-users-fail
git init
git remote add origin https://github.com/ZihanChen1995/when-synthetic-users-fail-a-cross-domain-benchmark-of-llm-simulated-human-survey-responses
git fetch --depth 1 origin 76aa0376835b60c5dd54851a6fc4e77e32753465
git checkout FETCH_HEAD
cd ../..
```

The Twin-2K-500 files are downloaded by `python -m scripts.twin2k_benchmark`
into `data/twin2k` (see the [README](../README.md)); run that first, then:

```bash
python -m scripts.cross_study_benchmark \
  --twin2k-dir data/twin2k \
  --gss-dir data/gss-when-synthetic-users-fail \
  --output artifacts/cross-study/gss_transfer.json
```

The run prints one summary line (cells, decision,
Brier gain with its interval, AUC with its interval). The report holds
aggregate metadata only.
