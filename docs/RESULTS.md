# Twin-2K-500: can the risk model predict where GPT-4.1-mini diverges from people?

Exploratory single-study result. The data are the published
[Twin-2K-500 dataset](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500),
pinned to revision `f883165a3026fde855dfd448e0cd16443ab257b6` (CC BY 4.0). The
comparison is between existing GPT-4.1-mini simulation outputs and the matching
human survey responses. No errors are planted and no model answers are
generated here. Respondent-level records are never committed; the benchmark
reports aggregates and file hashes only.

Reproduce: `python -m scripts.twin2k_benchmark --revision f883165a3026fde855dfd448e0cd16443ab257b6`
(see the [README](../README.md)).

## Out-of-sample measurements

The experiment holds out both respondents and question families: the risk model
is fitted on other questions and other people. On **104 withheld cells from 71
question families**, the GPT-human mean total-variation distance (TVD) was
**0.2674** and the modal-answer disagreement rate was **26.9%**. The
historical Wave 1-3 versus Wave 4 human TVD was **0.032**, a study-specific
context and not a general natural-noise estimate.

| Risk-prioritization arm | Family-balanced Brier (lower is better) | ROC AUC (higher is better) | Lift at highest-risk 20% | Requires human answers? |
|---|---:|---:|---:|---|
| Prevalence-only (family-balanced prior) | 0.2596 | 0.422 | 0.81x | During training only |
| Uninformative 0.5 forecast (reference) | 0.2500 | 0.500 | 1.00x | No |
| **Synthetic-only** | **0.1897** | **0.802** | **1.33x** | During training only, not for the held-out question |
| 10% contemporary human audit | 0.1655 | 0.883 | 1.55x | **Yes**, an audit for the held-out question |
| Historical human answers | 0.1409 | 0.922 | 1.55x | **Yes**, prior-wave answers for the same question |

The prevalence prior is fitted on the other question families in each fold, so
it can run opposite to the held-out families and score slightly worse than a
0.5 forecast. Gains are therefore reported against both (synthetic-only: +0.070
versus the prior, +0.060 versus 0.5). Top-20% lift is capped at
1/prevalence = 1.55, so lift cannot separate the audit and historical arms.

The target is TVD >= 0.15. That flag is a review trigger, not an observed
purchasing outcome, a monetary loss or a demonstrated business-decision
reversal. Family-balanced Brier gives each question family equal weight; the
cell-cluster-bootstrap intervals in the JSON report are not intervals for that
same family-balanced summary. The incremental audit-versus-synthetic analyses
are **exploratory**.

## What can and cannot be concluded

1. **Within one dataset:** synthetic-only features predict which GPT-4.1-mini
   survey items diverge substantially from held-out human answers. The model
   sees no human answer to the held-out question or from the held-out
   participants.
2. **Value of human reference data:** a 10% contemporaneous audit and
   historical human answers both improve performance descriptively. Showing
   an incremental gain over synthetic-only requires a paired, family-level
   comparison, not a comparison of overlapping bootstrap intervals against the
   constant model.
3. **Not demonstrated:** generalization across studies (the one transfer check
   is in [TRANSFER_GSS.md](TRANSFER_GSS.md)), commercial synthetic-audience
   vendors, other LLMs, countries or platforms, demographic representativeness,
   reliable per-study calibration, real decision reversals, or a prospective
   blinded replication.
4. **External validity:** one published study and one GPT-4.1-mini simulation
   release. Foundation models may have encountered publicly available
   questions. Some categorical variable semantics need verification against the
   official codebook, and filtering to comparable questions affects coverage.
5. **Not a certification.** The result is not an audited third-party vendor
   evaluation and no model is approved for use in any decision.
