# What this evidence does not establish

- **Simulated panels are demonstrations, not evaluations.** The offline demo
  scorecard uses simulated human and simulated synthetic panels with injected
  distortions. It tests the method and the code, not any real synthetic-audience
  product.
- **Evidence classes must not be mixed.** The offline demo is simulated; the
  Twin-2K-500 and GSS results compare real LLM output with real human answers.
  Brier scores from different benchmarks use different labels and weighting and
  are not comparable across them.
- **Twin-2K-500 is a single study.** Its results are respondent- and
  question-family-held-out with real GPT-4.1-mini output, but they cover one
  dataset, one model release and one survey instrument. Top-20% lift is capped
  at 1/prevalence = 1.55, so lift cannot separate the audit and historical arms.
  Whether the GPT twin personas included respondents' earlier answers to the
  same items is not verified here; if they did, the synthetic-only arm is less
  human-independent than its name suggests.
- **Cross-study transfer is limited.** On GSS the frozen Twin-2K model ranks
  discrepancy risk (AUC 0.75, 10 question families, approximate intervals) but
  its probabilities are not calibrated there. The analysis plan was fixed before
  GSS outcomes were computed, but the target data was already available to the
  author.
- **The 0.15 TVD threshold is a review trigger**, not a significance test, a
  certification threshold or a monetary loss. A flagged cell is a prompt for
  human review.
- **Decisions are the most common answer.** Real decisions need a value or loss
  function; expected regret and subgroup trade-offs are not modeled.
- **Survey weighting and design variance are not fully modeled.** The Kish
  effective n in the scorecard is a screening aid, never a confidence interval
  or a substitute for replicate-weight analysis.
- **Interval validity.** Question-by-subgroup and family cluster resampling do
  not capture all respondent overlap, and percentile intervals with 10 clusters
  are approximate. Treat the intervals as indicative.
- **Hashed respondent IDs are not a formal anonymization guarantee.** The
  benchmarks write aggregate outputs only; handle respondent-level data
  separately.
- **No commercial or vendor validation.** All models, prompts and items come
  from published academic datasets. Nothing here is a certification of any
  vendor, and no risk model is approved for deployment.
