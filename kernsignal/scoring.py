"""Privacy-preserving, deterministic score of aggregated survey distributions.

Do not put raw respondent data in the API. Risk predictions are conditional on a
specific held-out benchmark; the descriptive diagnostics remain useful without it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

FEATURES = (
    'tvd', 'js_divergence', 'absolute_mean_gap', 'audit_margin',
    'synthetic_margin', 'entropy_gap', 'log_audit_neff', 'log_synthetic_neff',
    'observed_winner_disagreement',
)
MIN_EFFECTIVE_N = 30.0


class InvalidCell(ValueError):
    """Invalid answer categories, sample counts, or effective sample sizes."""


def _validated_counts(values: Sequence[float], name: str) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not 2 <= len(a) <= 10:
        raise InvalidCell(f'{name}: expected 2-10 answer-category counts')
    if not np.isfinite(a).all() or (a < 0).any() or a.sum() <= 0:
        raise InvalidCell(f'{name}: counts must be finite, nonnegative, and not all zero')
    return a


def _entropy(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-np.sum(nz * np.log2(nz)))


def _margin(p: np.ndarray) -> float:
    return float(np.sort(p)[-1] - np.sort(p)[-2])


@dataclass(frozen=True)
class CellScore:
    tvd: float
    js_divergence: float
    mean_gap: float
    audit_margin: float
    synthetic_margin: float
    entropy_gap: float
    audit_neff: float
    synthetic_neff: float
    audit_choice: int
    synthetic_choice: int
    winner_disagreement: bool
    n_categories: int
    status: str

    def features(self) -> np.ndarray:
        return np.asarray([
            self.tvd, self.js_divergence, abs(self.mean_gap),
            self.audit_margin, self.synthetic_margin, self.entropy_gap,
            math.log1p(self.audit_neff), math.log1p(self.synthetic_neff),
            float(self.winner_disagreement),
        ], dtype=float)

    def as_dict(self) -> dict:
        return {
            'tvd': round(self.tvd, 6), 'js_divergence': round(self.js_divergence, 6),
            'mean_gap': round(self.mean_gap, 6),
            'audit_margin': round(self.audit_margin, 6),
            'synthetic_margin': round(self.synthetic_margin, 6),
            'entropy_gap': round(self.entropy_gap, 6),
            'audit_effective_n': round(self.audit_neff, 2),
            'synthetic_effective_n': round(self.synthetic_neff, 2),
            'audit_choice': self.audit_choice,
            'synthetic_choice': self.synthetic_choice,
            'observed_winner_disagreement': self.winner_disagreement,
            'status': self.status,
        }


def score_counts(
    audit_counts: Sequence[float],
    synthetic_counts: Sequence[float],
    audit_neff: float | None = None,
    synthetic_neff: float | None = None,
    min_effective_n: float = MIN_EFFECTIVE_N,
) -> CellScore:
    """Returns TVD, JS divergence, mean gap, decisions, and effective-n gating.

    `counts` may be weighted category sums. Supply Kish effective n separately
    for survey-weighted input; sum(counts) is only valid for unit-weight input.
    Decisions use the modal answer with deterministic lowest-index tie-breaking.
    """
    a = _validated_counts(audit_counts, 'audit_counts')
    b = _validated_counts(synthetic_counts, 'synthetic_counts')
    if len(a) != len(b):
        raise InvalidCell('Both panels must use the same ordered answer categories')
    na = float(a.sum()) if audit_neff is None else float(audit_neff)
    nb = float(b.sum()) if synthetic_neff is None else float(synthetic_neff)
    if not math.isfinite(na) or not math.isfinite(nb) or na <= 0 or nb <= 0:
        raise InvalidCell('Effective sample sizes must be positive and finite')
    if min_effective_n < 1:
        raise InvalidCell('min_effective_n must be >=1')
    pa, pb = a / a.sum(), b / b.sum()
    mix = 0.5 * (pa + pb)
    # np.where evaluates both arms eagerly. Mask first to avoid 0*log(0)
    # warnings (and NaNs in alternative implementations).
    js = 0.5 * float(np.sum(pa[pa > 0] * np.log2(pa[pa > 0] / mix[pa > 0])))
    js += 0.5 * float(np.sum(pb[pb > 0] * np.log2(pb[pb > 0] / mix[pb > 0])))
    positions = np.linspace(0.0, 1.0, num=len(pa))
    return CellScore(
        tvd=float(0.5 * np.abs(pa - pb).sum()),
        js_divergence=float(js),
        mean_gap=float(np.dot(pb - pa, positions)),
        audit_margin=_margin(pa),
        synthetic_margin=_margin(pb),
        entropy_gap=abs(_entropy(pa) - _entropy(pb)) / math.log2(len(pa)),
        audit_neff=na, synthetic_neff=nb,
        audit_choice=int(np.argmax(pa) + 1),
        synthetic_choice=int(np.argmax(pb) + 1),
        winner_disagreement=bool(np.argmax(pa) != np.argmax(pb)),
        n_categories=len(pa),
        status='scored' if min(na, nb) >= min_effective_n else 'insufficient_evidence',
    )


def weighted_counts(frame, k: int) -> tuple[np.ndarray, float]:
    """Weighted frequency counts and Kish n_eff; input has answer, weight."""
    answers = frame['answer'].to_numpy(dtype=int)
    weights = frame['weight'].to_numpy(dtype=float)
    if not len(answers) or np.any((answers < 1) | (answers > k)):
        raise InvalidCell('Unexpected answer code or empty cell')
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise InvalidCell('Weights must be finite and positive')
    counts = np.bincount(answers, weights=weights, minlength=k + 1)[1:]
    neff = float(weights.sum() ** 2 / np.dot(weights, weights))
    return counts, neff