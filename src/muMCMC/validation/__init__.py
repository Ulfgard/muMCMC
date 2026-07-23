"""Validation: sample-based estimators and calibration for muMCMC.

``evaluation`` holds single-posterior estimators (BAR evidence, posterior
density, entropy, information gain). ``coverage`` holds cross-object calibration
(the PIT of a statistic and its coverage over many objects).
"""
from .evaluation import PosteriorEvaluation
from .coverage import pit, coverage_ci, Coverage, sbc_rank, sbc_histogram, SBCHistogram

__all__ = ["PosteriorEvaluation", "pit", "coverage_ci", "Coverage",
           "sbc_rank", "sbc_histogram", "SBCHistogram"]
