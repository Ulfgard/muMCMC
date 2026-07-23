"""Validation: sample-based estimators and calibration for muMCMC.

``evaluation`` holds single-posterior estimators (BAR evidence, posterior
density, entropy, information gain). ``calibration`` holds simulation-based
calibration across many objects (SBC ranks and their histogram band).
"""
from .evaluation import PosteriorEvaluation
from .calibration import Calibration, sbc_histogram, SBCHistogram

__all__ = ["PosteriorEvaluation", "Calibration", "sbc_histogram", "SBCHistogram"]
