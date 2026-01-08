"""Post-processing modules for S-C evidence retrieval.

Includes:
- Score calibration: Map raw reranker scores to calibrated probabilities
- No-evidence detection: Detect queries with no evidence in the post
- Dynamic-k: Adaptively select top-k based on score distribution
"""

from final_sc_review.postprocessing.calibration import ScoreCalibrator
from final_sc_review.postprocessing.dynamic_k import DynamicKSelector
from final_sc_review.postprocessing.no_evidence import NoEvidenceDetector

__all__ = ["ScoreCalibrator", "DynamicKSelector", "NoEvidenceDetector"]
