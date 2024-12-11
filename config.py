from dataclasses import dataclass
from typing import Optional, List

@dataclass
class AnalysisConfig:
    """Configuration parameters for the founder analysis"""
    base_feature: Optional[str]
    feature_value: Optional[str]
    exclude_features: Optional[List[str]]
    persona: Optional[str]
    feature_combination: int = 1
    min_sample: int = 20
    sample_size: int = 8800
    num_results: int = 10
    decreasing_prob: bool = True
    confidence_level: float = 0.95
    real_world_scaling: float = 1.9
    include_negative: bool = False
    cluster_weights: List[float] = None