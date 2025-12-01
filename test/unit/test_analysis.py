import pytest
import pandas as pd
import numpy as np
from src.analysis import determine_important_features

def test_determine_important_features():
    cases = [
        {
            # Typical case
            "y_true": pd.Series([1, 1, 0]),
            "probabilities": np.array([0.9, 0.1, 0.1]),
            "threshold": 0.5,
            "local_attributions": pd.DataFrame([
                {"income": 10.0, "count": 2.0}, 
                {"income": -5.0, "count": -20.0},
                {"income": 100.0, "count": 100.0} 
            ]),
            "expected": {
                "tp_weakest_5": {"count": 2.0, "income": 10.0},
                "fn_strongest_5": {"count": -20.0, "income": -5.0}
            }
        },
        {
            # Empty false negatives
            "y_true": pd.Series([1, 1]),
            "probabilities": np.array([0.9, 0.9]),
            "threshold": 0.5,
            "local_attributions": pd.DataFrame([
                {"income": 10.0, "length": 10.0, "count": 0.0},
                {"income": 30.0, "length": 10.0, "count": 0.0},
            ]),
            "expected": {
                "tp_weakest_5": {"count": 0.0, "income": 20.0, "length": 10.0},
                "fn_strongest_5": {} 
            }
        }
    ]

    for case in cases:
        out = determine_important_features(case["y_true"], case["probabilities"], 
                                           case["local_attributions"], 
                                           case["threshold"])
        
        assert out["tp_weakest_5"] == case["expected"]["tp_weakest_5"]
        
        if case["expected"]["fn_strongest_5"]:
          assert out["fn_strongest_5"] == case["expected"]["fn_strongest_5"]