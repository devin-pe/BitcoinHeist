import numpy as np
import pandas as pd
import pytest
from src.model import RansomwareClassifier

def test_get_optimal_threshold():
    classifier = RansomwareClassifier()
    cases = [
        {
            # Perfect separation
            "y_true": pd.Series([0, 0, 1, 1]),
            "probabilities": np.array([0.1, 0.2, 0.8, 0.9]),
            "expected_range": (0.2, 0.8) 
        },
        {
            # Mixed performance
            "y_true": pd.Series([1, 0, 1, 1]),
            "probabilities": np.array([0.1, 0.6, 0.4, 0.9]),
            "expected_range": (0.9, 1.0)
        }
    ]

    for case in cases:
        out = classifier.get_optimal_threshold(case["y_true"], case["probabilities"])
        assert classifier.threshold == out
        assert case["expected_range"][0] <= out <= case["expected_range"][1]


def test_evaluate():
    classifier = RansomwareClassifier()

    cases = [
        {
            "y_true": pd.Series([0, 1, 0, 1]),
            "probabilities": np.array([0.1, 0.8, 0.3, 0.9]),
            "threshold": 0.5,
            "expected": {
                "accuracy": 1.0,
                "fn": 0, 
                "tp": 2
            }
        },
        {
            # Misprediction
            "y_true": pd.Series([0, 1, 0, 1]),
            "probabilities": np.array([0.1, 0.8, 0.3, 0.9]),
            "threshold": 0.05, 
            "expected": {
                "accuracy": 0.5, 
                "fn": 0, 
                "tp": 2
            }
        }
    ]

    for case in cases:
        classifier.threshold = case["threshold"]
        out = classifier.evaluate(case["y_true"], case["probabilities"])
        assert out["accuracy"] == case["expected"]["accuracy"]
        assert out["fn"] == case["expected"]["fn"]
        assert out["tp"] == case["expected"]["tp"]


def test_determine_important_features():
    classifier = RansomwareClassifier()

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
            # All attributions positive
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
        classifier.threshold = case["threshold"]
        out = classifier.determine_important_features(case["y_true"], case["probabilities"], 
                                                      case["local_attributions"])
        
        assert out["tp_weakest_5"] == case["expected"]["tp_weakest_5"]
        if case["expected"]["fn_strongest_5"]:
          assert out["fn_strongest_5"] == case["expected"]["fn_strongest_5"]