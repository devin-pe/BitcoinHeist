import numpy as np
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from src.model import get_optimal_threshold, evaluate # Adjust import to match your file structure

def test_get_optimal_threshold():
    cases = [
        {
            # Perfect separation
            "y_true": np.array([0, 0, 1, 1]),
            "probabilities": np.array([0.1, 0.2, 0.8, 0.9]),
            "expected_range": (0.2, 0.8)
        },
        {
            # Mixed performance
            "y_true": np.array([0, 0, 1, 1]),
            "probabilities": np.array([0.1, 0.6, 0.4, 0.9]),
            "expected_range": (0.4, 0.9)
        }
    ]

    for case in cases:
        out = get_optimal_threshold(case["y_true"], case["probabilities"])
        assert case["expected_range"][0] <= out <= (case["expected_range"][1])


def test_evaluate():
    cases = [
        {
            # Perfect prediction
            "y_true": np.array([0, 1, 0, 1]),
            "probabilities": np.array([0.1, 0.8, 0.3, 0.9]),
            "threshold": 0.5,
            "expected": {
                "accuracy": 1.0,
                "auc": 1.0,      
                "tn": 2, "fp": 0, "fn": 0, "tp": 2
            }
        },
        {
            # Misprediction
            "y_true": np.array([0, 1, 0, 1]),
            "probabilities": np.array([0.1, 0.8, 0.3, 0.9]),
            "threshold": 0.05,
            "expected": {
                "accuracy": 0.5, 
                "auc": 1.0,
                "tn": 0, "fp": 2, "fn": 0, "tp": 2
            }
        }
    ]

    for case in cases:
        out = evaluate(case["y_true"], case["probabilities"], case["threshold"])
        
        assert out["accuracy"] == case["expected"]["accuracy"]
        assert out["auc"] == case["expected"]["auc"]
        assert out["tn"] == case["expected"]["tn"]
        assert out["fp"] == case["expected"]["fp"]
        assert out["fn"] == case["expected"]["fn"]
        assert out["tp"] == case["expected"]["tp"]
        assert np.array_equal(out["probabilities"], (case["probabilities"] >= case["threshold"]))