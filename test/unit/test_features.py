import mock
import pytest
import pandas as pd
import numpy as np
from pyspark.testing.utils import assertDataFrameEqual
from src.features import FeatureExtractor
from configs.configs import FeatureConfig


def test_first_pass_feature_engineering(spark_fixture):
    epslion = FeatureConfig.epsilon
    cases = [
        {   # Typical case
            "data": spark_fixture.createDataFrame(pd.DataFrame([
                {
                    "count": 10.0, 
                    "neighbors": 1.0, 
                    "income": 100.0, 
                    "looped": 5.0, 
                    "length": 2.0, 
                    "weight": 4.0   
                }
            ])),
            "expected": spark_fixture.createDataFrame(pd.DataFrame([
                {
                    "log_count": np.log1p(10.0),
                    "log_neighbors": np.log1p(1.0),
                    "log_income": np.log1p(100.0),
                    "looped_per_count": 5.0 / (10.0 + epslion),
                    "looped_per_length": 5.0 / (2.0 + epslion),
                    "income_per_length": 100.0 / (2.0 + epslion),
                    "income_per_count": 100.0 / (10.0 + epslion),
                    "neighbors_per_length": 1.0 / (2.0 + epslion),
                    "neighbors_per_weight": 1.0 / (4.0 + epslion),
                    "weight_per_length": 4.0 / (2.0 + epslion),
                    "looped": 5.0, "length": 2.0, "weight": 4.0
                }
            ]))
        },
        {
            # Zero values
            "data": spark_fixture.createDataFrame(pd.DataFrame([
                {
                    "count": 0.0, 
                    "neighbors": 0.0, 
                    "income": 0.0,
                    "looped": 0.0, 
                    "length": 0.0, 
                    "weight": 0.0
                }
            ])),
            "expected": spark_fixture.createDataFrame(pd.DataFrame([
                {
                    "log_count": 0.0, 
                    "log_neighbors": 0.0, 
                    "log_income": 0.0,
                    "looped_per_count": 0.0, 
                    "looped_per_length": 0.0,
                    "income_per_length": 0.0, 
                    "income_per_count": 0.0,
                    "neighbors_per_length": 0.0, 
                    "neighbors_per_weight": 0.0,
                    "weight_per_length": 0.0,
                    "looped": 0.0, 
                    "length": 0.0, 
                    "weight": 0.0
                }
            ]))
        }
    ]

    for case in cases:
        with mock.patch("src.features.FeatureConfig") as MockConfig:
            MockConfig.epsilon = 1e-6
            MockConfig.cols_to_log = ["count", "neighbors", "income"]
            MockConfig.first_interaction_cols = [
                ("looped", "count"), ("looped", "length"), ("income", "length"), 
                ("income", "count"), ("neighbors", "length"), ("neighbors", "weight"),
                ("weight", "length")
            ]
            out = FeatureExtractor().first_pass_feature_engineering(case["data"])
            assertDataFrameEqual(out, case["expected"], ignoreColumnOrder=True)


def test_second_pass_feature_engineering(spark_fixture):
    epsilon = FeatureConfig.epsilon
    
    count = 10.0
    neighbors = 1.0
    income = 100.0
    looped = 5.0
    length = 2.0
    weight = 4.0
    
    log_count = np.log1p(count)
    log_neighbors = np.log1p(neighbors)
    log_income = np.log1p(income)
    
    neighbors_per_length = neighbors/(length + epsilon)
    looped_per_count = looped/(count + epsilon)
    looped_per_length = looped/(length + epsilon)
    income_per_length = income/(length + epsilon)
    income_per_count = income/(count + epsilon)
    neighbors_per_weight = neighbors/(weight + epsilon)
    weight_per_length = weight/(length + epsilon)

    mock_stats = {
        "length": {"mean": length,     "std": 1.0}, 
        "log_income": {"mean": log_income, "std": 1.0},
        "log_count": {"mean": 0.0, "std": 2.0},
        "log_neighbors": {"mean": 0.0, "std": 1.0},
        "neighbors_per_length": {"mean": 1.0, "std": 3.0},
    }

    cases = [
        {
            "data": spark_fixture.createDataFrame(pd.DataFrame([
                {
                    "count": count,
                    "neighbors": neighbors,
                    "income": income,
                    "looped": looped,
                    "length": length,
                    "weight": weight,
                    "log_count": log_count,
                    "log_neighbors": log_neighbors,
                    "log_income": log_income,
                    "neighbors_per_length": neighbors_per_length,
                    "looped_per_count": looped_per_count,
                    "looped_per_length": looped_per_length,
                    "income_per_length": income_per_length,
                    "income_per_count": income_per_count,
                    "neighbors_per_weight": neighbors_per_weight,
                    "weight_per_length": weight_per_length,
                }
            ])),
            
            "expected": spark_fixture.createDataFrame(pd.DataFrame([
                {
                    "count": count,
                    "neighbors": neighbors,
                    "income": income,
                    "looped": looped,
                    "weight": weight,
                    "looped_per_count": looped_per_count,
                    "looped_per_length": looped_per_length,
                    "income_per_length": income_per_length,
                    "income_per_count": income_per_count,
                    "neighbors_per_weight": neighbors_per_weight,
                    "log_income_per_log_neighbors": log_income/(log_neighbors + epsilon),
                    "weight_per_log_neighbors": weight/(log_neighbors + epsilon),
                    "weight_per_length": weight/(length + epsilon),
                    "length_per_log_neighbors": length/(log_neighbors + epsilon),
                    "length": 0.0, 
                    "log_income": 0.0,
                    "log_count": log_count/2.0,
                    "log_neighbors": log_neighbors,
                    "neighbors_per_length": (neighbors_per_length-1.0)/3.0,
                }
            ]))
        }
    ]

    for case in cases:
        extractor = FeatureExtractor()
        extractor._stats = mock_stats 
        out = extractor.second_pass_feature_engineering(case["data"])
        assertDataFrameEqual(out, case["expected"], ignoreColumnOrder=True)


def test_fit_stats(spark_fixture):
    cases = [
        {
            "data": spark_fixture.createDataFrame(pd.DataFrame({
                "log_count": np.array([-2.0, 0.0, 2.0]),
                "neighbors_per_length": np.array([-2.0, 1.0, 4.0]),
                "income": np.array([100.0, 200.0, 300.0]) 
        })),
            "expected": {
                "log_count": {"mean": 0.0, "std": 2.0},
                "neighbors_per_length": {"mean": 1.0, "std": 3.0}
            }
        }
    ]

    with mock.patch("src.features.FeatureConfig") as MockFeatureConfig:
        MockFeatureConfig.z_score_cols = ["log_count", "neighbors_per_length"]
        
        for case in cases:
            extractor = FeatureExtractor()
            extractor.fit_stats(case["data"])

            assert extractor._is_fitted is True
            for col, expected in case["expected"].items():
                stats = extractor._stats[col]
                np.testing.assert_almost_equal(stats["mean"], expected["mean"], decimal=5)
                np.testing.assert_almost_equal(stats["std"], expected["std"], decimal=5)
            
            assert "income" not in extractor._stats # Not in z_score cols


def test_transform_minimal_features(spark_fixture):
    epsilon = FeatureConfig.epsilon

    count = np.array([10.0, 20.0, 30.0])
    income = np.array([100.0, 200.0, 300.0])
    looped = np.array([1.0, 5.0, 1.0])

    log_count = np.log1p(count + epsilon)
    log_count_mean = np.mean(log_count)
    log_count_std = np.std(log_count, ddof=1)
    
    log_count_scaled = (log_count - log_count_mean) / log_count_std
    looped_per_log_count = looped / (log_count + epsilon)

    cases = [
        {   # Test multiple datapoints
            "data": spark_fixture.createDataFrame(pd.DataFrame({
                "count": count,
                "income": income,
                "looped": looped,
            })),
            
            "expected": spark_fixture.createDataFrame(pd.DataFrame({
                "income": income,
                "looped": looped,
                "income_per_count": income / (count + epsilon), 
                "looped_per_log_count": looped_per_log_count, 
                "log_count": log_count_scaled,
            }))
        }
    ]
    with mock.patch("src.features.FeatureConfig") as MockFeatureConfig:
        MockFeatureConfig.epsilon = epsilon
        MockFeatureConfig.default_feature_cols = ["count", "income", "looped"]
        MockFeatureConfig.cols_to_log = ["count"] 
        MockFeatureConfig.first_interaction_cols = [("income", "count")] 
        MockFeatureConfig.second_interaction_cols = [("looped", "log_count")]
        MockFeatureConfig.z_score_cols = ["log_count"]
        
        for case in cases:
            extractor = FeatureExtractor()
            out = extractor.transform(case["data"])
            assertDataFrameEqual(out, case["expected"], ignoreColumnOrder=True)