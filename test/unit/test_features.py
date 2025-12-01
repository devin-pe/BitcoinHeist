import mock
import pytest
import pandas as pd
import numpy as np
from pyspark.testing.utils import assertDataFrameEqual
from src.features import first_pass_feature_engineering, second_pass_feature_engineering
from configs.configs import FeatureConfig


def test_first_pass_feature_engineering(spark_fixture):
    epslion = FeatureConfig.epsilon
    cases = [
        {   # Typical case
            "name": "Standard Values",
            "data": spark_fixture.createDataFrame(pd.DataFrame([
                {
                    "count": 10.0, "neighbors": 1.0, "income": 100.0, 
                    "looped": 5.0, "length": 2.0, "weight": 4.0   
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
                    "count": 0.0, "neighbors": 0.0, "income": 0.0,
                    "looped": 0.0, "length": 0.0, "weight": 0.0
                }
            ])),
            "expected": spark_fixture.createDataFrame(pd.DataFrame([
                {
                    "log_count": 0.0, "log_neighbors": 0.0, "log_income": 0.0,
                    "looped_per_count": 0.0, "looped_per_length": 0.0,
                    "income_per_length": 0.0, "income_per_count": 0.0,
                    "neighbors_per_length": 0.0, "neighbors_per_weight": 0.0,
                    "weight_per_length": 0.0,
                    "looped": 0.0, "length": 0.0, "weight": 0.0
                }
            ]))
        }
    ]

    for case in cases:
        with mock.patch("configs.configs.FeatureConfig") as MockConfig:
            MockConfig.epsilon = 1e-6
            MockConfig.cols_to_log = ["count", "neighbors", "income"]
            MockConfig.first_interaction_cols = [
                ("looped", "count"), ("looped", "length"), ("income", "length"), 
                ("income", "count"), ("neighbors", "length"), ("neighbors", "weight"),
                ("weight", "length")
            ]
            out = first_pass_feature_engineering(case["data"])
            assertDataFrameEqual(out, case["expected"], ignoreColumnOrder=True)


def test_second_pass_feature_engineering(spark_fixture):
    epslion = 1e-6
    
    cases = [
        {
            # Typical case
            "data": spark_fixture.createDataFrame(pd.DataFrame([
                {
                    "count": 10.0, "neighbors": 1.0, "income": 100.0, 
                    "looped": 5.0, "length": 2.0, "weight": 4.0   
                }
            ])),
            "expected": spark_fixture.createDataFrame(pd.DataFrame([
                {
                    "looped": 5.0, "weight": 4.0,
                    "looped_per_count": 5.0 / (10.0 + epslion),
                    "looped_per_length": 5.0 / (2.0 + epslion),
                    "income_per_length": 100.0 / (2.0 + epslion),
                    "income_per_count": 100.0 / (10.0 + epslion),
                    "neighbors_per_weight": 1.0 / (4.0 + epslion),
                    "log_income_per_log_neighbors": np.log1p(100.0) / (np.log1p(1.0) + epslion),
                    "weight_per_log_neighbors": 4.0 / (np.log1p(1.0) + epslion),
                    "weight_per_length": 4.0 / (2.0 + epslion), 
                    "length_per_log_neighbors": 2.0 / (np.log1p(1.0) + epslion)
                }
            ]))
        }
    ]

    for case in cases:
        with mock.patch("configs.configs.FeatureConfig") as MockConfig:
            MockConfig.epslionilon = 1e-6
            MockConfig.cols_to_log = ["count", "neighbors", "income"]
            MockConfig.first_interaction_cols = [
                ("looped", "count"), ("looped", "length"), ("income", "length"), 
                ("income", "count"), ("neighbors", "length"), ("neighbors", "weight"),
                ("weight", "length")
            ]
            MockConfig.second_interaction_cols = [
                ("log_income", "log_neighbors"), ("weight", "log_neighbors"), 
                ("weight", "length"), ("length", "log_neighbors")
            ]
            MockConfig.drop_cols = [
                "length", "log_count", "log_income", 
                "log_neighbors", "neighbors_per_length"
            ]
            
            out = second_pass_feature_engineering(case["data"])
            assertDataFrameEqual(out, case["expected"], ignoreColumnOrder=True)
