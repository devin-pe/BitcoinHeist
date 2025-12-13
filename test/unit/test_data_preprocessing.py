import mock
import pytest
import tempfile
import datetime
import pandas as pd
from datetime import date
from pyspark.testing.utils import assertDataFrameEqual
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType
from src.data_preprocessing import _get_data_after, preprocess, downsample, split_data


def test_get_data_after(spark_fixture):
    cases = [
        {   # First day of year and filter data 
            "start_date": datetime.date(2021, 1, 1),
            "data": spark_fixture.createDataFrame(
              pd.DataFrame([
                {
                  "partition_date": "2020-12-30",
                  "address": "111K8kZAEnJg245r2cM6y9zgJGHZtJPy6",
                  "year": 2020, 
                  "day": 365, 
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "label": "white",
                  "income": 0.0
                },
                {
                  "partition_date": "2021-01-01",
                  "address": "111K8kZAEnJg245r2cM6y9zgJGHZtJPy6",
                  "year": 2021, 
                  "day": 1, 
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "label": "white",
                  "income": 0.0
                },
                {
                  "partition_date": "2021-01-01",
                  "address": "111K8kZAEnJg245r2cM6y9zgJGHZtJPy6",
                  "year": 2021, 
                  "day": 2, 
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "label": "white",
                  "income": 0.0
                },
              ])),
            "expected": spark_fixture.createDataFrame(
              pd.DataFrame([
                {
                  "partition_date": "2021-01-01",
                  "year": 2021, 
                  "day": 1, 
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "label": "white",
                  "income": 0.0
                },
                {
                  "partition_date": "2021-01-01",
                  "year": 2021, 
                  "day": 2, 
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "label": "white",
                  "income": 0.0
                },
            ])),
        },
        {
            # Start date outside of 1-31
            "start_date": datetime.date(2021, 2, 19), 
            "data": spark_fixture.createDataFrame(
              pd.DataFrame([
                {
                  "partition_date": "2021-02-18",
                  "year": 2021, 
                  "day": 49, 
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "label": "white",
                  "income": 0.0
                },
                {
                  "partition_date": "2021-02-19",
                  "year": 2021, 
                  "day": 50, 
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "label": "white",
                  "income": 0.0,
                },
            ])),
            "expected": spark_fixture.createDataFrame(
              pd.DataFrame([
                {
                  "partition_date": "2021-02-19",
                  "year": 2021, 
                  "day": 50,
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "label": "white",
                  "income": 0.0
                },
            ]))
        }
    ]

    for case in cases:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_path = f"{tmp_dir}/raw_data"
            case["data"].write.parquet(mock_path)
            with mock.patch("src.data_preprocessing.FileConfig") as MockFileConfig: 
                MockFileConfig.parquet_dir = mock_path
                out = _get_data_after(case["start_date"])
                assertDataFrameEqual(out, case["expected"], ignoreColumnOrder=True)


def test_preprocess(spark_fixture):
    cases = [
        {
            # Typical case, mix of benign and ransomware
            "start_date": date(2020, 1, 1),
            "data": spark_fixture.createDataFrame(
              pd.DataFrame([
                {
                  "partition_date": "2020-01-01",
                  "year": 2020, 
                  "day": 1, 
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "label": "white", 
                  "income": 0.0
                 },
                {
                  "partition_date": "2020-01-02",
                  "year": 2020, 
                  "day": 2, 
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "label": "wannacry",  
                  "income": 0.0
                 },
            ])),
            "expected": spark_fixture.createDataFrame(
              pd.DataFrame([
                {
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "income": 0.0, 
                  "is_ransomware": 0
                },
                {
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "income": 0.0,
                  "is_ransomware": 1
                }
            ])
          )
        },
        {
            # Test year with 366 days
            "start_date": date(2020, 2, 1), 
            "data": spark_fixture.createDataFrame(
              pd.DataFrame([
                {
                  "partition_date": "2020-02-01",
                  "year": 2020, 
                  "day": 32, 
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "label": "cryptolocker",
                  "income": 0.0
                },
                {
                  "partition_date": "2020-12-31",
                  "year": 2020, 
                  "day": 366,  
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "label": "white", 
                  "income": 0.0
                },
            ])
          ),
            "expected": spark_fixture.createDataFrame(
              pd.DataFrame([
                {
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "income": 0.0, 
                  "is_ransomware": 1
                },
                {
                  "length": 10,
                  "weight": 1,
                  "count": 246,
                  "looped": 1,
                  "neighbors": 2,
                  "income": 0.0,
                  "is_ransomware": 0
                }
            ])
          )
        }
    ]

    for case in cases:
        with mock.patch("src.data_preprocessing._get_data_after") as mock_get_data:
          mock_get_data.return_value = case["data"]
          out = preprocess(case["start_date"])
          expected = case["expected"]
          expected = expected.withColumn("is_ransomware", expected["is_ransomware"].cast(IntegerType()))
          assertDataFrameEqual(out, expected, ignoreColumnOrder=True)
        

def test_downsample(spark_fixture):
    cases = [
        {
            # Typical case
            "positive_data": spark_fixture.createDataFrame(pd.DataFrame([
                {"address": "112eFykaD53KEkKeYW9KW8eWebZYSbt2f5", "income": 100.0, "is_ransomware": 1},
                {"address": "112eFykaD53KEkKeYW9KW8eWebZYSbt2f5", "income": 200.0, "is_ransomware": 1}
            ])),
            "negative_data": spark_fixture.createDataFrame(pd.DataFrame([
                {"address": "112eFykaD53KEkKeYW9KW8eWebZYSbt2f5", "income": 10.0, "is_ransomware": 0},
                {"address": "112eFykaD53KEkKeYW9KW8eWebZYSbt2f5", "income": 20.0, "is_ransomware": 0},
                {"address": "112eFykaD53KEkKeYW9KW8eWebZYSbt2f5", "income": 5.0, "is_ransomware": 0},
                {"address": "112eFykaD53KEkKeYW9KW8eWebZYSbt2f5", "income": 1.0, "is_ransomware": 0},
                {"address": "112eFykaD53KEkKeYW9KW8eWebZYSbt2f5", "income": 50.0, "is_ransomware": 0}
            ])),
            "ratio": 1.0,
            "expected_count": 2
        },
    ]

    for case in cases:
        with mock.patch("src.data_preprocessing.RunConfig") as MockRunConfig:
            MockRunConfig.seed = 38
            out = downsample(case["negative_data"], case["positive_data"], case["ratio"])
            assert out.count() == case["expected_count"]
            


def test_split_data(spark_fixture):
    cases = [
        {
            # Typical case
            "data": spark_fixture.createDataFrame(pd.DataFrame([
                {"income": 100.0, "weight": 5.0, "count": 10, "is_ransomware": 1},
                {"income": 200.0, "weight": 6.0, "count": 12, "is_ransomware": 1},
                {"income": 0.5, "weight": 1.0, "count": 1, "is_ransomware": 0},
                {"income": 0.2, "weight": 1.0, "count": 1, "is_ransomware": 0},
                {"income": 1.5, "weight": 2.0, "count": 2, "is_ransomware": 0},
                {"income": 5.0, "weight": 1.0, "count": 5, "is_ransomware": 0},
            ])),
            
            "ratio": 1.0,
            "expected_total_rows": 4, # 2 pos, 2 neg
        }
    ]

    for case in cases:
        with mock.patch("src.data_preprocessing.PreprocessConfig") as MockPreprocessConfig, \
              mock.patch("src.data_preprocessing.RunConfig") as MockRunConfig, \
              mock.patch("src.data_preprocessing.downsample") as mock_downsample, \
              mock.patch("src.data_preprocessing.save_training_telemetry_baseline") as mock_telemetry:
            
            MockPreprocessConfig.target_col = "is_ransomware"
            MockPreprocessConfig.negative_to_positive_ratio = case["ratio"]
            MockRunConfig.seed = 42
            mock_downsampled_data = case["data"].filter("is_ransomware == 0").limit(2)
            mock_downsample.return_value = mock_downsampled_data
            
            train_X, train_y, test_X, _ = split_data(case["data"])

            total_rows_out = len(train_X) + len(test_X)
            assert total_rows_out == case["expected_total_rows"]
            mock_telemetry.assert_called_with(train_y)
    
            