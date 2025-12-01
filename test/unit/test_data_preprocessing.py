import mock
import pytest
import tempfile
import datetime
import pandas as pd
from datetime import date
from pyspark.testing.utils import assertDataFrameEqual
from pyspark.sql.types import IntegerType
from src.data_preprocessing import _get_data_after, preprocess


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
          expected = expected.withColumn("is_ransomware", 
                                         expected["is_ransomware"].cast(IntegerType())
                                         )
          assertDataFrameEqual(out, expected, ignoreColumnOrder=True)