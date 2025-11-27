import csv
from pathlib import Path
import mock
import pytest
import pandas as pd
from scripts.csv_to_parquet import compute_year_day_counts, get_partition_map


def test_compute_year_day_counts():
    cases = [
        { # Typical case, multiple chunks
            "chunks": [
                pd.DataFrame([{"year": "2020", "day": "1"}, {"year": "2020", "day": "1"}]),
                pd.DataFrame(
                    [
                        {"year": "2020", "day": "2"},
                        {"year": "2021", "day": "5"},
                        {"year": "2021", "day": "5"},
                        {"year": "2021", "day": "5"},
                    ]
                ),
            ],
            "expected": {"2020&1": 2, "2020&2": 1, "2021&5": 3},
        },
        {
          # Single chunk
            "chunks": [
                pd.DataFrame([{"year": "2019", "day": "10"}, {"year": "2019", "day": "10"}, {"year": "2019", "day": "11"}])
            ],
            "expected": {"2019&10": 2, "2019&11": 1},
        },
    ]
    for case in cases:
        with mock.patch("scripts.csv_to_parquet.pd.read_csv", return_value=case["chunks"]):
            out = compute_year_day_counts("test.csv", chunk_size=2)
        assert out == case["expected"]


def test_get_partition_map():
    cases = [
        { # Chunks split across days
            "group_counts": {"2020&1": 3, "2020&2": 3, "2020&3": 4, "2021&1": 2},
            "chunk_size": 5,
            "expected": {
                "2020&1": "part_2020_day_001",
                "2020&2": "part_2020_day_002",
                "2020&3": "part_2020_day_003",
                "2021&1": "part_2021_day_001",
            },
        },
        { # Multiple days in one chunk
            "group_counts": {"2019&100": 1, "2019&101": 2, "2020&1": 1},
            "chunk_size": 3,
            "expected": {
                "2019&100": "part_2019_day_100",
                "2019&101": "part_2019_day_100",
                "2020&1": "part_2020_day_001",
            },
        },
    ]
    for case in cases:
        out = get_partition_map(case["group_counts"], case["chunk_size"])
        assert out == case["expected"]

