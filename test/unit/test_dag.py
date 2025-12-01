import mock
import pytest
import numpy as np
from dagster import build_asset_context
from configs.configs import FileConfig
from src.dags.training_dag import preprocessing, first_pass_feature_eng, second_pass_feature_eng, _run_training_eval

def test_preprocessing():
    expected = "/tmp/mock_preprocessing.parquet"
    with mock.patch("src.dags.training_dag.preprocess") as mock_preprocess, \
          mock.patch.object(FileConfig, "preprocessing_data_path", expected):
        mock_data = mock.Mock()
        mock_preprocess.return_value = mock_data
        
        out = preprocessing()
        mock_data.write.mode.return_value.parquet.assert_called_with("/tmp/mock_preprocessing.parquet")
        assert out == expected


def test_first_pass_feature_eng():
    input_path = "/tmp/input.parquet"
    expected = "/tmp/output_feature_eng_1.parquet"
    with mock.patch("src.dags.training_dag.get_spark_session") as mock_get_spark_session, \
          mock.patch("src.dags.training_dag.first_pass_feature_engineering") as mock_feature_eng, \
          mock.patch.object(FileConfig, "first_features_data_path", expected):
           
        mock_spark = mock.Mock()
        mock_get_spark_session.return_value = mock_spark
        mock_read_data = mock.Mock()
        mock_spark.read.parquet.return_value = mock_read_data
        mock_transformed_data = mock.Mock()
        mock_feature_eng.return_value = mock_transformed_data
        
        out = first_pass_feature_eng(input_path)
        assert out == expected
        mock_spark.read.parquet.assert_called_with(input_path)
        mock_feature_eng.assert_called_with(mock_read_data)
        mock_transformed_data.write.mode.return_value.parquet.assert_called_with(expected)


def test_second_pass_feature_eng():
    input_path = "/tmp/input_feature_eng_1.parquet"
    expected = "/tmp/output_feature_eng_2.parquet"
    with mock.patch("src.dags.training_dag.get_spark_session") as mock_get_spark_session, \
          mock.patch("src.dags.training_dag.second_pass_feature_engineering") as mock_feature_eng, \
          mock.patch.object(FileConfig, "second_features_data_path", expected):
        
        mock_spark = mock.Mock()
        mock_get_spark_session.return_value = mock_spark
        mock_read_data = mock.Mock()
        mock_spark.read.parquet.return_value = mock_read_data
        mock_transformed_data = mock.Mock()
        mock_feature_eng.return_value = mock_transformed_data
        
        out = second_pass_feature_eng(input_path)
        assert out == expected
        mock_spark.read.parquet.assert_called_with(input_path)
        mock_feature_eng.assert_called_with(mock_read_data)
        mock_transformed_data.write.mode.return_value.parquet.assert_called_with(expected)
        

def test_run_training_eval():
    with mock.patch("src.dags.training_dag.get_spark_session") as mock_get_spark_session, \
          mock.patch("src.dags.training_dag.split_data") as mock_split, \
          mock.patch("src.dags.training_dag.train") as mock_train, \
          mock.patch("src.dags.training_dag.explain") as mock_explain, \
          mock.patch("src.dags.training_dag.get_optimal_threshold") as mock_get_threshold, \
          mock.patch("src.dags.training_dag.determine_important_features") as mock_determine_features, \
          mock.patch("src.dags.training_dag.evaluate") as mock_evaluate:

        mock_spark = mock.Mock()
        mock_get_spark_session.return_value = mock_spark
        mock_spark.read.parquet.return_value = "mock_data"

        mock_split.return_value = ("X_train", "y_train", "X_test", "y_test")
        mock_train.return_value = "mock_model"
        mock_explain.return_value = (np.array([0.5, 0.5]), "mock_attributions_data")
        mock_get_threshold.return_value = 0.5
        
        mock_determine_features.return_value = {
            "fn_strongest_5": {"income": 10.0},
            "tp_weakest_5": {"length": 1.0}
        }
        mock_metrics = {
            "accuracy": 0.95, "tp": 10, "fn": 2, 
        }
        mock_evaluate.return_value = mock_metrics

        context = build_asset_context()
        result = _run_training_eval("/tmp/fake_data.parquet", context)
        mock_split.assert_called_with("mock_data")
        mock_train.assert_called_with("X_train", "y_train")

        assert result.value == mock_metrics
        assert result.metadata["Accuracy"].value == 0.95
        assert result.metadata["True Positives"].value == 10
        assert result.metadata["False Negatives"].value == 2