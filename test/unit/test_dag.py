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
    
    with mock.patch("src.dags.training_dag.SparkSession") as MockSparkSession, \
      mock.patch("src.dags.training_dag.FeatureExtractor") as MockFeatureExtractor, \
      mock.patch.object(FileConfig, "first_features_data_path", expected):

        mock_spark = mock.Mock()
        MockSparkSession.builder.master.return_value.getOrCreate.return_value = mock_spark
        
        mock_df = mock.Mock()
        mock_spark.read.parquet.return_value = mock_df

        mock_transformed = mock.Mock()
        mock_extractor = MockFeatureExtractor.return_value
        mock_extractor.first_pass_feature_engineering.return_value = mock_transformed
        
        out = first_pass_feature_eng(input_path)
        
        assert out == expected
        mock_spark.read.parquet.assert_called_once_with(input_path)
        mock_extractor.first_pass_feature_engineering.assert_called_once_with(mock_df)
        mock_transformed.write.mode.return_value.parquet.assert_called_once_with(expected)


def test_second_pass_feature_eng():
    input_path = "/tmp/input_feature_eng_1.parquet"
    expected_path = "/tmp/output_feature_eng_2.parquet"
    mock_run_id = "test_run_123"
    
    with mock.patch("src.dags.training_dag.SparkSession") as MockSparkSession, \
      mock.patch("src.dags.training_dag.FeatureExtractor") as MockExtractor, \
      mock.patch("src.dags.training_dag.create_mlflow_experiment_if_not_exists") as mock_create_experiment, \
      mock.patch("src.dags.training_dag.mlflow") as mock_mlflow, \
      mock.patch.object(FileConfig, "second_features_data_path", expected_path):
        
        mock_spark = mock.Mock()
        MockSparkSession.builder.master.return_value.getOrCreate.return_value = mock_spark
        
        mock_df = mock.Mock()
        mock_spark.read.parquet.return_value = mock_df
        
        mock_transformed_data = mock.Mock()
        mock_extractor = MockExtractor.return_value
        mock_extractor._stats = mock.Mock()
        mock_extractor.second_pass_feature_engineering.return_value = mock_transformed_data
        
        mock_run = mock.Mock()
        mock_run.info.run_id = mock_run_id
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        out = second_pass_feature_eng(input_path)
        assert out == (expected_path, mock_run_id)

        mock_create_experiment.assert_called_once()
        mock_spark.read.parquet.assert_called_once_with(input_path)

        mock_extractor.fit_stats.assert_called_once_with(mock_df)
        mock_extractor.save_to_mlflow.assert_called_once_with(run_id=mock_run_id)
        
        mock_extractor.second_pass_feature_engineering.assert_called_once_with(mock_df)
        mock_transformed_data.write.mode.return_value.parquet.assert_called_once_with(expected_path)
        

def test_run_training_eval():
    with mock.patch("src.dags.training_dag.SparkSession") as MockSparkSession, \
      mock.patch("src.dags.training_dag.split_data") as mock_split, \
      mock.patch("src.dags.training_dag.RansomwareClassifier") as MockClassifierClass, \
      mock.patch("src.dags.training_dag.mlflow") as mock_mlflow, \
      mock.patch("src.dags.training_dag.create_mlflow_experiment_if_not_exists"), \
      mock.patch("src.dags.training_dag.ModelConfig") as MockModelConfig:

        MockModelConfig.true_positives_key_name = "tp_weakest_5"
        MockModelConfig.false_negatives_key_name = "fn_strongest_5"

        mock_spark = mock.Mock()
        MockSparkSession.builder.master.return_value.getOrCreate.return_value = mock_spark
        
        mock_df = mock.Mock()
        mock_spark.read.parquet.return_value = mock_df

        mock_split.return_value = ("X_train", "y_train", "X_test", "y_test")
        mock_classifier = mock.Mock()
        MockClassifierClass.return_value = mock_classifier
        
        mock_classifier.explain.return_value = (np.array([0.5, 0.5]), "mock_attributions")
        mock_classifier.get_optimal_threshold.return_value = 0.5
        mock_classifier.determine_important_features.return_value = {
            "fn_strongest_5": {"income": 10.0},
            "tp_weakest_5": {"length": 1.0}
        }
        mock_metrics = {
            "accuracy": 0.95, "tp": 19, "fn": 1, 
        }
        mock_classifier.evaluate.return_value = mock_metrics

        context = build_asset_context()
        out = _run_training_eval("/tmp/fake_data.parquet", context, "test_suffix")

        mock_spark.read.parquet.assert_called_once_with("/tmp/fake_data.parquet")
        mock_split.assert_called_once_with(mock_df)
        mock_classifier.train.assert_called_once_with("X_train", "y_train")
        mock_mlflow.log_metrics.assert_called_with(mock_metrics)

        assert out.value == mock_metrics
        assert out.metadata["Accuracy"].value == 0.95
        assert out.metadata["True Positives"].value == 19
        assert out.metadata["False Negatives"].value == 1