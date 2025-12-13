import sys
import mock
import os
import pickle
import json
import numpy as np
import pandas as pd


def init_tests() -> tuple[mock.Mock, mock.Mock, mock.Mock]:
    mock_stats_path = "/tmp/fake_stats.pkl"
    with open(mock_stats_path, "wb") as f:
        pickle.dump({"mean": 0, "std": 1}, f)
    
    mock_mlflow = mock.Mock()
    mock_mlflow.__version__ = "1.0.0" 
    mock_mlflow.artifacts.download_artifacts.return_value = mock_stats_path
    mock_mlflow.get_experiment_by_name.return_value = mock.Mock(experiment_id="1")
    mock_mlflow.search_runs.return_value = pd.DataFrame([{"run_id": "mock_run_id"}])
    
    mock_classifier = mock.Mock()
    mock_classifier.predict.return_value = np.array([0])
    mock_classifier.explain.return_value = (np.array([0.1]), None)
    mock_classifier.threshold = 0.5
    mock_classifier.load_from_mlflow = mock.Mock()
    
    mock_extractor = mock.Mock()
    mock_extractor.transform.return_value = mock.Mock()
    mock_extractor.load_from_mlflow = mock.Mock()
    
    return mock_mlflow, mock_classifier, mock_extractor
    

def test_predict_positive():
    
    mock_mlflow, mock_classifier, mock_extractor = init_tests()
    with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow, "pyspark.sql": mock.Mock()}), \
      mock.patch("src.utils.get_best_model_run_id", return_value="mock_run_id"), \
      mock.patch("src.model.RansomwareClassifier", return_value=mock_classifier), \
      mock.patch("src.features.FeatureExtractor", return_value=mock_extractor), \
      mock.patch("src.metrics.memory_usage"), \
      mock.patch("src.metrics.model_load_time"), \
      mock.patch("src.metrics.input_feature_mean"), \
      mock.patch("src.metrics.data_quality_counter"), \
      mock.patch("src.metrics.prediction_confidence"), \
      mock.patch("src.metrics.high_risk_predictions"), \
      mock.patch("src.metrics.pred_counter"):
      
        from src.main_api import app

        with mock.patch("src.main_api.get_model", return_value=mock_classifier), \
              mock.patch("src.main_api.get_feature_extractor", return_value=mock_extractor), \
              mock.patch("src.main_api.SparkSession.builder.master") as mock_spark_session, \
              mock.patch("src.main_api.get_best_model_run_id", return_value="mock_run_id"):
                
            mock_spark = mock.Mock()
            mock_spark.createDataFrame.return_value = "mock_spark_data"
            mock_spark_session.return_value.getOrCreate.return_value = mock_spark
            
            payload = {
                "length": 5, "weight": 1.0, "count": 10,
                "looped": 1, "neighbors": 1, "income": 100.0
            }

            with app.test_client() as client:
                response = client.post("/predict", data=json.dumps(payload), content_type='application/json')
                assert response.status_code == 200
                assert response.json["prediction"] == 0
                mock_extractor.transform.assert_called_with("mock_spark_data")


def test_api_predict_negative():
    mock_mlflow, mock_classifier, mock_extractor = init_tests()
    with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow, "pyspark.sql": mock.Mock()}), \
      mock.patch("src.utils.get_best_model_run_id", return_value="mock_run_id"), \
      mock.patch("src.model.RansomwareClassifier", return_value=mock_classifier), \
      mock.patch("src.features.FeatureExtractor", return_value=mock_extractor), \
      mock.patch("src.metrics.memory_usage"), \
      mock.patch("src.metrics.model_load_time"), \
      mock.patch("src.metrics.input_feature_mean"), \
      mock.patch("src.metrics.data_quality_counter"), \
      mock.patch("src.metrics.prediction_confidence"), \
      mock.patch("src.metrics.high_risk_predictions"), \
      mock.patch("src.metrics.pred_counter"):
      
        from src.main_api import app

        with mock.patch("src.main_api.get_model", return_value=mock_classifier), \
              mock.patch("src.main_api.get_feature_extractor", return_value=mock_extractor), \
              mock.patch("src.main_api.SparkSession.builder.master") as mock_spark_session, \
              mock.patch("src.main_api.get_best_model_run_id", return_value="mock_run_id"):
            
            mock_spark = mock.Mock()
            mock_spark.createDataFrame.return_value = "mock_spark_data"
            mock_spark_session.return_value.getOrCreate.return_value = mock_spark
            
            payload = {
                "length": 5, "weight": 1.0, "count": 10,
                "looped": 1, "neighbors": 1, "income": -100.0
            }

            with app.test_client() as client:
                response = client.post("/predict", data=json.dumps(payload), content_type='application/json')
                assert response.status_code == 200
                assert response.json["prediction"] == 0
                mock_extractor.transform.assert_called_with("mock_spark_data")


def test_explain_success():
    mock_attributions = pd.DataFrame([{
        "income": 10.0,
        "weight": -9.0,
        "count": 5.0,
        "neighbors": 4.0,
        "income & count": 3.0,
        "length": 0.1  
    }])
    
    mock_mlflow, mock_classifier, mock_extractor = init_tests()
    mock_classifier.explain.return_value = (np.array([0.9]), mock_attributions)
    
    mock_cache_hits = mock.Mock()
    mock_cache_hits._value = mock.Mock()
    mock_cache_hits._value.get.return_value = 10
    
    mock_cache_misses = mock.Mock()
    mock_cache_misses._value = mock.Mock()
    mock_cache_misses._value.get.return_value = 10
    
    with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow, "pyspark.sql": mock.Mock()}), \
      mock.patch("src.utils.get_best_model_run_id", return_value="mock_run_id"), \
      mock.patch("src.model.RansomwareClassifier", return_value=mock_classifier), \
      mock.patch("src.features.FeatureExtractor", return_value=mock_extractor), \
      mock.patch("src.metrics.memory_usage"), \
      mock.patch("src.metrics.model_load_time"), \
      mock.patch("src.metrics.explanation_requests"), \
      mock.patch("src.metrics.cache_hits", mock_cache_hits), \
      mock.patch("src.metrics.cache_misses", mock_cache_misses), \
      mock.patch("src.metrics.cache_hit_rate"):
        
        from src.main_api import app, PREDICTION_CACHE

        with mock.patch("src.main_api.get_model", return_value=mock_classifier), \
              mock.patch("src.main_api.get_best_model_run_id", return_value="mock_run_id"):
            
            request_id = "test-uuid"
            mock_features_data = pd.DataFrame([{"test": "data"}])
            
            PREDICTION_CACHE[request_id] = mock_features_data

            with app.test_client() as client:
                response = client.get(f"/explain/{request_id}")

                assert response.status_code == 200
                top_5_features = response.json
                
                assert len(top_5_features) == 5
                assert "length" not in top_5_features 
                assert top_5_features["weight"] == -9.0

                assert request_id not in PREDICTION_CACHE


def test_explain_id_not_found():
    mock_mlflow, mock_classifier, mock_extractor = init_tests()
    
    with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow, "pyspark.sql": mock.Mock()}), \
      mock.patch("src.utils.get_best_model_run_id", return_value="mock_run_id"), \
      mock.patch("src.model.RansomwareClassifier", return_value=mock_classifier), \
      mock.patch("src.features.FeatureExtractor", return_value=mock_extractor), \
      mock.patch("src.metrics.memory_usage"), \
      mock.patch("src.metrics.model_load_time"), \
      mock.patch("src.metrics.explanation_requests"), \
      mock.patch("src.metrics.cache_misses"):
        
        from src.main_api import app

        with app.test_client() as client:
            response = client.get("/explain/missing_id")
            
            assert response.status_code == 404
            assert "not found" in response.json["error"]