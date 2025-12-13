import logging
import uuid
import time
from datetime import datetime, timezone, timedelta
from functools import cache
from typing import Union
from http import HTTPStatus
from flask import Flask, request, jsonify, Request, Response
from prometheus_flask_exporter import PrometheusMetrics
from pyspark.sql import SparkSession, DataFrame
from configs.configs import RunConfig, ModelConfig, FeatureConfig, ApiConfig
from src.model import RansomwareClassifier
from src.features import FeatureExtractor
from src.utils import get_best_model_run_id, append_live_telemetry
from src.metrics import (
    model_load_time,
    pred_counter,
    data_quality_counter,
    prediction_confidence,
    cache_hit_rate,
    cache_hits,
    cache_misses,
    high_risk_predictions, 
    update_memory_metrics
)


logger = logging.getLogger(__name__)
PREDICTION_CACHE = {}


@cache
def get_model(run_id: str) -> RansomwareClassifier:
    start_time = time.time()
    classifier = RansomwareClassifier()
    classifier.load_from_mlflow(run_id)
    load_time = time.time() - start_time
    model_load_time.set(load_time)
    return classifier
  
@cache
def get_feature_extractor(run_id: str) -> FeatureExtractor:
    extractor = FeatureExtractor()
    extractor.load_from_mlflow(run_id=run_id)
    return extractor


logger.info("Loading cache and starting spark session.")
try:
    run_id = get_best_model_run_id()
    get_model(run_id)
    get_feature_extractor(run_id)
except AttributeError:
    logging.error("Model not trained yet")

SparkSession.builder.master(f"local[{RunConfig.n_jobs}]").getOrCreate()

logger.info("Starting flask app.")
app = Flask(__name__)
metrics = PrometheusMetrics(app, defaults_prefix=f"{RunConfig.app_name}")
metrics.info(
    f"{RunConfig.app_name}_model_version",
    "Model version information",
    experiment_name=RunConfig.experiment_name,
    run_name=RunConfig.run_name,
    model_name=ModelConfig.model_name,
)

@app.route("/health", methods=["GET"])
@metrics.do_not_track()
def health():
    return "OK", HTTPStatus.OK


def quality_control_prediction_request(request: Request) -> tuple[Response, HTTPStatus]:
    required_fields = FeatureConfig.default_feature_cols
    missing_fields = [field for field in required_fields if field not in request.json]
    if missing_fields:
        return jsonify({"error": f"Missing required fields: {missing_fields}"}), HTTPStatus.BAD_REQUEST

    for col in FeatureConfig.default_feature_cols:
        value = request.json[col]
        
        if value < 0:
            logging.warning(f"Received negative {col} in request.")
            data_quality_counter.labels(quality_rule=f"neg_{col}").inc()
            

def process_prediction_request(feature_extractor: FeatureExtractor, model: RansomwareClassifier, request_data: DataFrame) -> dict[str, Union[int, str, float]]:
    logging.info("Calculating features for the request.")
    features = feature_extractor.transform(request_data)
    features_data = features.toPandas()
    
    logging.info("Features are ready. Calculating prediction.")
    probabilities, _ = model.explain(features_data)
    probability = probabilities[0]
    prediction = model.predict(features_data)[0].item()
    
    prediction_confidence.observe(probability)
    
    if prediction == 1:
        if probability >= ApiConfig.very_high_risk.threshold:
            high_risk_predictions.labels(confidence_level=ApiConfig.very_high_risk.label).inc()
        elif probability >= ApiConfig.high_risk.threshold:
            high_risk_predictions.labels(confidence_level=ApiConfig.high_risk.label).inc()
        else:
            high_risk_predictions.labels(confidence_level="medium").inc()
    else:
        high_risk_predictions.labels(confidence_level="low").inc()
            
    pred_label = "ransomware" if prediction == 1 else "white"
    request_id = str(uuid.uuid4())
    PREDICTION_CACHE[request_id] = features_data

    response = {
        "prediction": prediction,
        "label": pred_label,
        "request_id": request_id,
        "confidence": float(probability)
    }
    pred_counter.labels(pred=pred_label).inc()
    return response
  

@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Received prediction request at /predict.")
    update_memory_metrics()
    
    quality_control_prediction_request(request)
    
    run_id = get_best_model_run_id()
    model = get_model(run_id)
    feature_extractor = get_feature_extractor(run_id)
    spark = SparkSession.builder.master(f"local[{RunConfig.n_jobs}]").getOrCreate()

    request_data = spark.createDataFrame(
        [
            {
                "length": request.json["length"],
                "weight": request.json["weight"],
                "count": request.json["count"],
                "looped": request.json["looped"],
                "neighbors": request.json["neighbors"],
                "income": request.json["income"],
            },
        ]
    )

    response = process_prediction_request(feature_extractor, model, request_data)
    
    central_european_timezone = timezone(timedelta(hours=1))
    central_european_time = datetime.now(central_european_timezone)
    telemetry_entry = {
        "timestamp": central_european_time.isoformat(),
        "is_ransomware": bool(response["prediction"] == 1),
        "confidence": response["confidence"]
    }
    append_live_telemetry(telemetry_entry)
    
    logging.info(f"Prediction complete for Request ID {response['request_id']}. Prediction: {response['prediction']} ({response['label']})")
    return jsonify(response)


@app.route("/explain/<request_id>", methods=["GET"])
def explain(request_id):

    update_memory_metrics()
    
    if request_id not in PREDICTION_CACHE:
        cache_misses.inc()
        return jsonify({"error": "Request ID not found"}), HTTPStatus.NOT_FOUND

    cache_hits.inc()
    total_requests = cache_hits._value.get() + cache_misses._value.get()
    if total_requests > 0:
        hit_rate = (cache_hits._value.get() / total_requests) * 100
        cache_hit_rate.set(hit_rate)
    
    features_data = PREDICTION_CACHE[request_id]
    run_id = get_best_model_run_id()
    model = get_model(run_id)
    _, local_attributions = model.explain(features_data)
    attributions_dict = local_attributions.iloc[0].to_dict()
    sorted_features = sorted(attributions_dict.items(), key=lambda item: abs(item[1]), reverse=True)

    top_5_features = {feature: float(value) for feature, value in sorted_features[:5]}

    del PREDICTION_CACHE[request_id] 

    return jsonify(top_5_features)