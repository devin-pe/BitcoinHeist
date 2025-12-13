import os
import psutil
from prometheus_client import Counter, Histogram, Gauge
from configs.configs import RunConfig


# System Monitoring Metrics
model_load_time = Gauge(
    f"{RunConfig.app_name}_model_load_seconds",
    "Time taken to load model from MLflow"
)

memory_usage = Gauge(
    f"{RunConfig.app_name}_memory_usage_bytes",
    "Current memory usage in bytes"
)

def update_memory_metrics():
    process = psutil.Process(os.getpid())
    memory_usage.set(process.memory_info().rss)

# Data Monitoring Metrics
pred_counter = Counter(
    f"{RunConfig.app_name}_predictions_total",
    "Count of predictions by label",
    labelnames=["pred"]
)

data_quality_counter = Counter(
    f"{RunConfig.app_name}_data_quality_total",
    "Count of data quality issues detected",
    labelnames=["quality_rule"]
)


prediction_confidence = Histogram(
    f"{RunConfig.app_name}_prediction_confidence",
    "Distribution of prediction probabilities",
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)


# Business Metrics
cache_hit_rate = Gauge(
    f"{RunConfig.app_name}_cache_hit_rate",
    "Percentage of cache hits for explanations"
)

cache_hits = Counter(
    f"{RunConfig.app_name}_cache_hits_total",
    "Total number of explanation cache hits"
)

cache_misses = Counter(
    f"{RunConfig.app_name}_cache_misses_total",
    "Total number of explanation cache misses"
)


high_risk_predictions = Counter(
    f"{RunConfig.app_name}_high_risk_predictions_total",
    "Count of high confidence ransomware predictions",
    labelnames=["confidence_level"]
)
