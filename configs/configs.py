import os
from dataclasses import dataclass
from datetime import date

env_mode = os.getenv("APP_ENV", "LOCAL").upper()

@dataclass(init=False, frozen=True)
class BaseRunConfig:
    app_name: str = "btc_ransomware"
    run_name: str = "Run_123"
    experiment_name: str = "BTC"
    n_jobs: int = 5
    seed: int = 42
    mlflow_tracking_uri: str = "http://localhost:8080"


@dataclass(init=False, frozen=True)
class FileConfig:
    csv_path: str = "data/BitcoinHeistData.csv"
    parquet_dir: str = "data/dataset"
    preprocessing_data_path: str = "data/intermediate/preprocessing/"
    first_features_data_path: str = "data/intermediate/first_features/"
    second_features_data_path: str = "data/intermediate/second_features"
    artifact_folder: str = "feature_extractor"
    artifact_file: str = "features.pkl"
    telemetry_training_data_path: str = "data/telemetry_training_data.json"
    telemetry_live_data_path: str = "data/telemetry_live_data.json"
    chunk_size: int = 100_000
    

class PreprocessConfig:
    start_date: date = date(2011, 1, 1)
    load_cols: list[str] = ["partition_date", "year", "day", 
                            "length", "weight", "count", "looped", 
                            "neighbors", "income", "label"]
    partition_date: str = "partition_date"
    label_col: str = "label"
    benign_label: str = "white"
    date_col: str = "transaction_date"
    year_col: str = "year"
    day_col: str = "day"
    target_col: str = "is_ransomware"
    drop_cols: list[str] = ["partition_date", "year", "day", "label"]
    negative_to_positive_ratio: float = 5.0
    
  
class FeatureConfig:
    default_feature_cols: list[str] = ["length", "weight", "count", "looped", "neighbors", "income"]
    epsilon: int =  1e-6
    cols_to_log: list[str] = ["count", "neighbors", "income"]
    first_interaction_cols: list[tuple[str]] = [
        ("looped", "count"), ("looped", "length"), ("income", "length"), 
        ("income", "count"), ("neighbors", "length"), ("neighbors", "weight"),
        ("weight", "length")                   
        ]
    second_interaction_cols: list[tuple[str]] = [
        ("log_income", "log_neighbors"), ("weight", "log_neighbors"), ("weight", "length"),
        ("length", "log_neighbors")
    ]
    z_score_cols: list[str] = ["length", "log_count", "log_income", "log_neighbors", "neighbors_per_length"]
    
    
@dataclass(
    init=False,
)
class ModelConfig:
    model_name: str = "alpha"
    n_jobs: int = 5
    max_rounds: int = 100
    model_name: str = "ebm_ransomware_classifier"
    threshold_file: str = "model_threshold.json"
    artifact_folder: str = "training_artifacts"
    true_positives_key_name: str = "tp_weakest_5"
    false_negatives_key_name: str = "fn_strongest_5"


@dataclass(frozen=True)
class RiskLevel:
    threshold: float
    label: str


@dataclass(init=False, frozen=True)
class BaseApiConfig:
    api_uri: str = "http://localhost:5001"
    very_high_risk: RiskLevel = RiskLevel(0.9, "very high")
    high_risk: RiskLevel = RiskLevel(0.7, "high")
    

class BaseTelemetryConfig:
    num_instances_for_trigger: int = 100
    epsilon: float = 1e-6
    push_gateway_uri: str = "http://localhost:9091"


if env_mode == "DOCKER":
    from .docker_config import DockerRunConfig as RunConfig
    from .docker_config import DockerApiConfig as ApiConfig
    from .docker_config import DockerTelemetryConfig as TelemetryConfig
    
elif env_mode == "LOCAL":
    from .local_config import LocalRunConfig as RunConfig
    from .local_config import LocalApiConfig as ApiConfig
    from .local_config import LocalTelemetryConfig as TelemetryConfig
     
else:
    raise ValueError(f"Unknown APP_ENV: {env_mode}")
