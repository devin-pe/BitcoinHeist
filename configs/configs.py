import os
from dataclasses import dataclass
from datetime import date

env_mode = os.getenv("APP_ENV", "LOCAL").upper()

@dataclass(init=False, frozen=True)
class BaseRunConfig:
    negative_to_positive_ratio: float = 5.0
    num_folds: int = 5
    experiment_name: str = "BTC"
    seed: int = 42
    n_jobs: int = 5
    max_rounds: int = 100
    mlflow_tracking_uri: str = "http://localhost:8080"


@dataclass(init=False, frozen=True)
class FileConfig:
    csv_path: str = "data/BitcoinHeistData.csv"
    parquet_dir: str = "data/dataset"
    preprocessing_data_path: str = "data/intermediate/preprocessing/"
    first_features_data_path: str = "data/intermediate/first_features/"
    second_features_data_path: str = "data/intermediate/second_features"
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
    
  
class FeatureConfig:
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
    drop_cols: list[str] = ["length", "log_count", "log_income", "log_neighbors", "neighbors_per_length"]
    
    
if env_mode == "DOCKER":
    from .docker_config import DockerRunConfig as RunConfig
    
elif env_mode == "LOCAL":
    from .local_config import LocalRunConfig as RunConfig

else:
    raise ValueError(f"Unknown APP_ENV: {env_mode}")