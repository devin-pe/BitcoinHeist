import os
from dataclasses import dataclass

env_mode = os.getenv("APP_ENV", "LOCAL").upper()

if env_mode == "DOCKER":
    from .docker_config import RunConfig as RunConfig
    
elif env_mode == "LOCAL":
    from .local_config import RunConfig as RunConfig

else:
    raise ValueError(f"Unknown APP_ENV: {env_mode}")


@dataclass(init=False, frozen=True)
class FileConfig:
    csv_path: str = "data/BitcoinHeistData.csv"
    parquet_dir: str = "data/dataset"
    chunk_size: int = 100_000
    

class PreprocessConfig:
    load_cols: list = ["partition_date", "year", "day", "length", "weight", "count", "looped", "neighbors", "income", "label"]
    partition_date: str = "partition_date"
    label_col: str = "label"
    benign_label: str = "white"
    date_col: str = "transaction_date"
    year_col: str = "year"
    day_col: str = "day"
    target_col: str = "is_ransomware"
    drop_cols: list = ["partition_date", "year", "day", "label"]
    
  
class FeatureConfig:
    numeric_cols: list = ["length", "weight", "count", "looped", "neighbors", "income"]
