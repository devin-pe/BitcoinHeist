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
    

@dataclass(init=False, frozen=True)    
class PreprocessConfig:
    load_cols = ["year", "day", "length", "weight", "count", "looped", "neighbors", "income", "label"]
    target_col = "label"
    benign_label = "white"
    date_col = "transaction_date"
    year_col = "year"
    day_col = "day"
    ransomware_col = "is_ransomware"
    drop_cols = ["year", "day", "label"]
