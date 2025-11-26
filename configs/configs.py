from dataclasses import dataclass

@dataclass(init=False, frozen=True)
class FileConfig:
    csv_path: str = "data/BitcoinHeistData.csv"
    parquet_dir: str = "data/dataset"
    chunk_size: int = 100_000
