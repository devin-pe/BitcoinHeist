import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import math
from collections import defaultdict
from configs.configs import FileConfig


def compute_year_day_counts(csv_path: str, chunk_size: int):
    group_counts = defaultdict(int)
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        year_day_key = chunk["year"].astype(str) + "&" + chunk["day"].astype(str)
        year_day_key = chunk["year"].astype(str) + "&" + chunk["day"].astype(str)
        year_day_count = year_day_key.value_counts().to_dict()
        group_counts.update(year_day_count)
    return group_counts


def get_partition_map(group_counts: dict, chunk_size: int):
    sorted_keys = []
    for key, count in group_counts.items():
        year, day = key.split('&')
        sorted_keys.append((int(year), int(day), count, key))
        
    sorted_keys.sort(key=lambda x: (x[0], x[1]))
    partition_map = {}
    current_batch_n_rows = 0
    parquet_file_name = None

    for year, day, count, year_day_key in sorted_keys:
        if (current_batch_n_rows + count > chunk_size):
            parquet_file_name = None
            current_batch_n_rows = 0
        if parquet_file_name is None:
            parquet_file_name = f"part_{year}_day_{day:03d}"
        partition_map[year_day_key] = parquet_file_name
        current_batch_n_rows += count
        
    return partition_map


def write_to_parquet(csv_path: str, parquet_dir: str, chunk_size: int, partition_map: dict):
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk["temp"] = chunk["year"].astype(str) + "&" + chunk["day"].astype(str)
        chunk["batch"] = chunk["temp"].map(partition_map)
        chunk = chunk.drop(columns=["temp"])
        table = pa.Table.from_pandas(chunk)
        
        ds.write_dataset(
            table, 
            parquet_dir, 
            format='parquet',
            partitioning=['batch'],
            existing_data_behavior='overwrite_or_ignore'
        )
        
    print("All Parquet parts written to disk.")
    

def convert_csv_to_parquet(csv_path: str, parquet_dir: str, chunk_size: int):
    os.makedirs(parquet_dir, exist_ok=True)
    group_counts = compute_year_day_counts(csv_path, chunk_size)
    partition_map = get_partition_map(group_counts, chunk_size)
    write_to_parquet(csv_path, parquet_dir, chunk_size, partition_map)
     
     
if __name__ == "__main__":
    csv_path = FileConfig.csv_path
    parquet_dir = FileConfig.parquet_dir
    chunk_size = FileConfig.chunk_size
    convert_csv_to_parquet(csv_path, parquet_dir, chunk_size)