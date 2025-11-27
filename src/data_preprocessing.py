from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F
from datetime import date
from configs.configs import FileConfig, RunConfig, PreprocessConfig


def _get_data_after(start_date: date) -> DataFrame:
  spark = SparkSession.builder.master("local[10]").getOrCreate()
  
  # Creates unique integer for year and day
  start_date_int = start_date.year * 1000 + int(start_date.strftime('%j'))
  return (
        spark.read.parquet(FileConfig.parquet_dir)
        .select(PreprocessConfig.load_cols)
        .filter((F.col(PreprocessConfig.year_col) * 1000 + F.col(PreprocessConfig.day_col)) >= start_date_int)
    )


def preprocess(start_date: date) -> DataFrame:
  data = _get_data_after(start_date)
  
  data = data.withColumn(PreprocessConfig.date_col, F.to_date(F.format_string("%d%03d", "year", "day"), "yyyyDDD"))
  
  data = data.withColumn(
        PreprocessConfig.ransomware_col,
        F.when(F.col(PreprocessConfig.target_col) == PreprocessConfig.benign_label, F.lit(0))
        .otherwise(F.lit(1))
        .cast(IntegerType())
    )
  
  data = data.drop(*PreprocessConfig.drop_cols)
  data = data.sort(F.col(PreprocessConfig.date_col))
  data = data.sample(fraction=RunConfig.sample_rate, seed=RunConfig.seed)
  
  return data