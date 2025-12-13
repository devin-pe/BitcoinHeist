from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F
from datetime import date
from configs.configs import FileConfig, RunConfig, PreprocessConfig
from src.utils import save_training_telemetry_baseline


def _get_data_after(start_date: date) -> DataFrame:
  spark = SparkSession.builder.master("local[10]").getOrCreate()
  return (
        spark.read
        .parquet(FileConfig.parquet_dir)
        .select(PreprocessConfig.load_cols)
        .filter(F.col(PreprocessConfig.partition_date) >= F.lit(start_date))
    )


def preprocess(start_date: date) -> DataFrame:
  data = _get_data_after(start_date)
  
  data = data.withColumn(PreprocessConfig.date_col, F.to_date(F.format_string("%d%03d", "year", "day"), "yyyyDDD"))
  
  data = data.withColumn(
        PreprocessConfig.target_col,
        F.when(F.col(PreprocessConfig.label_col) == PreprocessConfig.benign_label, F.lit(0))
        .otherwise(F.lit(1))
        .cast(IntegerType())
    )
  
  cols_to_drop = PreprocessConfig.drop_cols + [PreprocessConfig.date_col]
  data = data.drop(*cols_to_drop)
  
  data = data.dropna(subset=[PreprocessConfig.target_col])
  
  return data


def downsample(negative_data: DataFrame, positive_data: DataFrame, ratio: float) -> DataFrame:
    positive_count = positive_data.count()
    negative_count = negative_data.count()
    
    target_count = int(positive_count * ratio)
    fraction = target_count / negative_count
    
    return negative_data.sample(withReplacement=False, fraction=fraction, seed=RunConfig.seed)
  

def split_data(data: DataFrame):
    positive_data = data.filter(F.col(PreprocessConfig.target_col) == 1)
    negative_data = data.filter(F.col(PreprocessConfig.target_col) == 0)

    negative_data = downsample(
        negative_data, 
        positive_data, 
        ratio=PreprocessConfig.negative_to_positive_ratio
    )
    
    train_pos, test_pos = positive_data.randomSplit([0.8, 0.2], seed=RunConfig.seed)
    train_neg, test_neg = negative_data.randomSplit([0.8, 0.2], seed=RunConfig.seed)
    
    train_data = train_pos.union(train_neg)
    test_data = test_pos.union(test_neg)
    
    train_data = train_data.toPandas() # Convert early to avoid len mismatch
    test_data = test_data.toPandas()
    feature_cols = [c for c in train_data.columns if c != PreprocessConfig.target_col]
    
    train_X = train_data[feature_cols]
    train_y = train_data[PreprocessConfig.target_col]
    
    test_X = test_data[feature_cols]
    test_y = test_data[PreprocessConfig.target_col]
    
    save_training_telemetry_baseline(train_y)

    return train_X, train_y, test_X, test_y

  