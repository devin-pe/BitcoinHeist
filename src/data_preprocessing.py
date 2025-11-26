from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, log1p
from pyspark.ml.feature import VectorAssembler, RobustScaler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.types import IntegerType, DoubleType, FloatType

def load_data(parquet_dir):
  spark = SparkSession.builder.master("local[10]").getOrCreate()
  data = spark.read.parquet(parquet_dir)
  return data


def get_numerical_columns(data):
    numeric_types = [IntegerType, DoubleType, FloatType]
    return [f.name for f in data.schema.fields if isinstance(f.dataType, tuple(numeric_types))]


def preprocess(data):
  pass