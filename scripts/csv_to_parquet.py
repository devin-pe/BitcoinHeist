import os
import pandas as pd
from datetime import date, timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import math
from collections import defaultdict
from configs.configs import FileConfig

def write_parquet():
    spark = SparkSession.builder.master("local[2]").getOrCreate()
    data = spark.read.option("header", "true").option("inferSchema", "true").csv(FileConfig.csv_path)
    data = data.withColumn(
        "partition_date",
        F.date_add(F.make_date(F.col("year"), F.lit(1), F.lit(1)), F.col("day") - 1).cast("string") 
    )
    data.write \
        .mode("overwrite") \
        .option("maxRecordsPerFile", 100_000) \
        .partitionBy("partition_date") \
        .parquet(FileConfig.parquet_dir)
    
    
if __name__ == "__main__":  # pragma: no cover
    write_parquet()