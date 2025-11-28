from pyspark.sql import DataFrame
from pyspark.sql.functions import col, unix_timestamp
from pyspark.sql.types import LongType


def minimal_feature_engineering(data: DataFrame) -> DataFrame:
    data = data.select(
        "*", unix_timestamp(col("transaction_date")).cast(LongType()).alias("timestamp")
    )
    return data