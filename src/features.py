from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from configs.configs import FeatureConfig


def first_pass_feature_engineering(data: DataFrame) -> DataFrame:
    for col in FeatureConfig.cols_to_log:
        data = data.withColumn(f"log_{col}", F.log1p(F.col(col)))  

    for (numerator_col, denominator_col) in FeatureConfig.first_interaction_cols:
        data = data.withColumn(f"{numerator_col}_per_{denominator_col}", 
                               F.col(numerator_col) / (F.col(denominator_col) 
                                + FeatureConfig.epsilon))
    data = data.drop(*FeatureConfig.cols_to_log)
    
    return data


def second_pass_feature_engineering(data: DataFrame) -> DataFrame:
    for (numerator_col, denominator_col) in FeatureConfig.second_interaction_cols:
        data = data.withColumn(f"{numerator_col}_per_{denominator_col}", 
                               F.col(numerator_col) / (F.col(denominator_col) 
                                + FeatureConfig.epsilon))
    data = data.drop(*FeatureConfig.drop_cols)
    
    return data
