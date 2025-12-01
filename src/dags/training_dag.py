import pandas as pd
import json
from datetime import date
import dagster as dg
from pyspark.sql import SparkSession
from configs.configs import FileConfig, RunConfig, PreprocessConfig
from src.data_preprocessing import preprocess, split_data
from src.features import first_pass_feature_engineering, second_pass_feature_engineering
from src.model import train, explain, get_optimal_threshold
from src.model import evaluate
from src.analysis import determine_important_features


def get_spark_session():
    return SparkSession.builder.master(f"local[{RunConfig.n_jobs}]").getOrCreate()


@dg.asset(group_name="data_pipeline")
def preprocessing() -> str:
    data = preprocess(PreprocessConfig.start_date)
    path = FileConfig.preprocessing_data_path
    data.write.mode("overwrite").parquet(path)
    return path


@dg.asset(group_name="data_pipeline")
def first_pass_feature_eng(preprocessing: str) -> str:
    spark = get_spark_session()
    data = spark.read.parquet(preprocessing)
    data = first_pass_feature_engineering(data)
    path = FileConfig.first_features_data_path
    data.write.mode("overwrite").parquet(path)
    
    return path


@dg.asset(group_name="data_pipeline")
def second_pass_feature_eng(first_pass_feature_eng: str) -> str:
    spark = get_spark_session()
    data = spark.read.parquet(first_pass_feature_eng)
    data = second_pass_feature_engineering(data)
    path = FileConfig.second_features_data_path
    data.write.mode("overwrite").parquet(path)
    
    return path


def _run_training_eval(data_path: str, context: dg.AssetExecutionContext):
    spark = get_spark_session()
    data = spark.read.parquet(data_path)

    train_X, train_y, test_X, test_y = split_data(data)
    model = train(train_X, train_y)
    
    probabilities_train, local_attributions = explain(model, train_X)
    threshold = get_optimal_threshold(train_y, probabilities_train)
    
    feature_info = determine_important_features(
        train_y, 
        probabilities_train, 
        local_attributions, 
        threshold
    )
    
    probabilities_test, _ = explain(model, test_X)
    metrics = evaluate(test_y, probabilities_test, threshold)
    
    context.log.info(f"Results: {metrics}")
    
    return dg.Output(
        value=metrics,
        metadata={
            "Accuracy": metrics["accuracy"],
            "True Positives": int(metrics["tp"]),
            "False Negatives": int(metrics["fn"]),
            
            "Weakest TP Features": dg.MetadataValue.md(
                f"```json\n{json.dumps(feature_info['tp_weakest_5'], indent=2)}\n```"
            ),
            "Strongest FN Features": dg.MetadataValue.md(
                f"```json\n{json.dumps(feature_info['fn_strongest_5'], indent=2)}\n```"
            )
        }
    )


@dg.asset(group_name="model_training")
def results_baseline(context: dg.AssetExecutionContext, preprocessing: str):
    return _run_training_eval(preprocessing, context)


@dg.asset(group_name="model_training")
def results_first_pass_feature_engineering(context: dg.AssetExecutionContext, first_pass_feature_eng: str):
    return _run_training_eval(first_pass_feature_eng, context)


@dg.asset(group_name="model_training")
def results_second_pass_feature_engineering(context: dg.AssetExecutionContext, second_pass_feature_eng: str):
    return _run_training_eval(second_pass_feature_eng, context)

