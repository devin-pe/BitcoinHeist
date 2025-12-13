import mlflow
import pandas as pd
import json
from datetime import date
import dagster as dg
from pyspark.sql import SparkSession
from configs.configs import FileConfig, RunConfig, PreprocessConfig, ModelConfig
from src.data_preprocessing import preprocess, split_data
from src.features import FeatureExtractor
from src.model import RansomwareClassifier
from src.utils import create_mlflow_experiment_if_not_exists


@dg.asset(group_name="data_pipeline")
def preprocessing() -> str:
    data = preprocess(PreprocessConfig.start_date)
    path = FileConfig.preprocessing_data_path
    data.write.mode("overwrite").parquet(path)
    
    return path


@dg.asset(group_name="data_pipeline")
def first_pass_feature_eng(preprocessing: str) -> str:
    spark = SparkSession.builder.master(f"local[{RunConfig.n_jobs}]").getOrCreate()
    data = spark.read.parquet(preprocessing)
    extractor = FeatureExtractor()
    data = extractor.first_pass_feature_engineering(data)
    path = FileConfig.first_features_data_path
    data.write.mode("overwrite").parquet(path)
    
    return path


@dg.asset(group_name="data_pipeline")
def second_pass_feature_eng(first_pass_feature_eng: str) -> tuple[str, str]:
    create_mlflow_experiment_if_not_exists()
    run_name = f"{RunConfig.experiment_name}_feature_eng_2"
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        spark = SparkSession.builder.master(f"local[{RunConfig.n_jobs}]").getOrCreate()
        data = spark.read.parquet(first_pass_feature_eng)
        extractor = FeatureExtractor()
        extractor.fit_stats(data)
        extractor.save_to_mlflow(run_id=run_id)
        data = extractor.second_pass_feature_engineering(data)
        path = FileConfig.second_features_data_path
        data.write.mode("overwrite").parquet(path)
        
    return path, run_id


def _run_training_eval(data_path: str, context: dg.AssetExecutionContext, run_name_suffix: str, run_id: str = None):
    create_mlflow_experiment_if_not_exists()
    run_name = f"{RunConfig.experiment_name}_{run_name_suffix}"
    
    with mlflow.start_run(run_id=run_id, run_name=run_name) as run:
        run_id = run.info.run_id  # Capture run id in case it is initially None
        spark = SparkSession.builder.master(f"local[{RunConfig.n_jobs}]").getOrCreate()
        data = spark.read.parquet(data_path)
        mlflow.log_params({
            "data_path": data_path,
            "n_jobs": ModelConfig.n_jobs,
            "max_rounds": ModelConfig.max_rounds
        })

        train_X, train_y, test_X, test_y = split_data(data)
        classifier = RansomwareClassifier()
        classifier.train(train_X, train_y)

        probabilities_training, local_attributions_training = classifier.explain(train_X)
        threshold = classifier.get_optimal_threshold(train_y, probabilities_training)
        mlflow.log_param("optimized_threshold", threshold)
        
        feature_info = classifier.determine_important_features(
            train_y, probabilities_training, local_attributions_training)
        mlflow.log_dict(feature_info[ModelConfig.true_positives_key_name], 
                        f"{ModelConfig.true_positives_key_name}.json")
        mlflow.log_dict(feature_info[ModelConfig.false_negatives_key_name], 
                        f"{ModelConfig.false_negatives_key_name}.json")
        
        probabilities_testing, _ = classifier.explain(test_X)
        metrics = classifier.evaluate(test_y, probabilities_testing)
        mlflow.log_metrics(metrics)
        classifier.save_to_mlflow()
        
        context.log.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        context.log.info(f"Results: {metrics}")
        
        return dg.Output(
            value=metrics,
            metadata={
                "MLflow Run ID": run_id,
                "Accuracy": metrics["accuracy"],
                "True Positives": metrics["tp"],
                "False Negatives": metrics["fn"],
                
                "Weakest TP Features": dg.MetadataValue.md(
                    f"```json\n{json.dumps(feature_info[ModelConfig.true_positives_key_name], indent=2)}\n```"
                ),
                "Strongest FN Features": dg.MetadataValue.md(
                    f"```json\n{json.dumps(feature_info[ModelConfig.false_negatives_key_name], indent=2)}\n```"
                )
            }
    )

@dg.asset(group_name="model_training")
def results_baseline(context: dg.AssetExecutionContext, preprocessing: str):
    return _run_training_eval(preprocessing, context, "baseline")

@dg.asset(group_name="model_training")
def results_first_pass_feature_engineering(context: dg.AssetExecutionContext, first_pass_feature_eng: str):
    return _run_training_eval(first_pass_feature_eng, context, "feature_eng_1")

@dg.asset(group_name="model_training")
def results_second_pass_feature_engineering(context: dg.AssetExecutionContext, second_pass_feature_eng: tuple[str, str]):
    data_path, run_id = second_pass_feature_eng
    return _run_training_eval(data_path, context, "feature_eng_2", run_id)