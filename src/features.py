from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import os
import pickle
import tempfile
import typing
import mlflow
from configs.configs import FileConfig, FeatureConfig


class FeatureExtractor:
    def __init__(self):
        self._stats: dict[str, dict[str, float]] = {}
        self.epsilon = FeatureConfig.epsilon
        self._is_fitted = False


    def transform(self, data: DataFrame) -> DataFrame:
        data = self.first_pass_feature_engineering(data)

        if not self._is_fitted:
            self.fit_stats(data)

        data = self.second_pass_feature_engineering(data)
        
        return data
    
    
    def first_pass_feature_engineering(self, data: DataFrame) -> DataFrame:
        for col in FeatureConfig.cols_to_log:
            data = data.withColumn(f"log_{col}", F.log1p(F.col(col)))  

        for (numerator_col, denominator_col) in FeatureConfig.first_interaction_cols:
            data = data.withColumn(f"{numerator_col}_per_{denominator_col}", 
                                F.col(numerator_col) / (F.col(denominator_col) 
                                    + FeatureConfig.epsilon))
        data = data.drop(*FeatureConfig.cols_to_log)
        
        return data


    def second_pass_feature_engineering(self, data: DataFrame) -> DataFrame:
        for (numerator_col, denominator_col) in FeatureConfig.second_interaction_cols:
            data = data.withColumn(f"{numerator_col}_per_{denominator_col}", 
                                F.col(numerator_col) / (F.col(denominator_col) 
                                    + FeatureConfig.epsilon))
        
        for col, stats in self._stats.items():
            if col in data.columns:
                mean = stats["mean"]
                std = stats["std"]
                safe_std = std if std > self.epsilon else 1.0
                
                data = data.withColumn(col, (F.col(col) - F.lit(mean)) / F.lit(safe_std))
        
        return data


    def fit_stats(self, data: DataFrame):
        for col_name in FeatureConfig.z_score_cols:
            stats = data.select(F.mean(col_name), F.stddev(col_name)).first()
            self._stats[col_name] = {"mean": stats[0], "std":  stats[1]}

        self._is_fitted = True


    def save_to_mlflow(self, run_id: str):
        if not self._is_fitted:
             raise RuntimeError("FeatureExtractor not fitted yet.")

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, FileConfig.artifact_file)
            
            with open(file_path, "wb") as f:
                pickle.dump(self._stats, f)
            mlflow_client = mlflow.tracking.MlflowClient()
            mlflow_client.log_artifact(run_id, file_path, artifact_path=FileConfig.artifact_folder)


    def load_from_mlflow(self, run_id: str):
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, 
                artifact_path=f"{FileConfig.artifact_folder}/{FileConfig.artifact_file}",
                dst_path=temp_dir
            )
            
            with open(local_path, "rb") as f:
                self._stats = pickle.load(f)
            
            self._is_fitted = True
