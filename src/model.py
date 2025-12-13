import mlflow
import pickle
import pandas as pd
import numpy as np
import json
import tempfile
import os
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.utils import inv_link
from configs.configs import RunConfig, ModelConfig

class RansomwareClassifier:
    def __init__(self):
        self.model = ExplainableBoostingClassifier(
            random_state=RunConfig.seed,
            n_jobs=ModelConfig.n_jobs,
            max_rounds=ModelConfig.max_rounds,
            early_stopping_rounds=50
        )
        self.threshold: float = 0.5
        self._is_trained: bool = False


    def train(self, train_X: pd.DataFrame, train_y: pd.Series) -> None:
        self.model.fit(train_X, train_y)
        self._is_trained = True


    def explain(self, train_X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet.")
        
        local_explanation_values = self.model.eval_terms(train_X.values)
        local_attributions = pd.DataFrame(
            local_explanation_values,
            columns=self.model.term_names_,
            index=train_X.index
        )
        
        raw_preds = local_explanation_values.sum(axis=1) + self.model.intercept_
        probabilities = inv_link(raw_preds, self.model.link_)[:, 1] # get only ransomware probs
        
        return probabilities, local_attributions


    def get_optimal_threshold(self, y_true: pd.Series, probabilities: np.ndarray) -> None:
        fpr, tpr, thresholds = roc_curve(y_true, probabilities)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        self.threshold = thresholds[optimal_idx]
        return self.threshold  # for logging purposes

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs, _ = self.explain(X)
        return (probs >= self.threshold).astype(int)


    def evaluate(self, y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
        y_pred = (probabilities >= self.threshold).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        _, _, fn, tp = conf_matrix.ravel()
        
        return {
            "accuracy": float(acc),
            "fn": int(fn),
            "tp": int(tp),
        }
        
    
    def determine_important_features(self, y_true: pd.Series, probabilities: np.ndarray, local_attributions: pd.DataFrame
                                    ) -> dict[str, dict[str, float]]:
        predictions = (probabilities >= self.threshold)
        
        tp_mask = (y_true == 1) & (predictions == 1)
        fn_mask = (y_true == 1) & (predictions == 0)
        
        feature_info = {}
        
        for key, mask in [(ModelConfig.true_positives_key_name, tp_mask), 
                          (ModelConfig.false_negatives_key_name, fn_mask)]:
            if mask.sum() > 0:
                filtered_attributions = local_attributions.loc[mask]
                mean_attributions = filtered_attributions.mean(axis=0).sort_values(ascending=True).head(5)
                
                feature_info[key] = mean_attributions.to_dict()
            
        return feature_info


    def save_to_mlflow(self: str):
        mlflow_experiment = mlflow.get_experiment_by_name(RunConfig.experiment_name)
        if mlflow_experiment is None:
            raise RuntimeError(f"Experiment {RunConfig.experiment_name} does not exist in MLFlow.")

        with tempfile.TemporaryDirectory() as temp_dir:
            local_artifact_path = os.path.join(temp_dir, ModelConfig.artifact_folder)
            os.makedirs(local_artifact_path, exist_ok=True)

            model_path = os.path.join(local_artifact_path, f"{ModelConfig.model_name}.pkl")
            with open(model_path, "wb") as file:
                pickle.dump(self.model, file)

            meta_path = os.path.join(local_artifact_path, ModelConfig.threshold_file)
            with open(meta_path, "w") as file:
                json.dump({"threshold": self.threshold}, file)
            
            mlflow.log_artifacts(local_dir=temp_dir)


    def load_from_mlflow(self, run_id: str):
        mlflow_experiment = mlflow.get_experiment_by_name(RunConfig.experiment_name)
        if mlflow_experiment is None:
            raise RuntimeError(f"Experiment {RunConfig.experiment_name} does not exist in MLFlow.")

        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=ModelConfig.artifact_folder,
                dst_path=temp_dir,
                tracking_uri=RunConfig.mlflow_tracking_uri,
            )

            artifacts_path = os.path.join(temp_dir, ModelConfig.artifact_folder)
            
            model_file = f"{ModelConfig.model_name}.pkl"
            model_path = os.path.join(artifacts_path, model_file)
            threshold_path = os.path.join(artifacts_path, ModelConfig.threshold_file)


            with open(model_path, "rb") as file:
                self.model = pickle.load(file)

            with open(threshold_path, "r") as file:
                threshold_data = json.load(file)
                self.threshold = threshold_data.get("threshold")
                
            self._is_trained = True