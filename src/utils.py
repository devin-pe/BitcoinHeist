import mlflow
import json
from pathlib import Path
from configs.configs import RunConfig, FileConfig, PreprocessConfig
from functools import cache


def create_mlflow_experiment_if_not_exists():
    mlflow.set_tracking_uri(RunConfig.mlflow_tracking_uri)
    if not mlflow.get_experiment_by_name(RunConfig.experiment_name):
        mlflow.create_experiment(RunConfig.experiment_name)
    experiment = mlflow.get_experiment_by_name(RunConfig.experiment_name)
    experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    

@cache
def get_best_model_run_id() -> str:
    experiment = mlflow.get_experiment_by_name(RunConfig.experiment_name)
    filter_string = "status = 'FINISHED' AND attributes.run_name LIKE '%feature_eng_2'"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string, 
        order_by=["metrics.fn ASC"], 
        max_results=1
    )
    best_run = runs.iloc[0]
    return best_run.run_id


def append_live_telemetry(telemetry_entry: dict) -> None:
    telemetry_path = Path(FileConfig.telemetry_live_data_path)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    
    if telemetry_path.exists():
        with open(telemetry_path, "r") as f:
            telemetry_data = json.load(f)
    else:
        telemetry_data = []
    
    telemetry_data.append(telemetry_entry)
    
    with open(telemetry_path, "w") as f:
        json.dump(telemetry_data, f)


def save_training_telemetry_baseline(train_y) -> None:
    training_telemetry = {
        PreprocessConfig.target_col: {
            "true": int((train_y == 1).sum()),
            "false": int((train_y == 0).sum())
        }
    }
    
    telemetry_path = Path(FileConfig.telemetry_training_data_path)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(telemetry_path, "w") as f:
        json.dump(training_telemetry, f, indent=2)