from dataclasses import dataclass

@dataclass(init=False, frozen=True)
class RunConfig:
    sample_rate: float = 1.0
    negative_to_positive_ratio: float = 5.0
    num_folds: int = 5
    mlflow_tracking_uri: str = "http://mlflow:8080"
    experiment_name: str = "BTC"
    seed = 42