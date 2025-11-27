from dataclasses import dataclass

@dataclass(init=False, frozen=True)
class RunConfig:
    sample_rate: float = 0.1
    num_folds: int = 5
    mlflow_tracking_uri: str = "http://localhost:8080"
    experiment_name: str = "BTC"
    seed = 42