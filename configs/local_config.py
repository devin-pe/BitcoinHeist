from dataclasses import dataclass
from configs.configs import BaseRunConfig

@dataclass(init=False, frozen=True)
class LocalRunConfig(BaseRunConfig):
    mlflow_tracking_uri: str = "http://localhost:8080"