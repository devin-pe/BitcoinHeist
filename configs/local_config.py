from dataclasses import dataclass
from configs.configs import BaseRunConfig, BaseApiConfig, BaseTelemetryConfig


@dataclass(init=False, frozen=True)
class LocalRunConfig(BaseRunConfig):
    mlflow_tracking_uri: str = "http://localhost:8080"


@dataclass(init=False, frozen=True)
class LocalApiConfig(BaseApiConfig):
    api_uri: str = "http://localhost:5001"
    

@dataclass(init=False, frozen=True)
class LocalTelemetryConfig(BaseTelemetryConfig):
    push_gateway_uri: str = "http://localhost:9091"