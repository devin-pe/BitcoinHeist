import os
from dataclasses import dataclass
from configs.configs import BaseRunConfig, BaseApiConfig, BaseTelemetryConfig


@dataclass(init=False, frozen=True)
class DockerRunConfig(BaseRunConfig):
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI")


@dataclass(init=False, frozen=True)
class DockerApiConfig(BaseApiConfig):
    api_uri: str = "http://app:5001"
    

@dataclass(init=False, frozen=True)
class DockerTelemetryConfig(BaseTelemetryConfig):
    push_gateway_uri: str = "prometheus_push_gateway:9091"
    