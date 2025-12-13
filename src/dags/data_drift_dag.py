import json
import math
import logging
import typing
import dagster as dg
import pandas as pd
from prometheus_client import Gauge, CollectorRegistry, push_to_gateway
from configs.configs import FileConfig, TelemetryConfig, RunConfig, PreprocessConfig


logger = logging.getLogger(__name__)


def get_psi(training_percentages: typing.Tuple[float], latest_percentages: typing.Tuple[float]) -> float:
    psi = 0
    for training, latest in zip(training_percentages, latest_percentages):
        psi += (latest - training) * math.log(latest / training, math.e)
    return psi


@dg.asset(
    group_name="data_drift",
    automation_condition=dg.AutomationCondition.on_cron("* * * * *")
)
def monitor_data_drift(context: dg.AssetExecutionContext):
    context.log.info("Starting data drift monitoring")
    
    with open(FileConfig.telemetry_training_data_path, "r") as file:
        training_telemetry_data = json.load(file)
    with open(FileConfig.telemetry_live_data_path, "r") as file:
        live_telemetry_data = json.load(file)
        live_telemetry_data = pd.DataFrame.from_records(live_telemetry_data)

    live_telemetry_data.sort_values("timestamp", inplace=True, ascending=False)
    latest_telemetry_data = live_telemetry_data.iloc[0 : TelemetryConfig.num_instances_for_trigger]

    telemetry_data_count = latest_telemetry_data.shape[0]
    if telemetry_data_count < TelemetryConfig.num_instances_for_trigger:
        context.log.warning(
            f"Telemetry calculation has {telemetry_data_count} which is less than "
            f"required {TelemetryConfig.num_instances_for_trigger}. Skipping drift calculation."
        )
        return dg.Output(
            value={"status": "skipped", "reason": "insufficient_data", "count": telemetry_data_count},
            metadata={
                "Status": "Skipped",
                "Reason": "Insufficient data",
                "Data Count": telemetry_data_count,
                "Required Count": TelemetryConfig.num_instances_for_trigger
            }
        )

    psi_s = {}
    target = PreprocessConfig.target_col
    training_count = training_telemetry_data[target]["true"] + training_telemetry_data[target]["false"]
    # Adding Epsilon to avoid division by zero err for when there is no instance of a prediction
    training_percentages = (
        (training_telemetry_data[target]["true"] / training_count) + TelemetryConfig.epsilon,
        (training_telemetry_data[target]["false"] / training_count) + TelemetryConfig.epsilon,
    )

    latest_count = latest_telemetry_data[target].shape[0]
    # Adding Epsilon to avoid division by zero err for when there is no instance of a prediction
    latest_percentages = (
        (sum(latest_telemetry_data[target] == True) / latest_count) + TelemetryConfig.epsilon,
        (sum(latest_telemetry_data[target] == False) / latest_count) + TelemetryConfig.epsilon,
    )

    psi_s[target] = get_psi(training_percentages, latest_percentages)
    context.log.info(f"PSI for {target}: {psi_s[target]:.4f}")

    registry = CollectorRegistry()
    psi_gauge = Gauge(
        f"{RunConfig.app_name}_psi_s",
        "PSI calculations for feature and output data",
        labelnames=["target"],
        registry=registry,
    )
    for target in ["is_ransomware"]:
        psi_gauge.labels(target=target).set(psi_s[target])

    push_to_gateway(TelemetryConfig.push_gateway_uri, job="telemetryBatch", registry=registry)
    context.log.info("Successfully pushed PSI metrics to Prometheus Push Gateway")
    

    return dg.Output(
        value={
            "status": "success",
            "psi_scores": psi_s,
            "data_count": telemetry_data_count
        },
        metadata={
            "Status": "Success",
            "PSI for is_ransomware": psi_s["is_ransomware"],
            "Data Count": telemetry_data_count,
            "Training True %": float(training_percentages[0]),
            "Training False %": float(training_percentages[1]),
            "Latest True %": float(latest_percentages[0]),
            "Latest False %": float(latest_percentages[1])
        }
    )