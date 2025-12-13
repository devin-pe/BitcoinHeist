import dagster as dg
from src.dags import training_dag, data_drift_dag

assets = dg.load_assets_from_modules([training_dag, data_drift_dag])

defs = dg.Definitions(
    assets=assets,
    sensors=[dg.AutomationConditionSensorDefinition("automation_sensor", target="*")],
)

if __name__ == "__main__":
    result = dg.materialize(assets)
    
    if result.success:
        print("Run successful")
    else:
        print("Run failed")