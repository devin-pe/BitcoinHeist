# Bitcoin Heist

This project provides an end-to-end data pipeline, model training workflow, and monitoring system for detecting ransomware-related Bitcoin addresses. It uses Dagster for orchestration, Docker for deployment, and exposes prediction and explainability APIs.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:devin-pe/BitcoinHeist.git
cd BitcoinHeist
```

### 2. Download the Dataset

Create a data directory at the project root and download the dataset into it:

```bash
mkdir data
curl -L https://archive.ics.uci.edu/static/public/526/bitcoinheistransomwareaddressdataset.zip -o data/bitcoinheist.zip
unzip data/bitcoinheist.zip -d data
```

### 3. Format the Dataset

Convert the raw CSV data into Parquet format. From root:

```bash
python3 -m scripts.csv_to_parquet
```

### 4. Build and Launch the Containers

```bash
sudo make up
```

## Running the Pipeline

### 5. Materialize Dagster Assets

Once Dagster is running, navigate to:

```
http://localhost:4242
```

Materialize all six assets under:

- `data_pipeline`
- `model_training`

Wait for training to complete.

### 6. Enable Automation

After training finishes, go to the **Automation** tab in Dagster and enable the automation sensor.  
This ensures PSI monitoring is active.

## Making Predictions

### 7. Load the Model and Run a Prediction

Send an initial prediction request to load the model:

```bash
curl   -H "Content-Type: application/json"   -d '{"length":18,"weight":0.00833333333333333,"count":1,"looped":0,"neighbors":2,"income":100050000}'   -X POST http://127.0.0.1:5001/predict
```

You may also use any example from `api_request_example.sh`.

### 8. View Model Explanations

Copy the returned `request_id` and navigate to:

```
http://localhost:5001/explain/<your_request_id>
```

This page shows:
- Features contributing most positively to a ransomware prediction
- Features contributing most negatively (white prediction)

## Alerts and Monitoring

### 9. Simulate Traffic and Alerts

From the project root, run:

```bash
python3 -m scripts.alert_triggers
```

This script simulates API calls to populate dashboards and trigger alerts. You may have to wait a while for it to complete but you can already begin visualizing the metrics in real time. 

### 10. View Alerts and Metrics

- Alert Manager: http://localhost:9093  
- Grafana Metrics Visualization: http://localhost:3030

