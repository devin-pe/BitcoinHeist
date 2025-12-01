FROM python:3.13-slim-bullseye

RUN apt-get update
RUN apt-get install openjdk-17-jdk --yes

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV PYTHONPATH=/app/

COPY src/ src/
COPY configs/ configs/
COPY models/.placeholder models/.placeholder

RUN mkdir -p /app/dagster_home
ENV DAGSTER_HOME=/app/dagster_home