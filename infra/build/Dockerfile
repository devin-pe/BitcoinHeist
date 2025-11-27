FROM python:3.13-slim-bullseye

RUN apt-get update
RUN apt-get install openjdk-17-jdk --yes

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV PYTHONPATH=/app/

COPY src/ src/
COPY configs/ configs/
COPY data/dataset/ data/dataset/
COPY models/.placeholder models/.placeholder
