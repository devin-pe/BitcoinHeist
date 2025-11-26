FROM python:3.13-slim-bullseye

RUN apt-get update
RUN apt-get install openjdk-17-jdk --yes

WORKDIR /project

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV PYTHONPATH=/project/

COPY src/ src/
COPY configs/ configs/
COPY data/trx/ data/trx/
COPY models/.placeholder models/.placeholder
