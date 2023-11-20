# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.5
FROM continuumio/miniconda3 as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY conda_environment.yml .
RUN conda env create -f conda_environment.yml
RUN conda clean --all --yes

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "housing", "/bin/bash", "-c"]

COPY . .

# Set the entrypoint to run your Flask app
ENTRYPOINT ["/opt/conda/envs/housing/bin/python3", "-m", "flask", "run", "--host=0.0.0.0"]

EXPOSE 5000
