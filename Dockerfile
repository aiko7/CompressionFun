FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu20.04

RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev curl git

RUN curl -sSL https://install.python-poetry.org | python3.11 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry env use python3.11 && poetry install --no-dev

COPY . .

CMD ["poetry", "run", "python", "main.py"]
