FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

# data/ e models/ chegam por volume (ver docker-compose.yml)
ENV DATA_PATH=/app/data/gold.csv \
    MODELS_DIR=/app/models

CMD ["python", "train.py"]
