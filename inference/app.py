"""Backend de inferência: carrega models/model.joblib e prevê o fechamento do ouro no próximo dia útil."""

import os
import warnings

import joblib
import numpy as np
from flask import Flask, jsonify, request

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

app = Flask(__name__)
app.json.ensure_ascii = False  # mostra acentos nas respostas em vez de ç

# O modelo é carregado uma única vez, quando o serviço sobe. Se o artefato não
# existir, o serviço continua no ar e o /health avisa que o modelo não carregou.
try:
    artifact = joblib.load(MODEL_PATH)
    load_error = None
except Exception as e:
    artifact = None
    load_error = str(e)


def bad_request(message):
    return jsonify({"error": message}), 400


def parse_days(payload):
    """Valida o JSON e devolve a lista de dias, do mais antigo ao mais recente."""
    n_lags = artifact["n_lags"]
    use_range = artifact["use_range_rel"]

    if not isinstance(payload, dict) or not isinstance(payload.get("days"), list):
        raise ValueError('O corpo deve ser um JSON no formato {"days": [...]}')
    days = payload["days"]
    if len(days) != n_lags:
        raise ValueError(f"Envie exatamente {n_lags} dias (recebidos: {len(days)})")

    required = ["close", "high", "low"] if use_range else ["close"]
    for i, day in enumerate(days):
        if not isinstance(day, dict):
            raise ValueError(f"days[{i}] deve ser um objeto, ex.: {{\"close\": 4155.8}}")
        for field in required:
            value = day.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"days[{i}].{field} deve ser um número positivo")
    return days


def build_features(days):
    """Monta as features na mesma ordem usada no treino (t-4 ... t)."""
    closes = [day["close"] for day in days]
    values = closes
    if artifact["use_range_rel"]:
        values = closes + [(day["high"] - day["low"]) / day["close"] for day in days]
    return np.array([values], dtype=float)


@app.get("/health")
def health():
    if artifact is None:
        return jsonify({"status": "error", "model_loaded": False, "error": load_error}), 503
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "trained_until": artifact["trained_until"],
        "features": artifact["features"],
    })


@app.post("/predict")
def predict():
    if artifact is None:
        return jsonify({"error": "Modelo não carregado", "detail": load_error}), 503

    payload = request.get_json(silent=True)
    try:
        days = parse_days(payload)
    except ValueError as e:
        return bad_request(str(e))

    X = build_features(days)
    with warnings.catch_warnings():
        # O modelo foi treinado com um DataFrame; aqui a ordem das colunas é garantida por build_features
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        prediction = float(artifact["model"].predict(X)[0])

    return jsonify({
        "predicted_close": round(prediction, 2),
        "input_last_close": days[-1]["close"],
        "unit": "USD por onça troy",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
