"""Treina uma regressão linear para prever o fechamento do ouro (GC=F) no próximo dia útil.

Entrada: data/gold.csv (Date, Close, High, Low, Open, Volume)
Saída:   models/model.joblib, models/metrics.json, models/test_predictions.png
"""

import json
import os
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")  # sem interface gráfica dentro do container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

DATA_PATH = os.getenv("DATA_PATH", "data/gold.csv")
MODELS_DIR = os.getenv("MODELS_DIR", "models")

N_LAGS = 5  # últimos 5 pregões
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # o restante (15%) é teste


def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # O CSV pode ter sido baixado com o mercado aberto: a última linha não tem o
    # fechamento real. Descartá-la sempre é mais seguro do que comparar com a data
    # de hoje, que só funcionaria se o treino rodasse no mesmo dia do download.
    df = df.iloc[:-1]

    df["range_rel"] = (df["High"] - df["Low"]) / df["Close"]
    return df


def build_features(df):
    """Cada linha usa os dias t-4..t para prever o fechamento de t+1."""
    feats = pd.DataFrame({"Date": df["Date"]})
    for i in range(N_LAGS - 1, -1, -1):  # do mais antigo (t-4) ao mais recente (t)
        feats[f"close_t-{i}"] = df["Close"].shift(i)
    for i in range(N_LAGS - 1, -1, -1):
        feats[f"range_rel_t-{i}"] = df["range_rel"].shift(i)
    feats["target"] = df["Close"].shift(-1)

    # As 5 primeiras linhas não têm histórico suficiente e a última não tem alvo
    return feats.dropna().reset_index(drop=True)


def chronological_split(feats):
    n = len(feats)
    i_val = int(n * TRAIN_FRAC)
    i_test = int(n * (TRAIN_FRAC + VAL_FRAC))
    return feats.iloc[:i_val], feats.iloc[i_val:i_test], feats.iloc[i_test:]


def metrics(y_true, y_pred):
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "mape_pct": round(float(mean_absolute_percentage_error(y_true, y_pred)) * 100, 3),
    }


def period(part):
    return f"{part['Date'].iloc[0].date()} -> {part['Date'].iloc[-1].date()} ({len(part)} linhas)"


def main():
    df = load_data(DATA_PATH)
    feats = build_features(df)
    train, val, test = chronological_split(feats)

    print(f"Dados usados: {df['Date'].iloc[0].date()} -> {df['Date'].iloc[-1].date()}")
    print(f"Treino:    {period(train)}")
    print(f"Validação: {period(val)}")
    print(f"Teste:     {period(test)}")

    close_cols = [f"close_t-{i}" for i in range(N_LAGS - 1, -1, -1)]
    range_cols = [f"range_rel_t-{i}" for i in range(N_LAGS - 1, -1, -1)]
    configs = {
        "closes": close_cols,
        "closes+range_rel": close_cols + range_cols,
    }

    # 1) Validação: compara as configurações entre si e com o baseline "amanhã = hoje"
    print("\nValidação:")
    val_results = {"baseline": metrics(val["target"], val["close_t-0"])}
    for name, cols in configs.items():
        model = LinearRegression().fit(train[cols], train["target"])
        val_results[name] = metrics(val["target"], model.predict(val[cols]))
    for name, m in val_results.items():
        print(f"  {name:<18} MAE = {m['mae']:>7.2f}  MAPE = {m['mape_pct']:.3f}%")

    best = min(configs, key=lambda name: val_results[name]["mae"])
    features = configs[best]
    print(f"Configuração escolhida: {best}")

    # 2) Modelo final: treino + validação, avaliado uma única vez no teste
    train_val = pd.concat([train, val])
    model = LinearRegression().fit(train_val[features], train_val["target"])
    test_pred = model.predict(test[features])
    test_results = {
        "baseline": metrics(test["target"], test["close_t-0"]),
        "model": metrics(test["target"], test_pred),
    }
    print("\nTeste:")
    for name, m in test_results.items():
        print(f"  {name:<18} MAE = {m['mae']:>7.2f}  MAPE = {m['mape_pct']:.3f}%")

    # 3) Exporta o artefato e as evidências
    os.makedirs(MODELS_DIR, exist_ok=True)
    artifact = {
        "model": model,
        "features": features,
        "use_range_rel": best == "closes+range_rel",
        "n_lags": N_LAGS,
        "trained_until": str(train_val["Date"].iloc[-1].date()),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, os.path.join(MODELS_DIR, "model.joblib"))

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"first": str(df["Date"].iloc[0].date()), "last": str(df["Date"].iloc[-1].date())},
        "split": {"train": period(train), "validation": period(val), "test": period(test)},
        "validation": val_results,
        "chosen_config": best,
        "test": test_results,
        "coefficients": dict(zip(features, np.round(model.coef_, 4).tolist())),
        "intercept": round(float(model.intercept_), 4),
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plt.figure(figsize=(11, 5))
    plt.plot(test["Date"], test["target"], label="Real", linewidth=1.5)
    plt.plot(test["Date"], test_pred, label="Previsto (regressão linear)", linewidth=1.2)
    plt.title("Ouro (GC=F): fechamento do próximo dia útil, conjunto de teste")
    plt.ylabel("USD por onça troy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "test_predictions.png"), dpi=120)

    print(f"\nArtefato salvo em {MODELS_DIR}/model.joblib")


if __name__ == "__main__":
    main()
