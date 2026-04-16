import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = [
    BASE_DIR / "model.pkl",
    BASE_DIR / "models" / "champion_model.joblib",
    BASE_DIR / "mlruns" / "1" / "models" / "m-8723ce0c939f41c18def448a905427f3" / "artifacts" / "model.pkl",
]

DATA_PATHS = [
    BASE_DIR / "data" / "raw" / "cs-training.csv",
    BASE_DIR / "data" / "cs-training.csv",
    BASE_DIR / "dataset" / "cs-training.csv",
]

model_path = next((p for p in MODEL_PATHS if p.exists()), None)
if model_path is None:
    raise FileNotFoundError("Não encontrei modelo em model.pkl, models/champion_model.joblib ou mlruns")

data_path = next((p for p in DATA_PATHS if p.exists()), None)
if data_path is None:
    raise FileNotFoundError("Não encontrei cs-training.csv em data/raw, data ou dataset")

print(f"Usando modelo: {model_path}")
print(f"Usando base: {data_path}")

df = pd.read_csv(data_path)

target = "SeriousDlqin2yrs"
if target not in df.columns:
    raise ValueError(f"Coluna target '{target}' não encontrada")

df = df.dropna(subset=[target]).copy()
X = df.drop(columns=[target])
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load(model_path)

def align_features(X_input, model_obj):
    if hasattr(model_obj, "feature_names_in_"):
        cols = list(model_obj.feature_names_in_)
        X_aligned = X_input.copy()
        for c in cols:
            if c not in X_aligned.columns:
                X_aligned[c] = np.nan
        X_aligned = X_aligned[cols]
        return X_aligned
    return X_input

X_test_aligned = align_features(X_test, model)

if not hasattr(model, "predict_proba"):
    raise AttributeError("O modelo não possui predict_proba()")

y_proba = model.predict_proba(X_test_aligned)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n=== MATRIZ DE CONFUSÃO ===")
print(cm)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n=== MÉTRICAS GERAIS ===")
print(f"accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"f1       : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"roc_auc  : {roc_auc_score(y_test, y_proba):.4f}")
print(f"tn       : {tn}")
print(f"fp       : {fp}")
print(f"fn       : {fn}")
print(f"tp       : {tp}")

thresholds = np.arange(0.10, 0.91, 0.05)
rows = []

for t in thresholds:
    pred_t = (y_proba >= t).astype(int)
    rows.append({
        "threshold": round(float(t), 2),
        "accuracy": accuracy_score(y_test, pred_t),
        "precision": precision_score(y_test, pred_t, zero_division=0),
        "recall": recall_score(y_test, pred_t, zero_division=0),
        "f1": f1_score(y_test, pred_t, zero_division=0)
    })

results = pd.DataFrame(rows).sort_values(["f1", "recall"], ascending=False)

print("\n=== TOP THRESHOLDS POR F1 ===")
print(results.head(10).to_string(index=False))

results.to_csv(BASE_DIR / "threshold_results.csv", index=False)
print(f"\nArquivo salvo: {BASE_DIR / 'threshold_results.csv'}")

if hasattr(model, "feature_importances_"):
    imp = pd.DataFrame({
        "feature": X_test_aligned.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n=== TOP 15 FEATURE IMPORTANCES ===")
    print(imp.head(15).to_string(index=False))

    imp.to_csv(BASE_DIR / "feature_importances.csv", index=False)
    print(f"Arquivo salvo: {BASE_DIR / 'feature_importances.csv'}")

elif hasattr(model, "coef_"):
    coef = pd.DataFrame({
        "feature": X_test_aligned.columns,
        "coef": model.coef_[0]
    })
    coef["abs_coef"] = coef["coef"].abs()
    coef = coef.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])

    print("\n=== TOP 15 COEFICIENTES ===")
    print(coef.head(15).to_string(index=False))

    coef.to_csv(BASE_DIR / "feature_coefficients.csv", index=False)
    print(f"Arquivo salvo: {BASE_DIR / 'feature_coefficients.csv'}")
else:
    print("\nModelo sem feature_importances_ ou coef_.")
