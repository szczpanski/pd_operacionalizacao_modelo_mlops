import json
from pathlib import Path
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training import FEATURES, TARGET, extract_model_params, train_all_models
from src.evaluation import evaluate_classifier


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cs-training.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
EXPERIMENT_NAME = "risco_credito"


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset: {missing}")

    return df


def save_support_files(best_result: dict) -> tuple[Path, Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "champion_model.joblib"
    joblib.dump(best_result["model"], model_path)

    metrics_path = MODELS_DIR / "champion_metrics.json"
    metrics_payload = {
        "model_name": best_result["model_name"],
        "metrics": best_result["metrics"],
        "params": best_result["params"],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    return model_path, metrics_path


def main():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_dataset(DATA_PATH)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    fitted_models = train_all_models(X_train, y_train, random_state=42)

    comparison = []
    best_result = None

    for model_name, model in fitted_models.items():
        evaluation = evaluate_classifier(model, X_test, y_test)
        metrics = evaluation["metrics"]
        params = extract_model_params(model_name, model)

        current_result = {
            "model_name": model_name,
            "model": model,
            "metrics": metrics,
            "params": params,
        }
        comparison.append(
            {
                "model_name": model_name,
                **metrics,
            }
        )

        if best_result is None or metrics["f1"] > best_result["metrics"]["f1"]:
            best_result = current_result

    if best_result is None:
        raise RuntimeError("Nenhum modelo foi treinado com sucesso.")

    model_path, metrics_path = save_support_files(best_result)

    with mlflow.start_run(run_name=f"champion_{best_result['model_name']}") as run:
        mlflow.set_tags(
            {
                "project": "pd_operacionalizacao_modelo_mlops",
                "stage": "training",
                "champion": "true",
            }
        )

        mlflow.log_params(best_result["params"])
        mlflow.log_metric("train_rows", int(len(X_train)))
        mlflow.log_metric("test_rows", int(len(X_test)))

        for metric_name, metric_value in best_result["metrics"].items():
            mlflow.log_metric(metric_name, float(metric_value))

        signature = infer_signature(X_train, best_result["model"].predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=best_result["model"],
            artifact_path="model",
            signature=signature,
        )

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))

        comparison_path = MODELS_DIR / "all_model_results.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(str(comparison_path))

        run_id_path = MODELS_DIR / "champion_run_id.txt"
        run_id_path.write_text(run.info.run_id, encoding="utf-8")

    print("Treinamento concluído com sucesso.")
    print(f"Modelo campeão: {best_result['model_name']}")
    print(f"Métricas: {best_result['metrics']}")
    print(f"Modelo salvo em: {model_path}")
    print(f"Run ID salvo em: {MODELS_DIR / 'champion_run_id.txt'}")


if __name__ == "__main__":
    main()
