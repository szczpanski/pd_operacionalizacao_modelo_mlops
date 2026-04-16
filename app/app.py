"""
app.py
------
Interface Streamlit para inferência do modelo final do projeto de risco de crédito.

Uso:
    streamlit run app/app.py
"""

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Scoring de Crédito | Projeto de Disciplina",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
LOCAL_MODEL_FILE = MODEL_DIR / "champion_model.joblib"
RUN_ID_FILE = MODEL_DIR / "champion_run_id.txt"
MLFLOW_DB_FILE = PROJECT_ROOT / "mlflow.db"

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Utilização de crédito rotativo",
    "age": "Idade",
    "NumberOfTime30-59DaysPastDueNotWorse": "Atrasos 30–59 dias",
    "DebtRatio": "Razão de endividamento",
    "MonthlyIncome": "Renda mensal",
    "NumberOfOpenCreditLinesAndLoans": "Linhas de crédito abertas",
    "NumberOfTimes90DaysLate": "Atrasos acima de 90 dias",
    "NumberRealEstateLoansOrLines": "Empréstimos imobiliários",
    "NumberOfTime60-89DaysPastDueNotWorse": "Atrasos 60–89 dias",
    "NumberOfDependents": "Número de dependentes",
}

FEATURE_HELP = {
    "RevolvingUtilizationOfUnsecuredLines": "Percentual do limite rotativo utilizado. Valores altos indicam maior risco.",
    "age": "Idade do solicitante em anos.",
    "NumberOfTime30-59DaysPastDueNotWorse": "Quantidade de atrasos entre 30 e 59 dias nos últimos 2 anos.",
    "DebtRatio": "Razão entre pagamentos de dívida e renda mensal.",
    "MonthlyIncome": "Renda mensal estimada em dólares.",
    "NumberOfOpenCreditLinesAndLoans": "Total de linhas de crédito e empréstimos atualmente abertos.",
    "NumberOfTimes90DaysLate": "Quantidade de atrasos acima de 90 dias. Forte sinal de risco.",
    "NumberRealEstateLoansOrLines": "Quantidade de empréstimos ou linhas imobiliárias.",
    "NumberOfTime60-89DaysPastDueNotWorse": "Quantidade de atrasos entre 60 e 89 dias nos últimos 2 anos.",
    "NumberOfDependents": "Número de dependentes na família.",
}

DEFAULTS = {
    "RevolvingUtilizationOfUnsecuredLines": 0.35,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 5000.0,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0,
}


@st.cache_resource(show_spinner="Carregando modelo campeão...")
def load_model():
    model_uri = os.environ.get("MODEL_URI")

    if MLFLOW_DB_FILE.exists():
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_FILE}")

    if model_uri:
        if model_uri.endswith(".joblib") and Path(model_uri).exists():
            return joblib.load(model_uri), model_uri
        return mlflow.sklearn.load_model(model_uri), model_uri

    if LOCAL_MODEL_FILE.exists():
        return joblib.load(LOCAL_MODEL_FILE), str(LOCAL_MODEL_FILE)

    if RUN_ID_FILE.exists():
        run_id = RUN_ID_FILE.read_text(encoding="utf-8").strip()
        if run_id:
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            return model, model_uri

    raise FileNotFoundError(
        "Modelo não encontrado. Gere models/champion_model.joblib "
        "ou models/champion_run_id.txt, ou defina MODEL_URI."
    )


def get_model_name(model):
    clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else None
    return type(clf).__name__ if clf is not None else type(model).__name__


def detect_reduction(model):
    if not hasattr(model, "named_steps"):
        return None
    if "pca" in model.named_steps:
        return "PCA"
    if "lda" in model.named_steps:
        return "LDA"
    return None


def render_score_card(pred, proba):
    prob_good = float(proba[0])
    prob_bad = float(proba[1])

    if pred == 0:
        st.success("Baixo risco: o modelo classificou este solicitante como adimplente.")
    else:
        st.error("Alto risco: o modelo classificou este solicitante como potencialmente inadimplente.")

    c1, c2 = st.columns(2)
    c1.metric("Probabilidade adimplente", f"{prob_good:.1%}")
    c2.metric("Probabilidade inadimplente", f"{prob_bad:.1%}")

    fig = go.Figure(
        go.Bar(
            x=["Adimplente", "Inadimplente"],
            y=[prob_good, prob_bad],
            marker_color=["#43a047", "#e53935"],
            text=[f"{prob_good:.1%}", f"{prob_bad:.1%}"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Probabilidade estimada por classe",
        yaxis=dict(range=[0, 1.15], tickformat=".0%"),
        height=350,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_model_info(model, model_uri):
    reduction = detect_reduction(model)
    clf_name = get_model_name(model)

    col1, col2, col3 = st.columns(3)
    col1.metric("Modelo", clf_name)
    col2.metric("Redução dimensional", reduction or "Não aplicada")
    col3.metric("Fonte", "MLflow" if model_uri.startswith("runs:/") else "Arquivo local")

    st.caption(f"Artefato carregado: {model_uri}")


def render_feature_importance(model, reduction):
    if reduction:
        st.info(
            f"O modelo utiliza {reduction}, então a explicabilidade direta das features originais fica limitada."
        )
        return

    if not hasattr(model, "named_steps"):
        st.info("Estrutura do modelo não compatível com extração de importância.")
        return

    clf = model.named_steps.get("clf")
    if clf is None or not hasattr(clf, "feature_importances_"):
        st.info("Este modelo não expõe feature importance.")
        return

    importances = clf.feature_importances_
    df_imp = pd.DataFrame({"feature": FEATURES, "importance": importances}).sort_values(
        "importance", ascending=True
    )

    labels = [FEATURE_LABELS[f] for f in df_imp["feature"]]

    fig = go.Figure(
        go.Bar(
            x=df_imp["importance"],
            y=labels,
            orientation="h",
            marker_color="#1565c0",
            text=[f"{v:.3f}" for v in df_imp["importance"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Importância das variáveis no modelo",
        xaxis_title="Importância",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


st.title("Sistema de Scoring de Crédito")
st.caption(
    "Projeto de Disciplina — Fundamentos de Machine Learning com scikit-learn | "
    "Avaliação de risco de crédito com foco em reprodutibilidade, rastreabilidade e inferência."
)
st.divider()

model, model_uri = load_model()
reduction = detect_reduction(model)

with st.expander("Informações do modelo carregado", expanded=False):
    render_model_info(model, model_uri)

st.sidebar.header("Dados do solicitante")
st.sidebar.caption("Preencha os campos e clique em avaliar.")

inputs = {}
inputs["RevolvingUtilizationOfUnsecuredLines"] = st.sidebar.slider(
    FEATURE_LABELS["RevolvingUtilizationOfUnsecuredLines"],
    0.0, 1.0, float(DEFAULTS["RevolvingUtilizationOfUnsecuredLines"]), 0.01,
    help=FEATURE_HELP["RevolvingUtilizationOfUnsecuredLines"],
)
inputs["age"] = st.sidebar.number_input(
    FEATURE_LABELS["age"], min_value=18, max_value=100, value=int(DEFAULTS["age"]), step=1,
    help=FEATURE_HELP["age"],
)
inputs["NumberOfTime30-59DaysPastDueNotWorse"] = st.sidebar.number_input(
    FEATURE_LABELS["NumberOfTime30-59DaysPastDueNotWorse"], min_value=0, max_value=20,
    value=int(DEFAULTS["NumberOfTime30-59DaysPastDueNotWorse"]), step=1,
    help=FEATURE_HELP["NumberOfTime30-59DaysPastDueNotWorse"],
)
inputs["DebtRatio"] = st.sidebar.number_input(
    FEATURE_LABELS["DebtRatio"], min_value=0.0, max_value=10.0,
    value=float(DEFAULTS["DebtRatio"]), step=0.01, format="%.2f",
    help=FEATURE_HELP["DebtRatio"],
)
inputs["MonthlyIncome"] = st.sidebar.number_input(
    FEATURE_LABELS["MonthlyIncome"], min_value=0.0, max_value=500000.0,
    value=float(DEFAULTS["MonthlyIncome"]), step=100.0, format="%.0f",
    help=FEATURE_HELP["MonthlyIncome"],
)
inputs["NumberOfOpenCreditLinesAndLoans"] = st.sidebar.number_input(
    FEATURE_LABELS["NumberOfOpenCreditLinesAndLoans"], min_value=0, max_value=50,
    value=int(DEFAULTS["NumberOfOpenCreditLinesAndLoans"]), step=1,
    help=FEATURE_HELP["NumberOfOpenCreditLinesAndLoans"],
)
inputs["NumberOfTimes90DaysLate"] = st.sidebar.number_input(
    FEATURE_LABELS["NumberOfTimes90DaysLate"], min_value=0, max_value=20,
    value=int(DEFAULTS["NumberOfTimes90DaysLate"]), step=1,
    help=FEATURE_HELP["NumberOfTimes90DaysLate"],
)
inputs["NumberRealEstateLoansOrLines"] = st.sidebar.number_input(
    FEATURE_LABELS["NumberRealEstateLoansOrLines"], min_value=0, max_value=20,
    value=int(DEFAULTS["NumberRealEstateLoansOrLines"]), step=1,
    help=FEATURE_HELP["NumberRealEstateLoansOrLines"],
)
inputs["NumberOfTime60-89DaysPastDueNotWorse"] = st.sidebar.number_input(
    FEATURE_LABELS["NumberOfTime60-89DaysPastDueNotWorse"], min_value=0, max_value=20,
    value=int(DEFAULTS["NumberOfTime60-89DaysPastDueNotWorse"]), step=1,
    help=FEATURE_HELP["NumberOfTime60-89DaysPastDueNotWorse"],
)
inputs["NumberOfDependents"] = st.sidebar.number_input(
    FEATURE_LABELS["NumberOfDependents"], min_value=0, max_value=20,
    value=int(DEFAULTS["NumberOfDependents"]), step=1,
    help=FEATURE_HELP["NumberOfDependents"],
)

st.sidebar.divider()
run = st.sidebar.button("Avaliar risco", type="primary", use_container_width=True)

if run:
    x = pd.DataFrame([inputs])[FEATURES]
    pred = int(model.predict(x)[0])
    proba = model.predict_proba(x)[0]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Resultado da análise")
        render_score_card(pred, proba)

    with col2:
        st.subheader("Explicabilidade")
        render_feature_importance(model, reduction)

    with st.expander("Dados enviados ao modelo", expanded=False):
        st.dataframe(
            pd.DataFrame(
                {
                    "Variável": [FEATURE_LABELS[f] for f in FEATURES],
                    "Valor": [inputs[f] for f in FEATURES],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
else:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Como usar")
        st.markdown(
            """
            1. Preencha os dados do solicitante na barra lateral.
            2. Clique em **Avaliar risco**.
            3. Veja o resultado, as probabilidades e a explicabilidade do modelo.
            """
        )

    with c2:
        st.subheader("Resumo do projeto")
        st.markdown(
            """
            - Problema: classificação binária de inadimplência.
            - Métrica principal: F1-Score.
            - Modelo final: Random Forest.
            - Estratégia: comparação sistemática com baseline e árvores regularizadas.
            - Rastreamento: MLflow.
            """
        )
