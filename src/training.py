from typing import Dict, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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

TARGET = "SeriousDlqin2yrs"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_candidate_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    preprocessor = build_preprocessor()

    models = {
        "perceptron": Perceptron(
            max_iter=1000,
            tol=1e-3,
            random_state=random_state,
            class_weight="balanced",
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            random_state=random_state,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }

    pipelines = {}
    for name, clf in models.items():
        pipelines[name] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("clf", clf),
            ]
        )
    return pipelines


def train_all_models(X_train, y_train, random_state: int = 42) -> Dict[str, Pipeline]:
    pipelines = build_candidate_pipelines(random_state=random_state)

    fitted_models = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

    return fitted_models


def extract_model_params(model_name: str, pipeline: Pipeline) -> Dict:
    clf = pipeline.named_steps["clf"]
    params = clf.get_params()

    selected = {"model_name": model_name}
    for key in [
        "max_iter",
        "tol",
        "max_depth",
        "min_samples_leaf",
        "n_estimators",
        "random_state",
        "class_weight",
    ]:
        if key in params:
            selected[key] = params[key]

    return selected
