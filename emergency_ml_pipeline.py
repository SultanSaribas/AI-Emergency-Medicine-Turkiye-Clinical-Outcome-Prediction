from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


TARGET_COLS = ["adverse_outcome", "readmission_30d"]
ID_COLS = ["encounter_id"]
LEAKAGE_COLS = ["disposition"]

LAB_COLS_WITH_INFORMATIVE_MISSINGNESS = [
    "troponin_i_ngml",
    "bnp_pgl",
    "d_dimer_mgfl",
    "lactate_mmoll",
    "lipase_ul",
    "procalcitonin_ngl",
    "albumin_gdl",
    "egfr_ml_min",
]


@dataclass(frozen=True)
class BaselineResult:
    target: str
    model: Any
    metrics: pd.Series
    confusion_matrix: pd.DataFrame
    feature_importance: pd.DataFrame
    train_shape: tuple[int, int]
    valid_shape: tuple[int, int]
    numeric_features: list[str]
    categorical_features: list[str]


def data_paths(data_dir: str | Path = "data") -> dict[str, Path]:
    data_dir = Path(data_dir)
    return {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "supplemental": data_dir / "supplemental_data.csv",
        "sample_submission": data_dir / "sample_submission.csv",
        "hospital_reference": data_dir / "hospital_reference.csv",
        "icd10_reference": data_dir / "icd10_reference.csv",
    }


def summarize_files(data_dir: str | Path = "data", count_rows: bool = False) -> pd.DataFrame:
    rows = []
    for name, path in data_paths(data_dir).items():
        if not path.exists():
            continue

        header = pd.read_csv(path, nrows=0)
        item = {
            "name": name,
            "path": str(path),
            "size_mb": round(path.stat().st_size / 1024**2, 2),
            "columns": len(header.columns),
        }
        if count_rows:
            with path.open("rb") as handle:
                item["rows"] = max(sum(1 for _ in handle) - 1, 0)
        rows.append(item)

    return pd.DataFrame(rows)


def load_training_data(
    data_dir: str | Path = "data",
    include_supplemental: bool = False,
    sample_n: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    paths = data_paths(data_dir)
    train = pd.read_csv(paths["train"], low_memory=False, memory_map=True)

    if include_supplemental:
        supplemental = pd.read_csv(paths["supplemental"], low_memory=False, memory_map=True)
        train = pd.concat([train, supplemental], axis=0, ignore_index=True)

    if sample_n is not None and sample_n < len(train):
        train = train.sample(n=sample_n, random_state=random_state).sort_index()

    return train


def load_test_data(data_dir: str | Path = "data") -> pd.DataFrame:
    return pd.read_csv(data_paths(data_dir)["test"], low_memory=False, memory_map=True)


def schema_report(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "train_only": sorted(set(train.columns) - set(test.columns)),
        "test_only": sorted(set(test.columns) - set(train.columns)),
        "shared": sorted(set(train.columns) & set(test.columns)),
    }


def missing_summary(df: pd.DataFrame, min_missing_pct: float = 0.0) -> pd.DataFrame:
    out = (
        df.isna()
        .agg(["sum", "mean"])
        .T.rename(columns={"sum": "missing_count", "mean": "missing_pct"})
    )
    out["missing_pct"] = out["missing_pct"] * 100
    out = out[out["missing_pct"] > min_missing_pct]
    return out.sort_values("missing_pct", ascending=False)


def target_balance(df: pd.DataFrame, targets: Sequence[str] = TARGET_COLS) -> pd.DataFrame:
    frames = []
    for target in targets:
        if target not in df.columns:
            continue
        balance = df[target].value_counts(dropna=False).sort_index()
        frames.append(
            pd.DataFrame(
                {
                    "target": target,
                    "class": balance.index,
                    "count": balance.values,
                    "pct": balance.values / len(df) * 100,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def target_rates(
    df: pd.DataFrame,
    group_col: str,
    targets: Sequence[str] = TARGET_COLS,
    min_count: int = 500,
) -> pd.DataFrame:
    targets = [target for target in targets if target in df.columns]
    counts = df[group_col].value_counts(dropna=False).rename("n")
    rates = df.groupby(group_col, dropna=False)[targets].mean().mul(100)
    out = counts.to_frame().join(rates).sort_values("n", ascending=False)
    return out[out["n"] >= min_count].round(2)


def add_missing_indicators(
    df: pd.DataFrame,
    columns: Sequence[str] = LAB_COLS_WITH_INFORMATIVE_MISSINGNESS,
    copy: bool = True,
) -> pd.DataFrame:
    out = df.copy() if copy else df
    for col in columns:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype("int8")
    return out


def split_time_based(
    df: pd.DataFrame,
    year_col: str = "arrival_year",
    train_end_year: int = 2022,
    valid_years: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if valid_years is None:
        train_mask = df[year_col] <= train_end_year
        valid_mask = df[year_col] > train_end_year
    else:
        valid_years = set(valid_years)
        valid_mask = df[year_col].isin(valid_years)
        train_mask = ~valid_mask

    return df.loc[train_mask].copy(), df.loc[valid_mask].copy()


def make_feature_target(
    df: pd.DataFrame,
    target: str,
    leakage_cols: Sequence[str] = LEAKAGE_COLS,
    id_cols: Sequence[str] = ID_COLS,
    target_cols: Sequence[str] = TARGET_COLS,
    missing_indicator_cols: Sequence[str] = LAB_COLS_WITH_INFORMATIVE_MISSINGNESS,
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    y = df[target].astype("int8")
    X = make_feature_matrix(
        df,
        leakage_cols=leakage_cols,
        id_cols=id_cols,
        target_cols=target_cols,
        missing_indicator_cols=missing_indicator_cols,
    )
    return X, y


def make_feature_matrix(
    df: pd.DataFrame,
    leakage_cols: Sequence[str] = LEAKAGE_COLS,
    id_cols: Sequence[str] = ID_COLS,
    target_cols: Sequence[str] = TARGET_COLS,
    missing_indicator_cols: Sequence[str] = LAB_COLS_WITH_INFORMATIVE_MISSINGNESS,
) -> pd.DataFrame:
    X = add_missing_indicators(df, missing_indicator_cols, copy=True)
    drop_cols = set(id_cols) | set(leakage_cols) | set(target_cols)
    drop_cols = [col for col in drop_cols if col in X.columns]
    return X.drop(columns=drop_cols)


def separate_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_features = [col for col in X.columns if col not in categorical_features]
    return numeric_features, categorical_features


def _require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.base import BaseEstimator
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import SGDClassifier
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ModuleNotFoundError as exc:
        if exc.name == "sklearn":
            raise ModuleNotFoundError(
                "scikit-learn is required for the baseline modeling cells. "
                "Install it in this notebook kernel with: %pip install scikit-learn"
            ) from exc
        raise

    return {
        "BaseEstimator": BaseEstimator,
        "ColumnTransformer": ColumnTransformer,
        "SimpleImputer": SimpleImputer,
        "SGDClassifier": SGDClassifier,
        "accuracy_score": accuracy_score,
        "average_precision_score": average_precision_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    }


def _one_hot_encoder(min_frequency: int | float | None = 50) -> Any:
    sklearn = _require_sklearn()
    OneHotEncoder = sklearn["OneHotEncoder"]

    kwargs = {"handle_unknown": "ignore"}
    if min_frequency is not None:
        kwargs["min_frequency"] = min_frequency

    try:
        return OneHotEncoder(**kwargs, sparse_output=True)
    except TypeError:
        try:
            return OneHotEncoder(**kwargs, sparse=True)
        except TypeError:
            kwargs.pop("min_frequency", None)
            return OneHotEncoder(**kwargs, sparse=True)


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    min_category_frequency: int | float | None = 50,
) -> Any:
    sklearn = _require_sklearn()
    ColumnTransformer = sklearn["ColumnTransformer"]
    Pipeline = sklearn["Pipeline"]
    SimpleImputer = sklearn["SimpleImputer"]
    StandardScaler = sklearn["StandardScaler"]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder(min_frequency=min_category_frequency)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(numeric_features)),
            ("cat", categorical_pipe, list(categorical_features)),
        ],
        remainder="drop",
    )


def build_baseline_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    random_state: int = 42,
    min_category_frequency: int | float | None = 50,
) -> Any:
    sklearn = _require_sklearn()
    Pipeline = sklearn["Pipeline"]
    SGDClassifier = sklearn["SGDClassifier"]

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        min_category_frequency=min_category_frequency,
    )
    model = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-5,
        l1_ratio=0.05,
        class_weight="balanced",
        max_iter=1000,
        tol=1e-3,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def _positive_class_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))


def evaluate_binary_model(
    model: Any,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    threshold: float = 0.5,
) -> tuple[pd.Series, pd.DataFrame]:
    sklearn = _require_sklearn()
    roc_auc_score = sklearn["roc_auc_score"]
    average_precision_score = sklearn["average_precision_score"]
    accuracy_score = sklearn["accuracy_score"]
    precision_score = sklearn["precision_score"]
    recall_score = sklearn["recall_score"]
    f1_score = sklearn["f1_score"]
    confusion_matrix = sklearn["confusion_matrix"]

    y_score = _positive_class_scores(model, X_valid)
    y_pred = (y_score >= threshold).astype("int8")

    metrics = pd.Series(
        {
            "roc_auc": roc_auc_score(y_valid, y_score),
            "average_precision": average_precision_score(y_valid, y_score),
            "accuracy": accuracy_score(y_valid, y_pred),
            "precision": precision_score(y_valid, y_pred, zero_division=0),
            "recall": recall_score(y_valid, y_pred, zero_division=0),
            "f1": f1_score(y_valid, y_pred, zero_division=0),
            "positive_rate_pred": y_pred.mean(),
            "positive_rate_actual": y_valid.mean(),
        }
    )

    cm = pd.DataFrame(
        confusion_matrix(y_valid, y_pred),
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
    return metrics, cm


def feature_importance_from_linear_pipeline(
    model: Any,
    top_n: int = 40,
) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{i}" for i in range(estimator.coef_.shape[1])])

    coefficients = estimator.coef_.ravel()
    out = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    )
    return out.sort_values("abs_coefficient", ascending=False).head(top_n)


def run_baseline_for_target(
    df: pd.DataFrame,
    target: str,
    train_end_year: int = 2022,
    random_state: int = 42,
    min_category_frequency: int | float | None = 50,
    top_n_features: int = 40,
    leakage_cols: Sequence[str] = LEAKAGE_COLS,
    id_cols: Sequence[str] = ID_COLS,
) -> BaselineResult:
    train_df, valid_df = split_time_based(df, train_end_year=train_end_year)
    X_train, y_train = make_feature_target(
        train_df,
        target,
        leakage_cols=leakage_cols,
        id_cols=id_cols,
    )
    X_valid, y_valid = make_feature_target(
        valid_df,
        target,
        leakage_cols=leakage_cols,
        id_cols=id_cols,
    )

    numeric_features, categorical_features = separate_feature_types(X_train)
    model = build_baseline_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
        min_category_frequency=min_category_frequency,
    )
    model.fit(X_train, y_train)

    metrics, cm = evaluate_binary_model(model, X_valid, y_valid)
    importance = feature_importance_from_linear_pipeline(model, top_n=top_n_features)

    return BaselineResult(
        target=target,
        model=model,
        metrics=metrics,
        confusion_matrix=cm,
        feature_importance=importance,
        train_shape=X_train.shape,
        valid_shape=X_valid.shape,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def run_all_baselines(
    df: pd.DataFrame,
    targets: Sequence[str] = TARGET_COLS,
    train_end_year: int = 2022,
    random_state: int = 42,
    min_category_frequency: int | float | None = 50,
    leakage_cols: Sequence[str] = LEAKAGE_COLS,
    id_cols: Sequence[str] = ID_COLS,
) -> dict[str, BaselineResult]:
    return {
        target: run_baseline_for_target(
            df=df,
            target=target,
            train_end_year=train_end_year,
            random_state=random_state,
            min_category_frequency=min_category_frequency,
            leakage_cols=leakage_cols,
            id_cols=id_cols,
        )
        for target in targets
    }


def fit_full_model_for_target(
    df: pd.DataFrame,
    target: str,
    random_state: int = 42,
    min_category_frequency: int | float | None = 50,
    leakage_cols: Sequence[str] = LEAKAGE_COLS,
    id_cols: Sequence[str] = ID_COLS,
) -> Pipeline:
    X, y = make_feature_target(
        df,
        target=target,
        leakage_cols=leakage_cols,
        id_cols=id_cols,
    )
    numeric_features, categorical_features = separate_feature_types(X)
    model = build_baseline_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
        min_category_frequency=min_category_frequency,
    )
    model.fit(X, y)
    return model


def fit_full_models(
    df: pd.DataFrame,
    targets: Sequence[str] = TARGET_COLS,
    random_state: int = 42,
    min_category_frequency: int | float | None = 50,
    leakage_cols: Sequence[str] = LEAKAGE_COLS,
    id_cols: Sequence[str] = ID_COLS,
) -> dict[str, Pipeline]:
    return {
        target: fit_full_model_for_target(
            df=df,
            target=target,
            random_state=random_state,
            min_category_frequency=min_category_frequency,
            leakage_cols=leakage_cols,
            id_cols=id_cols,
        )
        for target in targets
    }


def predict_test_scores(
    models: dict[str, Pipeline],
    test_df: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:
    if not ID_COLS or ID_COLS[0] not in test_df.columns:
        raise ValueError(f"Test data must contain id column: {ID_COLS[0]}")

    X_test = make_feature_matrix(test_df)
    predictions = pd.DataFrame({ID_COLS[0]: test_df[ID_COLS[0]].values})

    for target, model in models.items():
        scores = _positive_class_scores(model, X_test)
        if threshold is None:
            predictions[target] = scores
        else:
            predictions[target] = (scores >= threshold).astype("int8")

    return predictions


def make_submission(
    models: dict[str, Pipeline],
    test_df: pd.DataFrame,
    sample_submission_path: str | Path = "data/sample_submission.csv",
    threshold: float | None = None,
) -> pd.DataFrame:
    sample_submission_path = Path(sample_submission_path)
    sample = pd.read_csv(sample_submission_path)
    id_col = ID_COLS[0]

    predictions = predict_test_scores(models, test_df, threshold=threshold)
    submission = sample[[id_col]].merge(predictions, on=id_col, how="left")

    target_cols = [col for col in sample.columns if col != id_col]
    missing_targets = [col for col in target_cols if col not in submission.columns]
    if missing_targets:
        raise ValueError(f"Missing prediction columns for sample submission: {missing_targets}")

    missing_predictions = submission[target_cols].isna().sum()
    if missing_predictions.any():
        raise ValueError(f"Missing predictions after id alignment: {missing_predictions.to_dict()}")

    return submission[sample.columns]


def save_submission(
    submission: pd.DataFrame,
    output_path: str | Path = "submission.csv",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path


def metrics_table(results: dict[str, BaselineResult]) -> pd.DataFrame:
    return pd.DataFrame({target: result.metrics for target, result in results.items()}).T
