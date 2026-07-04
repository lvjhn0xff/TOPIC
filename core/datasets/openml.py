from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np


def clean_and_impute_column(df, col_name, threshold=0.80):
    s = df[col_name].copy()

    s = s.replace(r"^\s*$", np.nan, regex=True)
    numeric = pd.to_numeric(s, errors="coerce")

    if numeric.notna().mean() >= threshold:
        df[col_name] = numeric.fillna(numeric.median())
    else:
        df[col_name] = s

    return df


def load_openml(name, version=1):
    dataset = fetch_openml(name, version=version, as_frame=True)

    # Dataset
    X = dataset.data.copy()
    y = dataset.target

    # Feature types
    numeric_columns = []
    categorical_columns = []

    for col in X.columns:
        X = clean_and_impute_column(X, col)

        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)

    # Preprocessor factory
    def make_preprocessor():
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        return ColumnTransformer([
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ])

    # Details
    details = {
        "feature_types": {
            "numeric": numeric_columns,
            "categorical": categorical_columns,
        },
        "unique_counts": {
            col: int(X[col].nunique(dropna=True))
            for col in X.columns
        },
        "labels": list(pd.Series(y).unique()),
        "label_frequencies": (
            pd.Series(y)
            .value_counts(dropna=False)
            .to_dict()
        ),
    }

    return X, y, make_preprocessor, details


if __name__ == "__main__":
    X, y, make_preprocessor, details = load_openml("telco-customer-churn")
    preprocessor = make_preprocessor()
    print(details)