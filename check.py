import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
import warnings

warnings.filterwarnings("ignore")


# 1. Custom Transformer for Dual-Valued Complementation (Mean & Variance)
class DualValuedComplementation(BaseEstimator, TransformerMixin):
    """Computes row-wise mean and variance across numeric features

    and appends them as separate values.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure we are working with a floating-point numpy array
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape[1] <= 1:
            return X_arr  # Can't compute variance across 1 feature

        row_means = np.mean(X_arr, axis=1, keepdims=True)
        row_vars = np.var(X_arr, axis=1, keepdims=True)

        # Horizontally stack the original features with their dual complements
        return np.hstack([X_arr, row_means, row_vars])


# 2. Helper to fetch and prepare OpenML data cleanly
def get_openml_data(name, version=1):
    print(f"Fetching {name}...")
    dataset = fetch_openml(name, version=version, as_frame=True, parser="auto")
    X = dataset.data.copy()
    y = dataset.target

    # Cast categorical columns to object to avoid scikit-learn validation errors
    numeric_cols = []
    categorical_cols = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].astype(object).fillna("Missing")
            categorical_cols.append(col)

    # Encode labels cleanly to integers
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.factorize(y)[0]
    else:
        y = y.fillna(y.mode()[0]).to_numpy(dtype=np.int64)

    return X, y, numeric_cols, categorical_cols


# 3. Execution Pipeline
def run_investigation():
    # Selection of OpenML datasets (mix of linear/non-linear boundaries)
    datasets = {
        "Blood Transfusion (Linear-ish)": ("blood-transfusion-service-center", 1),
        "Phoneme (Non-Linear)": ("phoneme", 1),
        "Diabetes (Mixed)": ("diabetes", 1),
    }

    # Classifiers to evaluate
    classifiers = {
        "LR": LogisticRegression(max_iter=1000, random_state=42),
        "RFC": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(max_iter=500, random_state=42),
        "SVM": SVC(random_state=42),
    }

    results = []

    for ds_name, (openml_id, ver) in datasets.items():
        try:
            X, y, num_cols, cat_cols = get_openml_data(openml_id, ver)
        except Exception as e:
            print(f"Failed to load {ds_name}: {e}")
            continue

        # Basic preprocessing layout
        base_preprocessor = ColumnTransformer(
            [
                ("num", StandardScaler(), num_cols),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    cat_cols,
                ),
            ]
        )

        cv = KFold(n_splits=10, shuffle=True, random_state=42)

        for clf_name, clf in classifiers.items():
            # Configuration A: Baseline Pipeline (No Complementation)
            baseline_pipe = Pipeline(
                [("preprocessor", base_preprocessor), ("classifier", clf)]
            )

            # Configuration B: Complementation Pipeline (Appends Row Mean & Var)
            complementation_pipe = Pipeline(
                [
                    ("preprocessor", base_preprocessor),
                    ("complementation", DualValuedComplementation()),
                    ("classifier", clf),
                ]
            )

            # Evaluate configurations
            score_base = cross_validate(
                baseline_pipe, X, y, cv=cv, scoring="accuracy"
            )
            score_comp = cross_validate(
                complementation_pipe, X, y, cv=cv, scoring="accuracy"
            )

            results.append(
                {
                    "Dataset": ds_name,
                    "Classifier": clf_name,
                    "Baseline Accuracy": np.mean(score_base["test_score"]),
                    "Complement Accuracy": np.mean(score_comp["test_score"]),
                    "Delta": np.mean(score_comp["test_score"])
                    - np.mean(score_base["test_score"]),
                }
            )

    # 4. Format and Display Final Results Table
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("PRELIMINARY INVESTIGATION RESULTS (10-FOLD CV ACCURACY)")
    print("=" * 80)
    print(
        df_results.to_string(
            index=False,
            formatters={
                "Baseline Accuracy": "{:.4f}".format,
                "Complement Accuracy": "{:.4f}".format,
                "Delta": "{:+.4f}".format,
            },
        )
    )


if __name__ == "__main__":
    run_investigation()