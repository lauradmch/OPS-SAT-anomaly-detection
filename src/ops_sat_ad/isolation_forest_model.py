"""Isolation Forest: unsupervised detector for comparison."""

import pandas as pd
import mlflow
from sklearn.ensemble import IsolationForest

from ops_sat_ad.evaluate import evaluate
from ops_sat_ad.baseline import load_dataset
from ops_sat_ad.gbm_model import FEATURE_COLS


def run_isolation_forest(df, feature_cols=FEATURE_COLS, contamination=0.2):
    train = df[df["train"]==1]
    test = df[df["train"]==0]

    X_train = train[feature_cols]
    X_test, y_test = test[feature_cols], test["anomaly"]

    mlflow.set_experiment("ops-sat-anomaly-detection")
    with mlflow.start_run(run_name="day2_isolation_forest"):
        mlflow.set_tags({
            "model_family": "isolation_forest",
            "feature_set": "full_18",
            "cv_scheme": "fixed_split",
            "training_regime": "unsupervised",
        })
        mlflow.log_param("features", feature_cols)
        mlflow.log_param("contamination", contamination)

        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X_train)  # no y_train —> unsupervised fit

        y_score = -model.decision_function(X_test)  # flip sign: higher = more anomalous
        y_pred = (model.predict(X_test) == -1).astype(int)  # sklearn: -1 = outlier

        results = evaluate(y_test.values, y_pred, y_score)
        mlflow.log_metrics(results)

    return results


if __name__ == "__main__":
    df = load_dataset()
    results = run_isolation_forest(df)
    print(results)