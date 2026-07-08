import pandas as pd
import mlflow
import lightgbm as lgb

from ops_sat_ad.evaluate import evaluate


def load_dataset(path="data/dataset.csv"):
    df = pd.read_csv(path)
    print("Columns:", list(df.columns))  # confirm names before trusting anything below
    return df


def run_metadata_baseline(df, feature_cols=("len", "duration", "sampling")):
    feature_cols = list(feature_cols)
    train = df[df["train"] == 1]
    test = df[df["train"] == 0]

    X_train, y_train = train[feature_cols], train["anomaly"]
    X_test, y_test = test[feature_cols], test["anomaly"]

    mlflow.set_experiment("ops-sat-anomaly-detection")
    with mlflow.start_run(run_name="day2_leakage_baseline") as run:
        print("Tracking URI:", mlflow.get_tracking_uri())
        print("Artifact URI:", run.info.artifact_uri)
        mlflow.set_tags({
            "model_family": "gbm",
            "feature_set": "metadata_only",
            "cv_scheme": "fixed_split",
        })
        mlflow.log_param("features", feature_cols)

        model = lgb.LGBMClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = (y_score > 0.5).astype(int)  # placeholder — real threshold comes in Step 4

        results = evaluate(y_test.values, y_pred, y_score)
        mlflow.log_metrics(results)

    return results


if __name__ == "__main__":
    df = load_dataset()
    results = run_metadata_baseline(df)
    print(results)