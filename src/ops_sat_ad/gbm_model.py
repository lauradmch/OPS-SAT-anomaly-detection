"""Full 18-feature model."""

import pandas as pd
import mlflow
import lightgbm as lgb

from ops_sat_ad.evaluate import evaluate
from ops_sat_ad.baseline import load_dataset

FEATURE_COLS = [
    "mean", "var", "std", "kurtosis", "skew", "n_peaks",
    "smooth10_n_peaks", "smooth20_n_peaks", "diff_peaks", "diff2_peaks",
    "diff_var", "diff2_var", "gaps_squared", "len_weighted",
    "var_div_duration", "var_div_len", "len", "duration", #,"sampling",
]


def run_full_model(df, feature_cols=FEATURE_COLS):
    train = df[df["train"]==1]
    test = df[df["train"]==0]

    X_train, y_train = train[feature_cols], train["anomaly"]
    X_test, y_test = test[feature_cols], test["anomaly"]

    mlflow.set_experiment("ops-sat-anomaly-detection")
    with mlflow.start_run(run_name="day2_full_features"):
        mlflow.set_tags({
            "model_family": "gbm",
            "feature_set": "full_18",
            "cv_scheme": "fixed_split",
        })
        mlflow.log_param("features", feature_cols)

        model = lgb.LGBMClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = (y_score > 0.5).astype(int)  # 0.5 is a threshold placeholder

        results = evaluate(y_test.values, y_pred, y_score)
        mlflow.log_metrics(results)

        # log importances as an artifact
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\nFeature importances:\n", importances)
        importances.to_csv("feature_importances_full.csv")
        mlflow.log_artifact("feature_importances_full.csv")

    return results, importances


if __name__ == "__main__":
    df = load_dataset()
    results, importances = run_full_model(df)
    print(results)