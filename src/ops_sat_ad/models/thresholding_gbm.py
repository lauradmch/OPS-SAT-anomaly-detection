"""Threshold selection (target recall) + per-campaign/per-channel breakdown."""

import numpy as np
import pandas as pd
import mlflow
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import recall_score

from ops_sat_ad.evaluate import evaluate
from ops_sat_ad.models.baseline import load_dataset
from ops_sat_ad.models.gbm_model import FEATURE_COLS

TARGET_RECALL = 0.90
EXCLUDE_CHANNELS = {"CADC0884", "CADC0886", "CADC0890"}  # D5: too few test anomalies


def add_campaign(df, segments_path="data/segments.csv"):
    seg = pd.read_csv(segments_path, parse_dates=["timestamp"])
    first_ts = seg.groupby("segment")["timestamp"].min().rename("first_ts") # earliest timestamp per segment
    return df.merge(first_ts, on="segment", how="left").assign(
        campaign=lambda d: d["first_ts"].dt.to_period("M").astype(str) # campaign = YYYY-MM of first timestamp in segment
    )


def threshold_for_recall(y_true, y_score, target_recall):
    for t in np.sort(np.unique(y_score))[::-1]:
        if recall_score(y_true, (y_score >= t).astype(int)) >= target_recall:
            return t
    return y_score.min() 


def run_thresholding(df, feature_cols=FEATURE_COLS, target_recall=TARGET_RECALL):
    train, test = df[df["train"]==1], df[df["train"]==0]
    X_train, y_train, groups = train[feature_cols], train["anomaly"], train["channel"]
    X_test, y_test = test[feature_cols], test["anomaly"]

    # cross-validation on train only -> threshold chosen without touching test
    gkf = GroupKFold(n_splits=5)
    oof_score = np.zeros(len(train))
    for tr_idx, val_idx in gkf.split(X_train, y_train, groups):
        m = lgb.LGBMClassifier(random_state=42)
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_score[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]

    threshold = threshold_for_recall(y_train.values, oof_score, target_recall)
    print(f"Frozen threshold for target recall {target_recall}: {threshold:.4f}")

    # new model trained on full train set, threshold applied to test set
    final_model = lgb.LGBMClassifier(random_state=42).fit(X_train, y_train)
    y_score_test = final_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_score_test >= threshold).astype(int)

    overall = evaluate(y_test.values, y_pred_test, y_score_test)

    mlflow.set_experiment("ops-sat-anomaly-detection")
    with mlflow.start_run(run_name="day2_gbm_thresholded"):
        mlflow.set_tags({"model_family": "gbm", "feature_set": "paper_18", "cv_scheme": "groupkfold_channel"})
        mlflow.log_params({"target_recall": target_recall, "threshold": threshold})
        mlflow.log_metrics(overall)
        mlflow.lightgbm.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="ops-sat-anomaly-detector",
        )

    return final_model, threshold, overall, test.assign(y_score=y_score_test, y_pred=y_pred_test)


def breakdown(test_scored, by="campaign", exclude_channels=EXCLUDE_CHANNELS):
    d = test_scored if by != "channel" else test_scored[~test_scored["channel"].isin(exclude_channels)]
    rows = []
    for key, g in d.groupby(by):
        rows.append({
            by: key, "n": len(g), "n_anomalous": g["anomaly"].sum(),
            "recall": recall_score(g["anomaly"], g["y_pred"]) if g["anomaly"].sum() > 0 else np.nan,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = add_campaign(load_dataset())
    model, threshold, overall, test_scored = run_thresholding(df)
    print("\nOverall test metrics:", overall)
    print("\nPer-campaign recall:\n", breakdown(test_scored, "campaign"))
    print("\nPer-channel recall (excl. near-empty):\n", breakdown(test_scored, "channel"))