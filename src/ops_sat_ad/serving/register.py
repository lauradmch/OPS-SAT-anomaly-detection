from __future__ import annotations

import mlflow
from mlflow.pyfunc import PythonModel

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT   = "ops-sat-ad"
MODEL_NAME   = "ops-sat-anomaly-detector"


class EnsembleDetector(PythonModel):
    """MLflow pyfunc wrapper: load the frozen bundle once, score segments."""

    def load_context(self, context):
        from ops_sat_ad.serving.predict import load_bundle
        self.bundle = load_bundle(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        from ops_sat_ad.serving.predict import predict_segment
        records = (model_input.to_dict("records")
                   if hasattr(model_input, "to_dict") else model_input)
        return [predict_segment(self.bundle, r["channel"], r["values"])
                for r in records]


def register(bundle_dir="models/serving_bundle", alias="production"):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="day5-register-ensemble"):
        info = mlflow.pyfunc.log_model(
            name="ensemble_detector",                 # MLflow 3.x: `name`, not `artifact_path`
            python_model=EnsembleDetector(),
            artifacts={"bundle": bundle_dir},          # the whole bundle dir travels with the model
            registered_model_name=MODEL_NAME,          # auto-registers a new version
        )
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, alias, info.registered_model_version)
    print(f"registered {MODEL_NAME} v{info.registered_model_version} -> @{alias}")
    return info

# export the model at build time
def export(dst="model_artifact", alias="production"):
    import os
    mlflow.set_tracking_uri(TRACKING_URI)
    os.makedirs(dst, exist_ok=True)
    path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}@{alias}", dst_path=dst)
    print("exported ->", path)
    return path

if __name__ == "__main__":
    register()