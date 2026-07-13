import numpy as np

from ops_sat_ad.models.autoencoder import Conv1dAE
from ops_sat_ad.serving.predict import Bundle, ecdf_percentile, predict_segment

# Validates the empirical CDF percentile helper function (ecdf_percentile) used for scoring anomalies
def test_ecdf_percentile_bounds():
    ref = np.sort(np.random.default_rng(0).normal(size=1000))
    assert abs(ecdf_percentile(float(ref.mean()), ref) - 0.5) < 0.05
    assert ecdf_percentile(float(ref.max()) + 1, ref) == 1.0
    assert ecdf_percentile(float(ref.min()) - 1, ref) < 0.01
    assert ecdf_percentile(0.0, np.array([])) == 0.5          # empty-ref guard

# Mock scaler
class _IdScaler:                                              # stand-in for ChannelScaler
    def transform_one(self, x, ch):
        return x

# Tiny Conv1dAE model
def _tiny_bundle():
    ecdf = {"CH": np.linspace(0.0, 1.0, 100)}
    return Bundle(model=Conv1dAE(bottleneck=8).eval(), scaler=_IdScaler(),
                  ae_ecdf=ecdf, hp_ecdf=ecdf, threshold=0.5,
                  target_recall=0.8, version="test")

# Verifies the output contract of predict_segment() (the core inference function)
def test_predict_segment_contract():
    out = predict_segment(_tiny_bundle(), "CH", np.sin(np.linspace(0, 6, 60)))
    expected = {"channel", "score", "is_anomaly", "threshold",
                "ae_pct", "hp_pct", "n_points", "model_version"}
    assert expected <= set(out)
    assert 0.0 <= out["score"] <= 1.0
    assert isinstance(out["is_anomaly"], bool)
    assert isinstance(out["score"], float)                   # JSON-safe (np.float64 caught)
    assert out["n_points"] == 60