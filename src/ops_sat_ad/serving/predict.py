from __future__ import annotations
import numpy as np
import torch
import os, joblib


def ecdf_percentile(x: float, ref_sorted: np.ndarray) -> float:
    """Fraction of frozen reference scores <= x.

    Inductive replacement for the offline `rankdata(s)/len(s)`: instead of
    ranking a score against its batch, we rank ONE score against a fixed
    per-channel reference distribution (the ECDF) computed on nominal-train.
    """
    if len(ref_sorted) == 0:
        return 0.5                       # unknown channel -> neutral score
    rank = np.searchsorted(ref_sorted, x, side="right")
    return rank / len(ref_sorted)


from dataclasses import dataclass
from ops_sat_ad.models.autoencoder import (
    Conv1dAE, ChannelScaler, resample_to_L, recon_error, hp_score,
)

@dataclass
class Bundle:
    model: Conv1dAE                    # AE with frozen weights
    scaler: ChannelScaler             # per-channel raw-value z-score, fit on train
    ae_ecdf: dict[str, np.ndarray]    # channel -> sorted nominal-train AE errors
    hp_ecdf: dict[str, np.ndarray]    # channel -> sorted nominal-train HP scores
    threshold: float                  # ensemble cutoff @ target recall (train)
    target_recall: float
    version: str = "unknown"

def predict_segment(bundle: Bundle, channel: str, values) -> dict:
    x = np.asarray(values, dtype=float)

    # AE path: resample -> per-channel z-score -> reconstruction error
    xz = bundle.scaler.transform_one(resample_to_L(x), channel)
    err = float(recon_error(bundle.model, xz[None, :])[0])

    # HP path: high-pass on the RAW signal (native resolution, no length confound)
    hp = hp_score(x)

    # inductive fusion: each raw score -> percentile vs its frozen per-channel ECDF
    ae_pct = ecdf_percentile(err, bundle.ae_ecdf.get(channel, np.array([])))
    hp_pct = ecdf_percentile(hp,  bundle.hp_ecdf.get(channel, np.array([])))
    score = 0.5 * (ae_pct + hp_pct)

    return {
        "channel": channel,
        "score": float(score),
        "is_anomaly": bool(score >= bundle.threshold),
        "threshold": bundle.threshold,
        "ae_pct": float(ae_pct),
        "hp_pct": float(hp_pct),
        "n_points": int(x.size),
        "model_version": bundle.version,
    }


def load_bundle(bundle_dir="models/serving_bundle") -> Bundle:
    refs = joblib.load(os.path.join(bundle_dir, "refs.joblib"))
    model = Conv1dAE(bottleneck=refs["bottleneck"])
    model.load_state_dict(torch.load(os.path.join(bundle_dir, "conv1d_ae.pt"),
                                     map_location="cpu"))
    model.eval()
    return Bundle(model, refs["scaler"], refs["ae_ecdf"], refs["hp_ecdf"],
                  refs["threshold"], refs["target_recall"], refs["version"])