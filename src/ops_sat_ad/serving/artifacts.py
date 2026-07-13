from __future__ import annotations
from collections import defaultdict

import numpy as np
import torch
import os, joblib

from ops_sat_ad.models.autoencoder import (
    BOTTLENECK, Conv1dAE, ChannelScaler,
    load_segment_arrays, build_matrix, recon_error, hp_score,
)
from ops_sat_ad.serving.predict import Bundle, predict_segment

WEIGHTS_PATH = "conv1d_ae.pt"


def _grouped_sorted(values, channels) -> dict[str, np.ndarray]:
    """channel -> sorted array of that channel's scores (the frozen ECDF)."""
    by_ch = defaultdict(list)
    for ch, v in zip(channels, values):
        by_ch[ch].append(v)
    return {ch: np.sort(np.asarray(vs, dtype=float)) for ch, vs in by_ch.items()}


def build_bundle(weights_path=WEIGHTS_PATH, target_recall=0.80,
                 version="day5-ens-v1") -> Bundle:
    segments, meta = load_segment_arrays()
    key = lambda r: (r.channel, r.segment)

    # 1. the SAME fit set the AE was trained on: nominal-train segments
    fit_rows = meta[(meta.is_train) & (meta.is_anomaly == 0)]
    fit_keys = [key(r) for r in fit_rows.itertuples(index=False)]
    fit_chans = [k[0] for k in fit_keys]

    # 2. reproduce the training scaler (deterministic from data + keys)
    scaler = ChannelScaler().fit(segments, fit_keys)

    # 3. load frozen AE weights -- no retraining
    model = Conv1dAE(bottleneck=BOTTLENECK)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # 4. frozen per-channel ECDFs from nominal-train scores
    X_fit = build_matrix(segments, scaler, fit_keys)
    ae_ecdf = _grouped_sorted(recon_error(model, X_fit), fit_chans)
    hp_ecdf = _grouped_sorted([hp_score(segments[k]) for k in fit_keys], fit_chans)

    # 5. provisional bundle (threshold filled in next)
    bundle = Bundle(model, scaler, ae_ecdf, hp_ecdf,
                    threshold=1.0, target_recall=target_recall, version=version)

    # 6. threshold @ target recall, scored through predict_segment on TRAIN only
    train_rows = meta[meta.is_train]
    scores = np.array([predict_segment(bundle, r.channel, segments[key(r)])["score"]
                       for r in train_rows.itertuples(index=False)])
    y = train_rows.is_anomaly.to_numpy()
    pos = scores[y == 1]
    bundle.threshold = float(np.quantile(pos, 1.0 - target_recall))

    # 7. report the operating point achieved on train
    flagged = scores >= bundle.threshold
    recall = flagged[y == 1].mean()
    precision = y[flagged].mean() if flagged.any() else float("nan")
    print(f"threshold={bundle.threshold:.4f} | train recall={recall:.3f} "
          f"precision={precision:.3f} | flagged {flagged.mean()*100:.1f}% of train")
    return bundle


def save_bundle(bundle: Bundle, out_dir="models/serving_bundle"):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(bundle.model.state_dict(), os.path.join(out_dir, "conv1d_ae.pt"))
    joblib.dump(
        {"scaler": bundle.scaler, "ae_ecdf": bundle.ae_ecdf, "hp_ecdf": bundle.hp_ecdf,
         "threshold": bundle.threshold, "target_recall": bundle.target_recall,
         "version": bundle.version, "bottleneck": BOTTLENECK},
        os.path.join(out_dir, "refs.joblib"),
    )
    print(f"bundle saved -> {out_dir}")


if __name__ == "__main__":
    save_bundle(build_bundle())