"""
Day 4 - 1D convolutional autoencoder (Conv1D-AE) anomaly detector.

Idea (semi-supervised / "train-on-normal-only" reconstruction detector):
  The AE learns to compress-and-rebuild the NOMINAL manifold of a segment.
  At inference, anomalous segments rebuild badly -> large reconstruction error
  -> high anomaly score. Labels are used ONLY to filter the training set,
  never for supervised fitting, so the detector stays unsupervised at scoring.

Method: resample → per-channel scaler → Conv1D-AE (trained nominal-only) 
    → per-channel error standardization → high-pass score → rank-mean ensemble 
    → macro/pooled eval + length-only null → paired bootstrap CIs → MLflow  

WHAT THIS SCRIPT REPORTS, AND WHY (lessons baked in from the notebook):
  * A naive POOLED AUC over all channels is inflated by base-rate concentration
    (76% of test anomalies live in 3 easy channels) -> we report MACRO
    (per-channel-averaged) AUC as the honest headline, pooled as secondary.
  * Reconstruction error partly re-encodes SEGMENT LENGTH (interpolation
    distortion), so we set a LENGTH-ONLY baseline as the null model the AE
    must beat. On macro the AE beats it by ~+0.21 AUROC; on channel CADC0888,
    where length is anti-correlated with the label, the AE still wins by ~+0.64
    -> proof the signal is waveform shape, not duration.
  * Per-segment error is standardized PER CHANNEL (mean/sd of that channel's
    nominal-train errors), which fixes the cross-channel pooling confound.
  * The AE (sustained/structural anomalies) is fused with the Day-3 high-pass
    detector (local sharp events) via a scale-free rank-mean ensemble. Gains are
    small and NOT significant at n~111 (paired bootstrap) -> the ensemble is
    recommended on COVERAGE (it patches both failure modes), not on a metric win.

Requires: torch (CPU is plenty for L=128), scipy, scikit-learn, mlflow.

Run:  python -m ops_sat_ad.models.autoencoder (from repo root, venv active)
"""

from __future__ import annotations
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

import torch
import torch.nn as nn

# configuration ---------------------------------------------------------------------
SEGMENTS_CSV = "data/segments.csv"
COL_CHANNEL, COL_SEGMENT, COL_VALUE = "channel", "segment", "value"
COL_LABEL, COL_TIME, COL_TRAIN = "label", "timestamp", "train"

L                  = 128           # fixed segment length (power of 2 -> clean stride arithmetic)
BOTTLENECK         = 8             # deliberately narrow code, the capacity constraint is the detector
EPOCHS             = 60
BATCH_SIZE         = 64
LR                 = 1e-3
TRAIN_NOMINAL_ONLY = True          # fit AE on is_train & anomaly==0 (semi-supervised)
MIN_ANOM_MACRO     = 5             # channels with fewer test anomalies are dropped from the macro
BOOTSTRAP_B        = 2000
SEED               = 0
DEVICE             = torch.device("cpu")

# Paper Table 3, unsupervised band (benchmark context only):
PAPER_BEST_AUCPR, PAPER_BEST_AUCROC = 0.779, 0.865      # MO-GAAL
PAPER_VAE_AUCPR                     = 0.450              # closest deep-reconstruction cousin


# data -----------------------------------------------------------------------------
def to_binary(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if np.issubdtype(s.dtype, np.number):
        return (s > 0).astype(int)
    return s.astype(str).str.lower().isin(
        {"1", "true", "anomaly", "anomalous", "yes"}).astype(int)


def load_segment_arrays(path: str = SEGMENTS_CSV):
    """Return (segments, meta):
       segments : {(channel, segment): raw value array}
       meta     : DataFrame[channel, segment, n(=len), is_anomaly, is_train]."""
    seg = pd.read_csv(path)
    if COL_TIME in seg.columns:
        seg = seg.sort_values([COL_CHANNEL, COL_SEGMENT, COL_TIME])
    seg["_label"] = to_binary(seg[COL_LABEL])
    seg["_train"] = (to_binary(seg[COL_TRAIN]).astype(bool)
                     if COL_TRAIN in seg.columns else True)

    segments, meta = {}, []
    for (ch, sg), g in seg.groupby([COL_CHANNEL, COL_SEGMENT]):
        x = g[COL_VALUE].to_numpy(float)
        if len(x) < 2:
            continue
        segments[(ch, sg)] = x
        meta.append({"channel": ch, "segment": sg, "n": len(x),
                     "is_anomaly": int(g["_label"].mean() > 0.5),
                     "is_train":  bool(g["_train"].mean() > 0.5)})
    meta = pd.DataFrame(meta)
    meta["len"] = meta["n"]
    return segments, meta


def resample_to_L(x: np.ndarray, length: int = L) -> np.ndarray:
    """Linear-interpolate onto `length` points over a normalized [0,1] index.
       NOTE: re-encodes duration as reconstruction difficulty (the length confound)."""
    n = len(x)
    if n == length:
        return x.astype(np.float64)
    if n == 1:
        return np.full(length, x[0], dtype=np.float64)
    return np.interp(np.linspace(0, 1, length),
                     np.linspace(0, 1, n), x).astype(np.float64)


class ChannelScaler:
    """Per-channel z-score of the raw signal VALUES, fit ONCE on training keys only."""
    def __init__(self):
        self.stats, self.g = {}, (0.0, 1.0)

    def fit(self, segments, train_keys):
        allv = np.concatenate([segments[k] for k in train_keys])
        self.g = (float(allv.mean()), float(allv.std() + 1e-12))
        by_ch = defaultdict(list)
        for (ch, sg) in train_keys:
            by_ch[ch].append(segments[(ch, sg)])
        for ch, arrs in by_ch.items():
            v = np.concatenate(arrs)
            self.stats[ch] = (float(v.mean()), float(v.std() + 1e-12))
        return self

    def transform_one(self, x, ch):
        mu, sd = self.stats.get(ch, self.g)
        return (x - mu) / sd


def build_matrix(segments, scaler, keys) -> np.ndarray:
    """resample -> per-channel z-score -> (len(keys), L) float32."""
    return np.stack([scaler.transform_one(resample_to_L(segments[k]), k[0])
                     for k in keys]).astype(np.float32)


# model ----------------------------------------------------------------------------- 
class Conv1dAE(nn.Module):
    """Encoder 3x Conv1d/stride-2 (1->16->32->32), 128->16; narrow bottleneck;
       mirrored ConvTranspose1d decoder, 16->128. Linear output (z-scored data is signed)."""
    def __init__(self, length: int = L, bottleneck: int = BOTTLENECK):
        super().__init__()
        self.reduced = length // 8
        self.flat = 32 * self.reduced
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.enc_fc = nn.Linear(self.flat, bottleneck)
        self.dec_fc = nn.Linear(bottleneck, self.flat)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 1,  3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        z = self.encoder(x).flatten(1)
        h = self.dec_fc(self.enc_fc(z)).view(-1, 32, self.reduced)
        return self.decoder(h)


def train_ae(X, epochs=EPOCHS, bs=BATCH_SIZE, lr=LR, bottleneck=BOTTLENECK,
             seed=SEED, verbose=True) -> Conv1dAE:
    torch.manual_seed(seed)
    model = Conv1dAE(bottleneck=bottleneck).to(DEVICE)
    Xt = torch.tensor(X).unsqueeze(1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    model.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), bs):
            xb = Xt[perm[i:i+bs]].to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  epoch {ep:3d}/{epochs}  train MSE = {tot/len(Xt):.5f}")
    return model


@torch.no_grad()
def recon_error(model, X) -> np.ndarray:
    """Per-segment score = mean squared residual over the L points."""
    model.eval()
    Xt = torch.tensor(X.astype(np.float32)).unsqueeze(1).to(DEVICE)
    return ((model(Xt) - Xt) ** 2).mean(dim=(1, 2)).cpu().numpy()


# scoring helpers ----------------------------------------------------------------------------- 
def per_channel_stats(values, channels):
    """mean/sd of a score per channel (fit on nominal-train)."""
    tmp = defaultdict(list)
    for ch, v in zip(channels, values):
        tmp[ch].append(v)
    return {ch: (np.mean(v), np.std(v) + 1e-12) for ch, v in tmp.items()}


def standardize(values, channels, stats):
    return np.array([(v - stats.get(ch, (0, 1))[0]) / stats.get(ch, (0, 1))[1]
                     for ch, v in zip(channels, values)])


def hp_score(x, frac=0.125, poly=2, min_len=9):
    """Day-3 within-segment high-pass: peak |x - savgol(x)| on the RAW signal
       (native resolution -> no resampling/length confound). Catches local sharp events."""
    n = len(x)
    if n < min_len:
        return 0.0
    w = max(poly + 2, int(round(frac * n)))
    w = min(w, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w += 1
    if w < poly + 2:
        return 0.0
    return float(np.abs(x - savgol_filter(x, w, poly)).max())


# metrics ----------------------------------------------------------------------------- 
def evaluate(df, score, min_anom=MIN_ANOM_MACRO):
    """Return (pooled_aucpr, pooled_aucroc), per-channel DataFrame, macro Series."""
    d = df.assign(_s=np.asarray(score))
    pooled = (average_precision_score(d.is_anomaly, d._s),
              roc_auc_score(d.is_anomaly, d._s))
    rows = []
    for ch, sub in d.groupby("channel"):
        if sub.is_anomaly.nunique() < 2:
            continue
        rows.append((ch, len(sub), int(sub.is_anomaly.sum()), sub.is_anomaly.mean(),
                     average_precision_score(sub.is_anomaly, sub._s),
                     roc_auc_score(sub.is_anomaly, sub._s)))
    per = pd.DataFrame(rows, columns=["channel", "n", "anom", "prev", "aucpr", "aucroc"])
    macro = per.loc[per.anom >= min_anom, ["aucpr", "aucroc"]].mean()
    return pooled, per, macro


def paired_bootstrap_macro(df, cols, channels, B=BOOTSTRAP_B, seed=SEED, min_anom=3):
    """Channel-stratified bootstrap of macro AUROC for each score column, on a FIXED
       channel set (so the macro is comparable across resamples). Returns {col: array}."""
    rng = np.random.default_rng(seed)
    ch_idx = {ch: df.index[df.channel == ch].values for ch in channels}

    def macro(sample, col):
        v = []
        for ch in channels:
            sub = sample[sample.channel == ch]
            if sub.is_anomaly.nunique() == 2 and sub.is_anomaly.sum() >= min_anom:
                v.append(roc_auc_score(sub.is_anomaly, sub[col]))
        return np.mean(v) if v else np.nan

    rec = {c: [] for c in cols}
    for _ in range(B):
        idx = np.concatenate([rng.choice(ix, len(ix), True) for ix in ch_idx.values()])
        sample = df.loc[idx]
        for c in cols:
            rec[c].append(macro(sample, c))
    return {c: np.array(v) for c, v in rec.items()}


def ci(arr, lo=2.5, hi=97.5):
    a = arr[~np.isnan(arr)]
    return float(a.mean()), float(np.percentile(a, lo)), float(np.percentile(a, hi))


# pipeline ----------------------------------------------------------------------------- 
def run(train_nominal_only=TRAIN_NOMINAL_ONLY, log_mlflow=True, verbose=True):
    segments, meta = load_segment_arrays()
    if verbose:
        print(f"{len(meta)} segments | {meta.is_anomaly.sum()} anomalous "
              f"({100*meta.is_anomaly.mean():.1f}%) | {meta.is_train.sum()} train")

    key = lambda r: (r.channel, r.segment)
    fit_rows = (meta[(meta.is_train) & (meta.is_anomaly == 0)] if train_nominal_only
                else meta[meta.is_train])
    fit_keys = [key(r) for r in fit_rows.itertuples(index=False)]

    # preprocessing fit on the SAME segments the AE trains on
    scaler = ChannelScaler().fit(segments, fit_keys)
    X_fit = build_matrix(segments, scaler, fit_keys)
    if verbose:
        print(f"fitting AE on {len(fit_keys)} "
              f"{'nominal' if train_nominal_only else 'all-train'} segments")
    model = train_ae(X_fit, verbose=verbose)

    # per-channel standardization stats from nominal-train residuals (AE + high-pass)
    err_fit = recon_error(model, X_fit)
    ae_stats = per_channel_stats(err_fit, [k[0] for k in fit_keys])
    hp_fit = np.array([hp_score(segments[k]) for k in fit_keys])
    hp_stats = per_channel_stats(hp_fit, [k[0] for k in fit_keys])

    # score the test split
    test = meta[~meta.is_train].copy()
    test_keys = [key(r) for r in test.itertuples(index=False)]
    X_test = build_matrix(segments, scaler, test_keys)
    chans = test.channel.values

    test["err"]  = recon_error(model, X_test)
    test["z"]    = standardize(test["err"].values, chans, ae_stats)           # AE score
    test["hp"]   = [hp_score(segments[k]) for k in test_keys]
    test["hp_z"] = standardize(test["hp"].values, chans, hp_stats)            # Day-3 score
    # scale-free rank-mean ensemble (robust to either detector's outliers)
    for c in ("z", "hp_z"):
        test[c + "_pct"] = test.groupby("channel")[c].transform(lambda s: rankdata(s) / len(s))
    test["ens"] = 0.5 * (test["z_pct"] + test["hp_z_pct"])
    test["len_score"] = test["len"].astype(float)                            # null model

    # evaluate
    results = {}
    for name in ["len_score", "z", "hp_z", "ens"]:
        pooled, per, macro = evaluate(test, test[name].values)
        results[name] = {"pooled": pooled, "per": per, "macro": macro}

    macro_channels = list(results["z"]["per"].query("anom >= @MIN_ANOM_MACRO").channel)
    bs = paired_bootstrap_macro(test, ["z", "hp_z", "ens"], macro_channels)
    diff = bs["ens"] - bs["z"]
    ci_summary = {name: ci(bs[name]) for name in ["z", "hp_z", "ens"]}
    ci_summary["ens_minus_ae"] = ci(diff)
    ci_summary["P_ens_gt_ae"] = float((diff[~np.isnan(diff)] > 0).mean())

    if verbose:
        print("\n=== MACRO AUROC (honest headline) with 95% bootstrap CI ===")
        print(f"  length-only (null): {results['len_score']['macro'].aucroc:.3f}")
        for name, tag in [("z", "AE "), ("hp_z", "HP "), ("ens", "ENS")]:
            m, lo, hi = ci_summary[name]
            print(f"  {tag}: {m:.3f}  [{lo:.3f}, {hi:.3f}]")
        m, lo, hi = ci_summary["ens_minus_ae"]
        print(f"  paired ENS-AE: {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
              f"P(ENS>AE)={ci_summary['P_ens_gt_ae']:.2f}")
        print(f"\n  pooled AE AUROC={results['z']['pooled'][1]:.3f} "
              f"(inflated by base-rate concentration; prefer macro)")
        print(f"  paper unsup. band: AUCPR {PAPER_BEST_AUCPR} / AUCROC {PAPER_BEST_AUCROC} "
              f"(MO-GAAL); VAE {PAPER_VAE_AUCPR}")
        print("\nper-channel AUROC (AE | HP | ENS):")
        for ch in results["z"]["per"].channel:
            sub = test[test.channel == ch]
            print(f"  {ch}: {roc_auc_score(sub.is_anomaly, sub.z):.3f} | "
                  f"{roc_auc_score(sub.is_anomaly, sub.hp_z):.3f} | "
                  f"{roc_auc_score(sub.is_anomaly, sub.ens):.3f}")

    out = {"model": model, "scaler": scaler, "ae_stats": ae_stats, "hp_stats": hp_stats,
           "test": test, "results": results, "ci": ci_summary,
           "train_nominal_only": train_nominal_only}
    if log_mlflow:
        _log_mlflow(out)
    return out


def _log_mlflow(out):
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("ops-sat-ad")
        tag = "nominal" if out["train_nominal_only"] else "alltrain"
        with mlflow.start_run(run_name=f"day4-conv1d-ae-{tag}"):
            mlflow.log_params({
                "model": "conv1d_ae+highpass_ensemble", "L": L, "bottleneck": BOTTLENECK,
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                "train_nominal_only": out["train_nominal_only"],
                "scoring": "per_channel_standardized", "ensemble": "rank_mean",
            })
            r, c = out["results"], out["ci"]
            metrics = {
                "macro_aucroc_len": r["len_score"]["macro"].aucroc,
                "macro_aucpr_len":  r["len_score"]["macro"].aucpr,
                "macro_aucroc_ae":  c["z"][0],  "macro_aucroc_ae_lo": c["z"][1],  "macro_aucroc_ae_hi": c["z"][2],
                "macro_aucroc_hp":  c["hp_z"][0], "macro_aucroc_hp_lo": c["hp_z"][1], "macro_aucroc_hp_hi": c["hp_z"][2],
                "macro_aucroc_ens": c["ens"][0], "macro_aucroc_ens_lo": c["ens"][1], "macro_aucroc_ens_hi": c["ens"][2],
                "macro_aucpr_ens":  r["ens"]["macro"].aucpr,
                "ens_minus_ae":     c["ens_minus_ae"][0], "P_ens_gt_ae": c["P_ens_gt_ae"],
                "pooled_aucroc_ae": r["z"]["pooled"][1],
            }
            for ch, a, ar in zip(r["z"]["per"].channel, r["z"]["per"].aucroc, r["z"]["per"].aucpr):
                metrics[f"aucroc_{ch}"] = a
            mlflow.log_metrics(metrics)
            torch.save(out["model"].state_dict(), "conv1d_ae.pt")
            mlflow.log_artifact("conv1d_ae.pt")
        print("\nMLflow logged. UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    except Exception as e:
        print("MLflow logging skipped:", e)


# entrypoint ----------------------------------------------------------------------------- 
if __name__ == "__main__":
    print("### Day 4 - Conv1D-AE + high-pass ensemble ###\n")
    run(train_nominal_only=TRAIN_NOMINAL_ONLY, log_mlflow=True)
