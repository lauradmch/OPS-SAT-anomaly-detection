"""Within-Segment Smoothness-Residual Detector"""

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score

SEGMENTS_CSV = "data/segments.csv"
COL_CHANNEL, COL_SEGMENT, COL_VALUE = "channel", "segment", "value"
COL_LABEL, COL_TIME, COL_TRAIN = "label", "timestamp", "train"

SAVGOL_FRAC = 0.125   # smoothing window as a fraction of segment length (1/8). Sets the low-pass cutoff:
                       # the window must be wide enough to smooth the normal wave but narrow enough to
                       # PASS sharp anomalies. Main tuning knob -- widen if normal fast waves leak through,
                       # narrow if broad anomalies are missed. Validated at 1/8 across irregular periods.
POLYORDER   = 2        # local polynomial degree for Savitzky-Golay
MIN_LEN     = 9        # segments shorter than this can't be assessed for smoothness

# Split segments into train/test, and convert labels to binary. Segments are grouped by (channel, segment).
seg = pd.read_csv(SEGMENTS_CSV)
if COL_TIME in seg.columns:
    seg = seg.sort_values([COL_CHANNEL, COL_SEGMENT, COL_TIME])
print("segments.csv:", seg.shape, "| columns:", list(seg.columns))

def to_binary(s):
    if s.dtype == bool: return s.astype(int)
    if np.issubdtype(s.dtype, np.number): return (s > 0).astype(int)
    return s.astype(str).str.lower().isin({"1","true","anomaly","anomalous","yes"}).astype(int)

seg["_label"] = to_binary(seg[COL_LABEL])
seg["_train"] = to_binary(seg[COL_TRAIN]).astype(bool) if COL_TRAIN in seg.columns else True

segments, meta = {}, []
for (ch, sg), g in seg.groupby([COL_CHANNEL, COL_SEGMENT]):
    x = g[COL_VALUE].to_numpy(float)
    if len(x) < 2: continue
    segments[(ch, sg)] = x
    meta.append({"channel": ch, "segment": sg, "n": len(x),
                 "is_anomaly": int(g["_label"].mean() > 0.5),
                 "is_train": bool(g["_train"].mean() > 0.5)})
meta = pd.DataFrame(meta)
print(f"{len(meta)} segments | {meta.is_anomaly.sum()} anomalous "
      f"({100*meta.is_anomaly.mean():.1f}%) | {meta.is_train.sum()} train")

# Detector: compute residuals from smoothed signal, then score segments by residual magnitude.
def highpass(x, frac, polyorder, min_len):
    """Per-point smoothness residual |x - smooth(x)|. Smooth = Savitzky-Golay (local polynomial),
    window = frac*len (odd). This is a high-pass filter: ~0 for a smooth signal, large exactly where
    local smoothness breaks (jump/spike/frequency-burst). No cross-segment template involved."""
    n = len(x)
    if n < min_len: return np.zeros(n)
    w = max(polyorder + 2, int(round(frac * n)))
    w = min(w, n if n % 2 == 1 else n - 1)      # window <= n, odd
    if w % 2 == 0: w += 1
    if w > n: w = n if n % 2 == 1 else n - 1
    if w < polyorder + 2: return np.zeros(n)
    return np.abs(x - savgol_filter(x, w, polyorder))

class SmoothnessResidualDetector:
    """Within-segment high-pass residual detector. Template-free -> immune to irregular periodicity.
    score() -> (combined, peak_z, regional_z, len_z, residual_profile)."""
    def __init__(self, frac=0.125, polyorder=2, min_len=9):
        self.frac, self.polyorder, self.min_len = frac, polyorder, min_len
        self.stats = {}     # channel -> dict of calibration constants
    def _resid(self, x):
        return highpass(x, self.frac, self.polyorder, self.min_len)
    def fit(self, seg_dict):                       # seg_dict: {channel: [raw arrays]} (training only)
        for ch, arrs in seg_dict.items():
            arrs = [a for a in arrs if len(a) >= self.min_len]
            if not arrs: continue
            pooled = np.concatenate([self._resid(a) for a in arrs])
            scale = np.median(pooled) * 1.4826 + 1e-9        # channel's normal high-freq jitter (MAD-around-0)
            peak = np.array([self._resid(a).max()/scale for a in arrs])
            rms  = np.array([np.sqrt(np.mean((self._resid(a)/scale)**2)) for a in arrs])
            lens = np.array([len(a) for a in arrs], float)
            self.stats[ch] = dict(
                scale=scale,
                pk_med=np.median(peak), pk_sc=np.percentile(peak,99)-np.median(peak)+1e-9,
                rm_med=np.median(rms),  rm_sc=np.percentile(rms,99)-np.median(rms)+1e-9,
                len_med=np.median(lens), len_mad=np.median(np.abs(lens-np.median(lens)))*1.4826+1e-9)
        return self
    def score(self, x, ch):
        if ch not in self.stats or len(x) < self.min_len:
            return np.nan, np.nan, np.nan, np.nan, np.zeros(len(x))
        s = self.stats[ch]
        r = self._resid(x) / s["scale"]
        peak_z = (r.max() - s["pk_med"]) / s["pk_sc"]
        reg_z  = (np.sqrt(np.mean(r**2)) - s["rm_med"]) / s["rm_sc"]
        len_z  = abs(len(x) - s["len_med"]) / s["len_mad"]
        return float(max(peak_z, reg_z)), float(peak_z), float(reg_z), float(len_z), r

# Fit on training split
train_by_ch = defaultdict(list)
for r in meta[meta.is_train].itertuples(index=False):
    train_by_ch[r.channel].append(segments[(r.channel, r.segment)])

det = SmoothnessResidualDetector(frac=SAVGOL_FRAC, polyorder=POLYORDER, min_len=MIN_LEN).fit(train_by_ch)
print("calibrated channels:", sorted(det.stats))

# Score all segments and evaluate
rows = []
for r in meta.itertuples(index=False):
    c, pz, rz, lz, _ = det.score(segments[(r.channel, r.segment)], r.channel)
    rows.append({"channel": r.channel, "segment": r.segment, "is_anomaly": r.is_anomaly,
                 "is_train": r.is_train, "combined": c, "peak_z": pz, "reg_z": rz, "len_z": lz})
scores = pd.DataFrame(rows).dropna(subset=["combined"])

test = scores[~scores.is_train]
def m(y, s): return average_precision_score(y, s), roc_auc_score(y, s)
for col in ["combined", "peak_z", "reg_z", "len_z"]:
    ap, roc = m(test.is_anomaly, test[col].fillna(0))
    print(f"OVERALL (test)  AUCPR={ap:.3f}  AUCROC={roc:.3f}   [{col}]")

print("\nper-channel (combined):")
for ch, g in test.groupby("channel"):
    if g.is_anomaly.nunique() < 2:
        print(f"  {ch}: single-class in test, skipped"); continue
    ap, roc = m(g.is_anomaly, g.combined)
    print(f"  {ch}: AUCPR={ap:.3f}  AUCROC={roc:.3f}  (n={len(g)}, anom={int(g.is_anomaly.sum())})")

try:
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ops-sat-ad")
    with mlflow.start_run(run_name="day3-smoothness-residual"):
        mlflow.log_params({"savgol_frac": SAVGOL_FRAC, "polyorder": POLYORDER, "min_len": MIN_LEN,
                           "detector": "within_segment_highpass", "label_free": True})
        ap_c, roc_c = m(test.is_anomaly, test.combined)
        ap_p, _ = m(test.is_anomaly, test.peak_z.fillna(0))
        ap_r, _ = m(test.is_anomaly, test.reg_z.fillna(0))

        # Overall test metrics for all columns
        for col in ["combined", "peak_z", "reg_z", "len_z"]:
            ap, roc = m(test.is_anomaly, test[col].fillna(0))
            mlflow.log_metrics({f"test_aucpr_{col}": ap, f"test_aucroc_{col}": roc})

        # Per-channel test metrics for combined
        channel_metrics = {}
        for ch, g in test.groupby("channel"):
            if g.is_anomaly.nunique() < 2: continue
            ap, roc = m(g.is_anomaly, g.combined.fillna(0))
            channel_metrics.update({
                f"test_aucpr_{ch}": ap,
                f"test_aucroc_{ch}": roc,
                f"n_test_{ch}": len(g),
                f"n_anom_{ch}": int(g.is_anomaly.sum())
            })
        mlflow.log_metrics(channel_metrics)
        
    print("logged. UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")
except Exception as e:
    print("MLflow logging skipped:", e)