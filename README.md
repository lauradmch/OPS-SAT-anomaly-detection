# OPS-SAT telemetry anomaly detection

Anomaly detection on segmented spacecraft telemetry from the ESA **OPS-SAT**
benchmark (Kotowski et al., *Scientific Data* 2025, `s41597-025-05035-3`). Each
telemetry channel is cut into variable-length **segments**; the task is to flag
anomalous segments per channel.

The deliverable is a small deployable detector: a semi-supervised
Conv1D autoencoder with a template-free high-pass detector, served behind a
FastAPI endpoint with a grounded natural-language report layer. Results are
reported **macro** (per-channel-averaged), not pooled, and every headline number
is confound-controlled.

## Headline result (macro, confound-controlled)

A Conv1D autoencoder (narrow 8-D bottleneck, trained on nominal segments only)
scores per-segment reconstruction error, standardized **per channel**. Reported
as **macro** (per-channel-averaged) AUROC; the pooled number is inflated by
base-rate concentration (76% of test anomalies live in 3 easy channels).

| detector | macro AUROC (95% CI) | macro AUCPR |
|---|---|---|
| length-only (null model) | 0.614 | 0.386 |
| Conv1D-AE | 0.833 [0.754, 0.928] | 0.727 |
| high-pass | 0.841 [0.762, 0.924] | 0.754 |
| **AE + high-pass ensemble** | **0.851 [0.767, 0.939]** | **0.770** |

The AE beats the length-only null by **+0.21 AUROC**, and by **+0.64** on the one
channel where length is anti-correlated with the label (proof the signal is
waveform shape, not duration). The ensemble's edge over the AE is consistent but
not significant at n≈111 (paired bootstrap: +0.018, `P(ENS>AE)=0.93`). Paper
unsupervised band for context: AUCPR 0.779 / AUCROC 0.865 (MO-GAAL); VAE 0.450.

## Methodology

### 1. Data exploration

Characterize the channel/segment structure, anomaly prevalence (~20%), and the
per-channel scale and length distributions. Key finding that shapes the rest
of the work: anomalies are **concentrated** (76% of test anomalies live in 3 of
9 channels) which is why any single pooled metric is misleading and everything
downstream is reported per channel.

### 2. Feature-based baselines (and a leakage fix)

Per-segment tabular features (`data/dataset.csv`) with Gradient Boosting Machine
(GBM) and Isolation Forest baselines. A **metadata-leakage** bug was identified and
fixed: the segment's duration/length are an artifact from the human annotation 
process itself. These baselines established that duration/length features become
predictive.

### 3. Within-segment high-pass detector

A template-free **Savitzky–Golay** smoothness-residual detector: score each
segment by the peak of `|x − smooth(x)|` on the *raw* signal at native
resolution. This flags local sharp events (spikes, dropouts) and is immune to
irregular periodicity because it never assumes a template. Channel-level
calibration is learned once on the training split. It needs no training and no
GPU.

### 4. Conv1D autoencoder + ensemble

Each variable-length segment is resampled to fixed length `L=128` by linear
interpolation, per-channel z-scored using training statistics only, and passed
through a Conv1D autoencoder (encoder `1→16→32→32`, `128→16` timesteps; a
deliberately narrow **8-D bottleneck**; mirrored `ConvTranspose1D` decoder;
pointwise MSE loss). It is trained on **nominal training segments only**, labels
are used solely to filter the training set, so the detector is unsupervised at
scoring. The anomaly score is the per-segment reconstruction error, standardized
per channel against that channel's nominal-training error distribution. This
catches sustained/structural deviations (level-shifts, drifts, waveform
distortion) that the high-pass detector misses.

The two detectors are correlated (+0.62) but split on the hard channels in
opposite directions, so they are fused with a scale-free **rank-mean ensemble**
that inherits both wins. The per-channel oracle ceiling (best detector per
channel, using labels) is ≈0.856 (only ~0.005 above the ensemble) so the
bottleneck is the two hard channels, not the combiner.

### 5. Serving: inductive fusion + FastAPI

Training-time ranking (`rankdata(scores)/len`) is **transductive**; it needs the
whole batch. For online serving it is replaced by an **inductive** equivalent:
each raw score is mapped to a percentile against a *frozen per-channel ECDF*
(empirical cumulative distribution function) computed once on nominal-train
(`src/ops_sat_ad/serving/predict.py`). The ensemble score is
`0.5·(ae_pct + hp_pct)`, thresholded at a target-recall cutoff fixed on the
training split.

The model is packaged as an MLflow `pyfunc` bundle and served by FastAPI
(`api/main.py`): `POST /predict` returns the score, verdict, threshold, per-detector
percentiles (`ae_pct`, `hp_pct`) and model version; `GET /health` reports load
state. The model loads once at startup via the lifespan hook.

### 6. Grounded report (`/report`)

A `POST /report` endpoint turns the numeric prediction into a readable operator
report under a strict **faithfulness ≫ fluency** rule: no un-grounded number may
appear. A deterministic template **owns every number** (rendered once at fixed
precision); the LLM writes only number-free framing at `temperature=0`; a
number-extraction **faithfulness checker** asserts every numeral in the output
traces to an allowed token, else the composer falls back to the pure template.
The narrator sits behind a `Protocol`, so the backend is provider-agnostic. The
report also names the *dominant driver* inferred deterministically (never
by the LLM): **shape** (`ae_pct > hp_pct`) vs
**transient** (`hp_pct > ae_pct`) vs **mixed**.

## Why the naive result was wrong (and how it was fixed)

The first AE run reported pooled AUROC **0.958**, apparently beating every
unsupervised baseline in the paper. Three symptoms exposed it as inflated:

1. **Pooled ≠ honest.** A single pooled AUC is dominated by base-rate
   concentration (76% of anomalies in 3 easy channels). The **macro** AUROC is
   **0.83**, which place it in the paper's unsupervised band, not above it. 
   Fix: report macro.
2. **Length confound.** Resampling to fixed length re-encodes duration as
   reconstruction difficulty; a length-only null scores macro AUROC **0.61**.
   Fix: report the null model and per-channel standardization; treat per-channel
   wins on positively length-correlated channels as unattributable.
3. **Weak ablation.** Nominal-only vs all-train training was within noise,
   because the score was partly driven by the confound rather than the intended
   reconstruction mechanism.

The decisive control is channel **CADC0888**, where length is *negatively*
correlated with the label (−0.24) so a length detector scores **0.26** (worse
than chance) while the AE scores **0.907**. The +0.64 jump hints the AE learned
waveform shape, not duration.

## Detectability = anomaly type × channel SNR

A reconstruction AE flags an anomaly only when its contribution to the error 
rises above the channel's noise floor. On clean channels (CADC0888) spikes stand 
out; on noisy, low-SNR channels (CADC0894) the error is dominated by irreducible 
noise the AE never reproduces, so it fails (AUROC 0.464) and the high-pass partially 
rescues it (0.643). Next step: a per-channel denoising front-end or a period-
aware detector.

## Recommendation

Report **macro, not pooled**. Use the **AE + high-pass rank-mean ensemble** when
robustness matters as it patches both failure modes. Prefer the **high-pass alone**
when on-board compute/simplicity dominates: it is statistically tied with the
ensemble, needs no training or GPU, and OPS-SAT explicitly targets on-board
deployability. At the current test size, per-channel anomaly counts as low as 5 so 
differences below ~0.05 are within sampling noise.

## Run

```bash
# from repo root, venv active
pip install torch --index-url https://download.pytorch.org/whl/cpu   # AE only

# train, evaluate, log to MLflow (experiment: ops-sat-ad)
python -m ops_sat_ad.models.autoencoder
mlflow ui --backend-store-uri sqlite:///mlflow.db                    # view runs

# serve the detector + report endpoints
uvicorn api.main:app --reload
# POST /predict and POST /report with {"channel": "CADC0872", "values": [...]}
```
