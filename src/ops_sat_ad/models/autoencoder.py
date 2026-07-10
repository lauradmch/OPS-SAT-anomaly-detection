"""
Day 4 — 1D convolutional autoencoder (Conv1D-AE) anomaly detector.

Idea (semi-supervised / "train-on-normal-only" reconstruction detector):
  The AE learns to compress-and-rebuild the NOMINAL manifold of a segment.
  At inference, anomalous segments rebuild badly -> large reconstruction error
  -> high anomaly score. Labels are used ONLY to filter the training set,
  never for supervised fitting, so the detector stays unsupervised at scoring.

"""