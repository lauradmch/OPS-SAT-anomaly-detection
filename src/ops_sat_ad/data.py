import pandas as pd
import matplotlib.pyplot as plt

def load_segments(path="data/segments.csv"):
    return pd.read_csv(path, parse_dates=["timestamp"])

def load_dataset(path="data/dataset.csv"):
    return pd.read_csv(path)

def plot_channel(seg_df, channel: str, save_dir="notebooks"):
    sub = seg_df[seg_df.channel == channel].sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(14, 4))
    for seg_id, g in sub.groupby("segment"):
        color = "red" if g["anomaly"].iloc[0] == 1 else "steelblue"
        ax.plot(g["timestamp"], g["value"], color=color, linewidth=0.8)
    ax.set_title(f"{channel}  (red = anomalous segment)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{channel}_overview.png", dpi=120)
    plt.show()

if __name__ == "__main__":
    seg = load_segments()
    ds = load_dataset()

    print("Channels:", seg.channel.unique())
    print("\nSegments per channel:\n", seg.groupby("channel")["segment"].nunique())
    print("\ndataset.csv train/anomaly counts:\n", ds.groupby(["train", "anomaly"]).size())

    # sanity: segments.csv and dataset.csv should reference the same segment IDs
    assert set(ds.segment) == set(seg.segment), "segment ID mismatch between files!"
    print("\nsegment IDs match between segments.csv and dataset.csv ✔")

    plot_channel(seg, channel=seg.channel.unique()[0])