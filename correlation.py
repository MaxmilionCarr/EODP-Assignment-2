import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import quick_data

# Graphing
TITLE_SIZE = 24
SUBTITLE_SIZE = 18
LABEL_SIZE = 14

# Wasted time
JOURNEY_FILES = [
    "datasets/journey_work.csv",
    "datasets/journey_education.csv",
]
JOURNEY_COLS = ["persid", "journey_travel_time", "journey_elapsed_time"]

# Transport types
GROUPS = {
    "Public": ["Train", "Public Bus", "Tram", "School Bus"],
    "Private": ["Motorcycle", "Vehicle Driver", "Vehicle Passenger", "Taxi / Rideshare"],
    "Active": ["Walking", "Bicycle"],
}

OUT_DIR = "correlation_graphs"

# Calculate wasted time per person
def wasted_time():
    frames = []

    for p in JOURNEY_FILES:
        df = pd.read_csv(p, usecols=JOURNEY_COLS, low_memory=False)
        df[JOURNEY_COLS[1:]] = df[JOURNEY_COLS[1:]].apply(pd.to_numeric, errors="coerce")
        df["wasted_time"] = (df["journey_elapsed_time"] - df["journey_travel_time"]).clip(lower=0)
        frames.append(df[["persid", "wasted_time"]])

    j = pd.concat(frames, ignore_index=True)
    return j.groupby("persid", as_index=False)["wasted_time"].mean()

# Filter feature matrix
def get_X(df):
    drop_cols = {"persid", "wasted_time", "most_used_mode", "perspoststratweight", "overall_trip_efficiency"}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return X.select_dtypes(include=[np.number]).fillna(0)

# Gets pearson r values for all numeric features
def corr_values(X, y):
    ok = np.isfinite(y)
    X = X.loc[ok]
    y = y.loc[ok]
    r = X.apply(lambda col: col.corr(y, method="pearson")).dropna()
    return r.reindex(r.abs().sort_values(ascending=False).index)

# Writes data summary txt file for graph features
def write_report(series, out_path):
    out_dir = os.path.dirname(out_path)
    report_path = os.path.join(out_dir, "corr_summary.txt")
    graph = os.path.splitext(os.path.basename(out_path))[0]
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(f"{graph}\n")
        for feat, val in series.items():
            f.write(f"  {feat}: {val:.6f}\n")
        f.write("\n")

# Helper function to plot bar graphs
def plot_bar(series, xlabel, title, out_path):
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(series) + 2)))

    ax.barh(series.index, series.values)
    ax.axvline(0, linestyle="--", linewidth=1)
    # Manually set x axis scale for easier graph comparisons
    ax.set_xlim(-0.17, 0.17)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel("Feature", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.invert_yaxis()

    plt.tight_layout()
    out = os.path.abspath(out_path)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    write_report(series, out)

# Perform Pearson correlation per travel mode
def travel_mode_corr(df, X, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    groups = {
        name: modes
        for name, modes in GROUPS.items()
        if df["most_used_mode"].isin(modes).astype(int).nunique() == 2
    }

    R = pd.DataFrame(index=X.columns, columns=groups.keys(), dtype=float)

    for name, modes in groups.items():
        y = df["most_used_mode"].isin(modes).astype(float)
        ok = np.isfinite(y)
        R[name] = X.loc[ok].apply(lambda col: col.corr(y.loc[ok], method="pearson"))

    R = R.fillna(0.0)
    order = R.abs().max(axis=1).sort_values(ascending=False).index.tolist()

    for group_name in groups.keys():
        s = R.loc[order, group_name]
        plot_bar(
            s,
            xlabel="Pearson r",
            title=f"Feature Correlations with {group_name} Transport",
            out_path=os.path.join(out_dir, f"{group_name.lower()}_transport.png"),
        )

# Construct dataframe and produce plots
def corr_analysis():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Rewrite file
    with open(os.path.join(OUT_DIR, "corr_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Correlation summary\n\n")

    df = quick_data()
    df = df.merge(wasted_time(), on="persid", how="left").dropna(subset=["wasted_time"])

    y = df["wasted_time"].astype(float)
    x = get_X(df)
    ranks = corr_values(x, y)

    plot_bar(
        ranks,
        xlabel="Pearson r",
        title="Feature Correlations with Wasted Time",
        out_path=os.path.join(OUT_DIR, "wasted_time.png"),
    )
    travel_mode_corr(df, x, out_dir=OUT_DIR)

# Run program
if __name__ == "__main__":
    corr_analysis()