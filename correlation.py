import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import quick_data

# Graphing
TITLE_SIZE = 24
SUBTITLE_SIZE = 18
LABEL_SIZE = 14

# Transport types
GROUPS = {
    "Public": ["Train", "Public Bus", "Tram", "School Bus"],
    "Private": ["Motorcycle", "Vehicle Driver", "Vehicle Passenger", "Taxi / Rideshare"],
    "Active": ["Walking", "Bicycle"],
}

OUT_DIR = "correlation_graphs"

# Filter feature matrix
def get_X(df):
    drop_cols = {"persid", "wasted_time", "most_used_mode", "perspoststratweight",
                 "overall_trip_efficiency", "overall_trip_length"}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return X.select_dtypes(include=[np.number]).fillna(0)

# Computes a global feature order for graphs
def feature_order(df, X):
    targets = {}

    # Scalar targets
    targets["wasted_time"] = pd.to_numeric(df["wasted_time"], errors="coerce")
    targets["trip_efficiency"] = pd.to_numeric(df["overall_trip_efficiency"], errors="coerce")
    targets["trip_length"] = pd.to_numeric(df["overall_trip_length"], errors="coerce")

    # Mode-group targets
    for name in ("Public", "Private", "Active"):
        y = (df["most_used_mode"] == name).astype(float)
        if y.nunique() == 2:
            targets[f"{name}_transport"] = y

    R_all = pd.DataFrame(index=X.columns, dtype=float)
    for key, y in targets.items():
        ok = np.isfinite(y)
        R_all[key] = X.loc[ok].apply(lambda col: col.corr(y.loc[ok], method="pearson"))
    R_all = R_all.fillna(0.0)
    order = R_all.abs().max(axis=1).sort_values(ascending=False).index
    return order, R_all

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
    ax.set_xlim(-0.35, 0.35)
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
def travel_mode_corr(df, X, order, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    for name in ("Public", "Private", "Active"):
        y = (df["most_used_mode"] == name).astype(float)

        ok = np.isfinite(y)
        s = X.loc[ok].apply(lambda col: col.corr(y.loc[ok], method="pearson")).fillna(0.0)
        s = s.reindex(order)

        plot_bar(s, "Pearson r",
                 f"Feature Correlations with {name} Transport",
                 os.path.join(out_dir, f"{name.lower()}_transport.png"))

# Construct dataframe and produce plots
def corr_analysis():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Rewrite file
    with open(os.path.join(OUT_DIR, "corr_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Correlation summary\n\n")

    df = quick_data()

    x = get_X(df)

    # Compute global feature order for plots
    order, R_all = feature_order(df, x)

    # Wasted time plot
    plot_bar(
        R_all.loc[order, "wasted_time"],
        xlabel="Pearson r",
        title="Feature Correlations with Wasted Time",
        out_path=os.path.join(OUT_DIR, "wasted_time.png"),
    )

    # Travel mode plots
    travel_mode_corr(df, x, order, out_dir=OUT_DIR)

    # Trip efficiency plot
    plot_bar(
        R_all.loc[order, "trip_efficiency"],
        xlabel="Pearson r",
        title="Feature Correlations with Trip Efficiency",
        out_path=os.path.join(OUT_DIR, "trip_efficiency.png"),
    )

    # Trip length plot
    plot_bar(
        R_all.loc[order, "trip_length"],
        xlabel="Pearson r",
        title="Feature Correlations with Trip Length",
        out_path=os.path.join(OUT_DIR, "trip_length.png"),
    )

def main():
    corr_analysis()

# Run program
if __name__ == "__main__":
    main()