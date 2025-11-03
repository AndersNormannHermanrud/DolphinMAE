


# === CONFIG (edit these) ======================================================
folder = f"finetunepretrain_data"
EPOCH_CSV = f"C:\\Users\\ander\\Github\\Masters\\Results_from_cluster\\mae\\Final_Experiment\\{folder}\\epoch_loss.csv"         # train epoch loss CSV from TensorBoard
STEP_CSV  = f"C:\\Users\\ander\\Github\\Masters\\Results_from_cluster\\mae\\Final_Experiment\\{folder}\\step_loss.csv"          # train step loss CSV
VAL_CSV   = f"C:\\Users\\ander\\Github\\Masters\\Results_from_cluster\\mae\\Final_Experiment\\{folder}\\val_epoch_loss.csv"     # validation epoch loss CSV
EPOCH_MAP_CSV = f"C:\\Users\\ander\\Github\\Masters\\Results_from_cluster\\mae\\Final_Experiment\\{folder}\\epoch.csv"
# Optional: if your CSVs contain multiple tags, filter to the exact tag(s)
# Set to None to keep all tags.
EPOCH_TAG_FILTER = None #r"(train/.+loss_epoch|loss_epoch|train/loss_epoch)"
STEP_TAG_FILTER  = None #r"(train/.+loss(?!_epoch)|train/step_loss|loss)"   # example
VAL_TAG_FILTER   = None #r"(val/.+loss_epoch|val/loss_epoch|validation/loss)"

TITLE_BASE = "AudioMAE Pretraining"     # figure suptitle prefix
SMOOTH_ALPHA = 0.0                      # 0.0 = no smoothing; e.g., 0.1–0.3 for EMA
USE_LOG_Y = False                       # True if your field prefers log-scale loss
OUT_BASENAME = "finetunepretrain_loss"          # filenames will be appended with panel name
# =============================================================================

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable, Tuple

# --- Styles for publication ---
mpl.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 18,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.8,
    "lines.linewidth": 2.2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

def ema(series: pd.Series, alpha: float) -> pd.Series:
    return series if not alpha or alpha <= 0 else series.ewm(alpha=alpha, adjust=False).mean()

def load_tb_csv(path: str, tag_regex: str | None):
    """Load a TensorBoard CSV and optionally filter rows by regex on 'tag'."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Expected: wall_time, step, value, run, tag (TB sometimes omits tag/run)
    if "step" not in df.columns or "value" not in df.columns:
        raise ValueError(f"{path}: missing required columns ('step','value'). Got: {df.columns}")
    if "run" not in df.columns: df["run"] = Path(path).stem
    if "tag" not in df.columns: df["tag"] = Path(path).stem
    if tag_regex is not None:
        pat = re.compile(tag_regex)
        df = df[df["tag"].apply(lambda t: bool(pat.search(str(t))))]
    return df

def compute_common_ylim(dfs: list[pd.DataFrame], smooth: float):
    vmin, vmax = np.inf, -np.inf
    for df in dfs:
        if df is None or df.empty:
            continue
        for _, g in df.groupby("run"):
            vals = ema(g.sort_values("step")["value"], smooth).to_numpy()
            if vals.size:
                vmin = min(vmin, np.nanmin(vals))
                vmax = max(vmax, np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return (0.0, 1.0)
    if vmin == vmax:
        pad = 0.05 * (abs(vmin) + 1e-12)
        return (vmin - pad, vmax + pad)
    pad = 0.03 * (vmax - vmin)
    return (vmin - pad, vmax + pad)

def load_epoch_map(csv_path: str) -> Tuple[Callable[[np.ndarray], np.ndarray],
                                           Callable[[np.ndarray], np.ndarray]]:
    """
    Build monotonic transforms between step and epoch using the epoch CSV.
    Returns (step_to_epoch, epoch_to_step).
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Expect 'step' and 'value' (value = epoch index)
    if "step" not in df.columns or "value" not in df.columns:
        raise ValueError(f"{csv_path}: needs 'step' and 'value' columns.")
    df = df[["step", "value"]].dropna()
    # De-duplicate and sort
    df = df.sort_values("step").drop_duplicates("step", keep="last")
    steps = df["step"].to_numpy().astype(float)
    epochs = df["value"].to_numpy().astype(float)

    # Ensure monotonicity for interpolation (best-effort fix if needed)
    # If epochs are non-decreasing integers {0,1,2,...}, this is already monotonic.
    # Otherwise, enforce monotonic with cummax.
    epochs = np.maximum.accumulate(epochs)

    # Guard against degenerate mapping
    if steps.size < 2 or np.allclose(steps[0], steps[-1]):
        # fallback: identity
        return (lambda s: s, lambda e: e)

    # Build piecewise-linear interpolation; extrapolate by clamping to ends
    def step_to_epoch(x):
        x = np.asarray(x, dtype=float)
        y = np.interp(x, steps, epochs, left=epochs[0], right=epochs[-1])
        return y

    # For inverse, ensure epochs are strictly increasing for interp; add a tiny epsilon if needed
    eps = 1e-9
    epochs_inc = epochs.copy()
    for i in range(1, len(epochs_inc)):
        if epochs_inc[i] <= epochs_inc[i-1]:
            epochs_inc[i] = epochs_inc[i-1] + eps

    def epoch_to_step(y):
        y = np.asarray(y, dtype=float)
        x = np.interp(y, epochs_inc, steps, left=steps[0], right=steps[-1])
        return x

    return step_to_epoch, epoch_to_step


def plot_panel(df: pd.DataFrame, title: str, xlabel: str, ylimits, outname: str,
               smooth: float, logy: bool, replace_bottom_ticks=False, is_epoch_panel=False):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    handles, labels = [], []

    if df is not None and not df.empty:
        for i, (run, g) in enumerate(sorted(df.groupby("run"), key=lambda kv: kv[0])):
            g = g.sort_values("step")
            y = ema(g["value"], smooth)
            h, = ax.plot(g["step"].to_numpy(), y.to_numpy(),
                         linestyle=linestyles[i % len(linestyles)], label="Batch Step Loss")
            handles.append(h); labels.append(run)

    ax.set_title(title)
    ax.set_xlabel(xlabel.capitalize())
    ax.set_ylabel("BCE Loss")
    ax.grid(True, axis="both", linestyle="--", linewidth=0.5)
    if logy: ax.set_yscale("log")
    ax.set_ylim(*ylimits)

    # Optionally replace bottom tick labels with epoch numbers (still step-scaled)
    if is_epoch_panel and HAS_EPOCH_MAP and replace_bottom_ticks:
        # formatter to display epoch at the step tick
        def _fmt_epoch(x, pos):
            e = STEP_TO_EPOCH(np.array([x]))[0]
            return f"{int(round(e))}"

        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_fmt_epoch))

    ax.legend(loc="upper right", frameon=False, fontsize=16)

    fig.tight_layout()
    plt.show()
    pdf = f"{outname}.pdf"
    svg = f"{outname}.svg"
    fig.savefig(pdf)
    fig.savefig(svg)
    #print(f"Saved: {pdf} and {svg}")
    #plt.close(fig)

def plot_epoch_two_lines(train_df: pd.DataFrame, val_df: pd.DataFrame,
                         out_basename: str, title: str, replace_bottom_ticks=False, is_epoch_panel=False, ylims=None):

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.grid(True, linestyle="--", linewidth=0.5)

    # Style: use solid for train, dashed for val; distinguish runs by thin variations
    def plot_grouped(df, label_prefix, linestyle):
        handles = []
        for i, (run, g) in enumerate(sorted(df.groupby("run"), key=lambda kv: kv[0])):
            g = g.sort_values("step")
            y = ema(g["value"], SMOOTH_ALPHA)
            (h,) = ax.plot(g["step"].to_numpy(), y.to_numpy(),
                           linestyle=linestyle, label=f"{label_prefix}")
            handles.append(h)
        return handles

    h_train = plot_grouped(train_df, "Train", "solid")
    h_val   = plot_grouped(val_df,   "Validation", "solid")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    if USE_LOG_Y: ax.set_yscale("log")
    ax.set_ylim(*ylims)

    # Optionally replace bottom tick labels with epoch numbers (still step-scaled)
    if is_epoch_panel and HAS_EPOCH_MAP and replace_bottom_ticks:
        # formatter to display epoch at the step tick
        def _fmt_epoch(x, pos):
            e = STEP_TO_EPOCH(np.array([x]))[0]
            # show as integer if near integer
            return f"{int(round(e))}"

        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_fmt_epoch))

    # Consolidated legend (readable in grayscale)
    ax.legend(loc="upper right", frameon=False, fontsize=16)

    fig.tight_layout()
    plt.show()
    fig.savefig(f"{out_basename}.pdf")
    #fig.savefig(f"{out_basename}.svg")
    #print(f"Saved: {out_basename}.pdf/.svg")
    #plt.close(fig)

# --- Load, align y-limits, and plot three separate figures ---
df_epoch = load_tb_csv(EPOCH_CSV, EPOCH_TAG_FILTER)
#df_step  = load_tb_csv(STEP_CSV,  STEP_TAG_FILTER)
df_val   = load_tb_csv(VAL_CSV,   VAL_TAG_FILTER)

#ylims = compute_common_ylim([df_epoch, df_step, df_val], SMOOTH_ALPHA)
ylims = (0.000, 0.35)

# Build the mapping (safe if file missing -> optional)
try:
    STEP_TO_EPOCH, EPOCH_TO_STEP = load_epoch_map(EPOCH_MAP_CSV)
    HAS_EPOCH_MAP = True
except Exception as e:
    print(f"[warn] epoch map disabled: {e}")
    raise e
    #STEP_TO_EPOCH = lambda s: s
    #EPOCH_TO_STEP = lambda e: e
    #HAS_EPOCH_MAP = False
"""
plot_panel(df_epoch,
           f"",
           xlabel="epoch",
           ylimits=ylims,
           outname=f"{OUT_BASENAME}_epoch",
           smooth=SMOOTH_ALPHA,
           logy=USE_LOG_Y,
           replace_bottom_ticks=True,
           is_epoch_panel=True)
"""
"""
plot_panel(df_step,
           f"",
           xlabel="step (million)",
           ylimits=(0.0006, 0.0043),
           outname=f"{OUT_BASENAME}_step",
           smooth=SMOOTH_ALPHA,
           logy=USE_LOG_Y,
           replace_bottom_ticks=False,
           is_epoch_panel=False)
"""
"""
plot_panel(df_val,
           f"",
           xlabel="epoch",
           ylimits=ylims,
           outname=f"{OUT_BASENAME}_val",
           smooth=SMOOTH_ALPHA,
           logy=USE_LOG_Y,
           replace_bottom_ticks=True,
           is_epoch_panel=True)
"""
plot_epoch_two_lines(df_epoch,
                     df_val,
                     out_basename=f"{OUT_BASENAME}_joint",
                     title=f"",
                     replace_bottom_ticks=True,
                     is_epoch_panel=True,
                     ylims=ylims)