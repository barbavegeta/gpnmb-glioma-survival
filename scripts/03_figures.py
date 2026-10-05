#!/usr/bin/env python
"""Figures: GPNMB by molecular class, Kaplan-Meier by tertile, and a forest plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from gpnmb_surv.utils import ensure_dirs, load_config

INK, INK2, INK3, GRID = "#221c12", "#4b4232", "#6e6450", "#e6dfcd"
TERTILE_COLOURS = {"Low": "#5e9f96", "Middle": "#1d7068", "High": "#0a3c37"}  # sequential teal
TERTILE_STYLES = {"Low": ":", "Middle": "--", "High": "-"}
ACCENT = "#0b6e66"
CLASS_ORDER = ["IDHmut-codel", "IDHmut-noncodel", "IDHwt"]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK3, "axes.labelcolor": INK2, "xtick.color": INK3,
    "ytick.color": INK3, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def gpnmb_by_class(cohort: pd.DataFrame, path: Path) -> None:
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    data = [cohort.loc[cohort["molecular_class"] == c, "gpnmb_log2"].dropna() for c in CLASS_ORDER]
    for i, values in enumerate(data):
        x = i + rng.uniform(-0.18, 0.18, len(values))
        ax.scatter(x, values, s=9, color=ACCENT, alpha=0.35, linewidths=0)
        ax.hlines(values.median(), i - 0.3, i + 0.3, color=INK, linewidth=2)
    names = ["IDH-mutant\ncodeleted", "IDH-mutant\nnon-codeleted", "IDH-wildtype\n"]
    ax.set_xticks(range(len(CLASS_ORDER)), [f"{n}\nn = {len(v)}" for n, v in zip(names, data)])
    ax.set_ylabel("GPNMB, log2 normalised expression")
    ax.set_title("GPNMB is highest in IDH-wildtype glioma", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, path)


def km_panels(km: pd.DataFrame, path: Path) -> None:
    panels = [("All gliomas", km),
              ("IDH-mutant", km.loc[km["molecular_class"].str.startswith("IDHmut")]),
              ("IDH-wildtype", km.loc[km["molecular_class"] == "IDHwt"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
    for ax, (title, sub) in zip(axes, panels):
        # Tertiles are cut within each panel so every curve holds a third of that group.
        tertile = pd.qcut(sub["gpnmb_log2"], 3, labels=["Low", "Middle", "High"])
        for level in ("Low", "Middle", "High"):
            g = sub.loc[tertile == level]
            kmf = KaplanMeierFitter().fit(g["os_months"], g["os_event"], label=f"{level} GPNMB tertile")
            kmf.plot_survival_function(ax=ax, ci_show=False, color=TERTILE_COLOURS[level],
                                       linestyle=TERTILE_STYLES[level], linewidth=2, legend=False)
        p = multivariate_logrank_test(sub["os_months"], tertile, sub["os_event"]).p_value
        ax.set_title(title, loc="left", pad=20)
        ax.text(0, 1.025, f"n = {len(sub)}, {int(sub['os_event'].sum())} deaths · log-rank P = {p:.2g}",
                transform=ax.transAxes, fontsize=8.5, color=INK2)
        ax.set_xlabel("Months from diagnosis")
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 1.02)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Overall survival")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.legend(handles, labels, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.05, 1.0))
    save(fig, path)


def forest(tables: Path, path: Path) -> None:
    seq = pd.read_csv(tables / "model_sequence.csv")
    idh = pd.read_csv(tables / "by_idh_status.csv")
    sens = pd.read_csv(tables / "sensitivity.csv")
    labels = {"unadjusted": "Unadjusted", "demographics": "+ age, sex", "primary": "+ grade, IDH/1p19q (primary)"}
    rows = [("Adjustment", None)] + [(labels[r.model], r) for r in seq.itertuples()]
    rows += [("Within IDH class", None)] + [(r.subgroup, r) for r in idh.itertuples()]
    rows += [("Sensitivity", None)] + [(r.analysis, r) for r in sens.itertuples()]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    y = len(rows)
    yticks, ylabels = [], []
    for label, r in rows:
        y -= 1
        if r is None:
            ax.annotate(label, xy=(0, y), xycoords=("axes fraction", "data"), xytext=(-8, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=9.5, fontweight="bold", color=INK)
            continue
        primary = label.endswith("(primary)")
        colour = ACCENT if primary else INK2
        ax.plot([r.ci_low, r.ci_high], [y, y], color=colour, linewidth=2, solid_capstyle="round")
        ax.plot(r.hr, y, "o", color=colour, markersize=7 if primary else 6, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(3.6, y, f"{r.hr:.2f} ({r.ci_low:.2f}–{r.ci_high:.2f})", fontsize=9, color=INK2, va="center")
        yticks.append(y)
        ylabels.append(label)
    ax.axvline(1, color=INK3, linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlim(0.7, 3.4)
    ax.set_xticks([0.75, 1, 1.5, 2, 3], ["0.75", "1", "1.5", "2", "3"])
    ax.minorticks_off()
    ax.set_yticks(yticks, ylabels)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Hazard ratio per SD of GPNMB (95% CI), log scale")
    ax.set_title("Adjustment removes most of the association", loc="left")
    ax.text(3.6, len(rows) - 0.6, "HR (95% CI)", fontsize=9, color=INK3, va="center")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, path)


def main() -> None:
    cfg = load_config()
    results = Path(cfg["paths"]["results"])
    tables, figs = results / "tables", results / "figures"
    ensure_dirs(figs)
    cohort = pd.read_csv(Path(cfg["paths"]["processed"]) / "cohort.tsv", sep="\t", index_col=0)
    km = pd.read_csv(tables / "km_input.csv", index_col=0)
    km["gpnmb_tertile"] = pd.Categorical(km["gpnmb_tertile"], ["Low", "Middle", "High"])

    gpnmb_by_class(cohort, figs / "gpnmb_by_molecular_class")
    km_panels(km, figs / "km_gpnmb_tertiles")
    forest(tables, figs / "forest_gpnmb")
    print("Figures written to", figs)


if __name__ == "__main__":
    main()
