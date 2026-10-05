#!/usr/bin/env python
"""Assemble one primary tumour per patient with GPNMB, survival and covariates."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from gpnmb_surv.data import build_cohort
from gpnmb_surv.utils import ensure_dirs, load_config


def describe(cohort: pd.DataFrame) -> pd.DataFrame:
    """Table 1, by molecular class (descriptive only, no outcome information)."""
    groups = cohort.assign(molecular_class=cohort["molecular_class"].fillna("Unclassified"))
    rows = []
    for name, g in [("All", groups)] + list(groups.groupby("molecular_class")):
        rows.append({
            "group": name,
            "n": len(g),
            "LGG": int((g["cancer"] == "LGG").sum()),
            "GBM": int((g["cancer"] == "GBM").sum()),
            "age_median": round(g["age"].median(), 1),
            "male_pct": round(100 * g["male"].mean(), 1),
            "grade_2_3_4": "/".join(str(int((g["grade"] == k).sum())) for k in (2, 3, 4)),
            "gpnmb_log2_median": round(g["gpnmb_log2"].median(), 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    raw, processed = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["processed"])
    tables = Path(cfg["paths"]["results"]) / "tables"
    ensure_dirs(processed, tables)

    cohort = build_cohort(raw, cfg)
    cohort.to_csv(processed / "cohort.tsv", sep="\t")
    table1 = describe(cohort)
    table1.to_csv(tables / "table1_cohort.csv", index=False)

    print(f"{len(cohort):,} patients ({(cohort['cancer'] == 'LGG').sum()} LGG, {(cohort['cancer'] == 'GBM').sum()} GBM)")
    print(f"Missing: molecular class {cohort['molecular_class'].isna().sum()}, grade {cohort['grade'].isna().sum()}, "
          f"OS time {cohort['os_months'].isna().sum()}")
    print(table1.to_string(index=False))


if __name__ == "__main__":
    main()
