#!/usr/bin/env python
"""Download the open-access TCGA Pan-Cancer Atlas inputs from the UCSC Xena hub.

Only the target gene is pulled from the expression matrix, so the download is a few
hundred kilobytes rather than the full ~11,000-sample matrix.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import pandas as pd
import xenaPython as xena

from gpnmb_surv.utils import ensure_dirs, load_config

warnings.filterwarnings("ignore", category=FutureWarning)


def _decode(value, labels: list[str] | None):
    if value is None or value == "NaN" or (isinstance(value, float) and math.isnan(value)):
        return None
    return labels[int(value)] if labels else value


def fetch_table(hub: str, dataset: str) -> pd.DataFrame:
    """Fetch every field of a Xena clinical matrix as a sample x field table.

    Categorical fields come back as integer codes and are mapped to their labels.
    """
    samples = xena.dataset_samples(hub, dataset, None)
    fields = [f for f in xena.dataset_field(hub, dataset) if f != "sampleID"]
    _, values = xena.dataset_probe_values(hub, dataset, samples, fields)
    codes = {c["name"]: c["code"].split("\t") for c in xena.field_codes(hub, dataset, fields) if c.get("code")}
    columns = {f: [_decode(v, codes.get(f)) for v in vals] for f, vals in zip(fields, values)}
    return pd.DataFrame(columns, index=pd.Index(samples, name="sample"))


def main() -> None:
    cfg = load_config()
    src = cfg["source"]
    hub = src["hub"]
    raw = Path(cfg["paths"]["raw"])
    ensure_dirs(raw)

    for key in ("survival", "subtypes", "phenotype"):
        table = fetch_table(hub, src[key])
        table.to_csv(raw / f"{key}.tsv", sep="\t")
        print(f"{key}: {table.shape[0]:,} samples x {table.shape[1]} fields")

    gene = cfg["project"]["target_gene"]
    samples = xena.dataset_samples(hub, src["expression"], None)
    _, [values] = xena.dataset_probe_values(hub, src["expression"], samples, [gene])
    expr = pd.DataFrame({gene: [_decode(v, None) for v in values]}, index=pd.Index(samples, name="sample"))
    expr.to_csv(raw / "expression.tsv", sep="\t")
    print(f"expression: {gene} for {expr[gene].notna().sum():,} samples")


if __name__ == "__main__":
    main()
