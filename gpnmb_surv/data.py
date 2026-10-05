from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 365.25 / 12


def grade_from_cdr(cancer: pd.Series, histological_grade: pd.Series) -> pd.Series:
    """WHO grade 2-4. GBM is grade 4 by definition; TCGA-CDR leaves its grade blank."""
    lgg = histological_grade.map({"G2": 2, "G3": 3})
    return pd.Series(np.where(cancer == "GBM", 4, lgg), index=cancer.index, dtype="float").astype("Int64")


def molecular_class(cluster: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Map Ceccarelli 2016 glioma clusters (e.g. 'GBM_LGG.Codel') to IDH/1p19q classes."""
    label = cluster.astype("string").str.replace(r"^GBM_LGG\.", "", regex=True)
    return label.map(mapping).astype("string")


def one_sample_per_patient(samples: pd.Index, type_code: str) -> pd.Index:
    """Keep samples of one TCGA sample type (characters 14-15), first per patient."""
    samples = pd.Index(samples)
    keep = samples[samples.str[13:15] == type_code]
    patients = keep.str[:12]
    return keep[~patients.duplicated()]


def build_cohort(raw: Path, cfg: dict) -> pd.DataFrame:
    gene = cfg["project"]["target_gene"]
    ccfg = cfg["cohort"]

    surv = pd.read_csv(raw / "survival.tsv", sep="\t", index_col=0)
    subtypes = pd.read_csv(raw / "subtypes.tsv", sep="\t", index_col=0)
    expr = pd.read_csv(raw / "expression.tsv", sep="\t", index_col=0)

    surv = surv.loc[surv["cancer type abbreviation"].isin(ccfg["cancer_types"])]
    samples = one_sample_per_patient(surv.index.intersection(expr.index), str(ccfg["sample_type_code"]))
    s = surv.loc[samples]

    out = pd.DataFrame(index=pd.Index(samples, name="sample"))
    out["patient"] = s["_PATIENT"]
    out["cancer"] = s["cancer type abbreviation"]
    out["gpnmb_log2"] = expr.loc[samples, gene].astype(float)
    out["age"] = s["age_at_initial_pathologic_diagnosis"].astype(float)
    out["age_10y"] = out["age"] / cfg["analysis"]["age_scale_years"]
    out["male"] = (s["gender"].str.upper() == "MALE").astype(int)
    out["grade"] = grade_from_cdr(s["cancer type abbreviation"], s["histological_grade"])
    out["molecular_class"] = molecular_class(subtypes["Subtype_Selected"].reindex(samples), cfg["molecular_class"])
    for endpoint in ("OS", "PFI"):
        out[f"{endpoint.lower()}_event"] = s[endpoint].astype("Int64")
        out[f"{endpoint.lower()}_months"] = s[f"{endpoint}.time"].astype(float) / DAYS_PER_MONTH
    return out


def analysis_set(cohort: pd.DataFrame, endpoint: str, covariates: list[str]) -> pd.DataFrame:
    """Complete cases for one endpoint, with GPNMB standardised within that set."""
    time, event = f"{endpoint.lower()}_months", f"{endpoint.lower()}_event"
    needed = ["gpnmb_log2", time, event] + covariates
    df = cohort.dropna(subset=needed).copy()
    df = df.loc[df[time] > 0]
    df[event] = df[event].astype(int)
    df["gpnmb_z"] = (df["gpnmb_log2"] - df["gpnmb_log2"].mean()) / df["gpnmb_log2"].std()
    return df
