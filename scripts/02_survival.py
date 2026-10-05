#!/usr/bin/env python
"""Pre-specified Cox analyses of GPNMB and survival in TCGA diffuse glioma."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test

from gpnmb_surv.data import analysis_set
from gpnmb_surv.models import (
    COVARIATE_COLUMNS,
    add_model_columns,
    c_index_gain,
    expand,
    fit_cox,
    hr_row,
    lrt,
    ph_test,
)
from gpnmb_surv.utils import ensure_dirs, load_config

warnings.filterwarnings("ignore", category=FutureWarning)


def prepare(cohort: pd.DataFrame, endpoint: str, covariates: list[str]) -> tuple[pd.DataFrame, str, str]:
    df = add_model_columns(analysis_set(cohort, endpoint, covariates))
    return df, f"{endpoint.lower()}_months", f"{endpoint.lower()}_event"


def primary_with_ph_rule(df, time, event, covariates, alpha):
    """Fit the primary model; stratify any categorical covariate that violates PH.

    Returns the final model, its columns, strata and the PH table of the first fit.
    """
    cols = expand(covariates, df) + ["gpnmb_z"]
    model = fit_cox(df, time, event, cols)
    ph = ph_test(model, df, time, event, cols)
    violators = set(ph.loc[ph["p"] < alpha, "term"])
    strata = [cov for cov in ("grade", "molecular_class") if violators & set(COVARIATE_COLUMNS[cov])]
    if strata:
        cols = expand([c for c in covariates if c not in strata], df) + ["gpnmb_z"]
        model = fit_cox(df, time, event, cols, strata=strata)
    return model, cols, strata, ph


def main() -> None:
    cfg = load_config()
    acfg = cfg["analysis"]
    results = Path(cfg["paths"]["results"])
    tables = results / "tables"
    ensure_dirs(tables)
    rng = np.random.default_rng(int(cfg["project"]["random_seed"]))

    cohort = pd.read_csv(Path(cfg["paths"]["processed"]) / "cohort.tsv", sep="\t", index_col=0)
    covariates = acfg["primary_covariates"]
    df, time, event = prepare(cohort, acfg["primary_endpoint"], covariates)
    print(f"Primary analysis set: {len(df)} patients, {df[event].sum()} deaths")

    # 1. Model sequence: how adjustment changes the GPNMB estimate.
    seq = []
    for name, covs in acfg["model_sequence"].items():
        m = fit_cox(df, time, event, expand(covs) + ["gpnmb_z"])
        seq.append({"model": name, "covariates": ", ".join(covs) or "none", "n": len(df),
                    "events": int(df[event].sum()), **hr_row(m)})
    seq = pd.DataFrame(seq)
    seq.to_csv(tables / "model_sequence.csv", index=False)

    # 2. Primary model, with the pre-specified PH rule.
    primary, cols, strata, ph = primary_with_ph_rule(df, time, event, covariates, acfg["ph_test_alpha"])
    ph.to_csv(tables / "ph_test_primary.csv", index=False)
    coefs = primary.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    coefs.columns = ["hr", "ci_low", "ci_high", "p"]
    coefs.to_csv(tables / "primary_model.csv")

    # 3. Non-linearity: spline vs linear GPNMB in the primary model.
    base = [c for c in cols if c != "gpnmb_z"]
    spline_formula = " + ".join(base + [f"bs(gpnmb_z, df={acfg['spline_df']})"])
    spline = fit_cox(df, time, event, base + ["gpnmb_z"], strata=strata or None, formula=spline_formula)
    nonlin = lrt(spline, primary)

    # 4. Effect modification by IDH status.
    df["idh_wt"] = (df["molecular_class"] == "IDHwt").astype(int)
    df["gpnmb_x_idhwt"] = df["gpnmb_z"] * df["idh_wt"]
    inter = fit_cox(df, time, event, cols + ["gpnmb_x_idhwt"], strata=strata or None)
    interaction = lrt(inter, primary)
    by_idh = []
    for label, sub in (("IDH-mutant", df.loc[df["idh_wt"] == 0]), ("IDH-wildtype", df.loc[df["idh_wt"] == 1])):
        sub_cols = expand(covariates, sub) + ["gpnmb_z"]
        m = fit_cox(sub, time, event, sub_cols)
        by_idh.append({"subgroup": label, "n": len(sub), "events": int(sub[event].sum()),
                       "adjusted_for": ", ".join(c for c in sub_cols if c != "gpnmb_z"), **hr_row(m)})
    by_idh = pd.DataFrame(by_idh)
    by_idh.to_csv(tables / "by_idh_status.csv", index=False)

    # 5. Discrimination: does GPNMB add to the clinical-molecular model?
    cgain = c_index_gain(df, time, event, base, strata or None, int(acfg["bootstrap_reps"]), rng)

    # 6. Descriptive Kaplan-Meier groups (tertiles; no optimised cut-point).
    df["gpnmb_tertile"] = pd.qcut(df["gpnmb_log2"], 3, labels=["Low", "Middle", "High"])
    km_test = multivariate_logrank_test(df[time], df["gpnmb_tertile"], df[event])
    df[["gpnmb_log2", "gpnmb_tertile", "molecular_class", time, event]].to_csv(tables / "km_input.csv")

    # 7. Sensitivity analyses.
    sens = []
    pfi, ptime, pevent = prepare(cohort, acfg["secondary_endpoint"], covariates)
    for label, data, t, e in (
        ("PFI endpoint", pfi, ptime, pevent),
        ("LGG only", df.loc[df["cancer"] == "LGG"], time, event),
        ("GBM only", df.loc[df["cancer"] == "GBM"], time, event),
    ):
        sub_cols = expand(covariates, data) + ["gpnmb_z"]
        m = fit_cox(data, t, e, sub_cols)
        sens.append({"analysis": label, "n": len(data), "events": int(data[e].sum()),
                     "adjusted_for": ", ".join(c for c in sub_cols if c != "gpnmb_z"), **hr_row(m)})
    sens = pd.DataFrame(sens)
    sens.to_csv(tables / "sensitivity.csv", index=False)

    summary = {
        "n": len(df), "events": int(df[event].sum()),
        "primary_hr_per_sd": hr_row(primary),
        "stratified_by": strata,
        "ph_min_p": float(ph["p"].min()),
        "nonlinearity_lrt": {"chi2": nonlin[0], "df": nonlin[1], "p": nonlin[2]},
        "idh_interaction_lrt": {"chi2": interaction[0], "df": interaction[1], "p": interaction[2]},
        "c_index": cgain,
        "km_tertile_logrank": {"chi2": float(km_test.test_statistic), "p": float(km_test.p_value)},
    }
    (tables / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 160)
    print("\nModel sequence (HR per SD of GPNMB):")
    print(seq.round(4).to_string(index=False))
    print(f"\nPH test, primary model: min P = {ph['p'].min():.3f}; stratified by: {strata or 'none'}")
    print(f"Non-linearity LRT: chi2 {nonlin[0]:.2f}, df {nonlin[1]}, P = {nonlin[2]:.3f}")
    print(f"GPNMB x IDH-wildtype LRT: chi2 {interaction[0]:.2f}, df {interaction[1]}, P = {interaction[2]:.3f}")
    print(by_idh.round(4).to_string(index=False))
    print(f"\nHarrell's C: {cgain['c_base']:.4f} -> {cgain['c_with_gpnmb']:.4f}; gain {cgain['gain_apparent']:.4f} "
          f"apparent, {cgain['gain_corrected']:.4f} optimism-corrected ({cgain['bootstrap_reps_used']} reps)")
    print(f"KM tertiles log-rank: chi2 {km_test.test_statistic:.1f}, P = {km_test.p_value:.2e}")
    print("\nSensitivity:")
    print(sens.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
