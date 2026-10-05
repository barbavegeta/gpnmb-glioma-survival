from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from scipy import stats

# Expansion of the pre-specified covariates into model columns (reference levels:
# grade 2, IDH-mutant 1p/19q-codeleted).
COVARIATE_COLUMNS = {
    "age_10y": ["age_10y"],
    "male": ["male"],
    "grade": ["grade_3", "grade_4"],
    "molecular_class": ["idhmut_noncodel", "idhwt"],
}


def add_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grade = df["grade"].astype(float)
    df["grade_3"] = (grade == 3).astype(int)
    df["grade_4"] = (grade == 4).astype(int)
    df["idhmut_noncodel"] = (df["molecular_class"] == "IDHmut-noncodel").astype(int)
    df["idhwt"] = (df["molecular_class"] == "IDHwt").astype(int)
    return df


def expand(covariates: list[str], df: pd.DataFrame | None = None) -> list[str]:
    """Model columns for covariates.

    With df given, drop columns that are redundant within that subset: constant columns,
    and dummies that are collinear with the baseline hazard (e.g. in GBM, where no tumour
    is 1p/19q-codeleted, the two molecular-class dummies sum to one).
    """
    cols = [c for cov in covariates for c in COVARIATE_COLUMNS[cov]]
    if df is None:
        return cols
    kept: list[str] = []
    design = np.ones((len(df), 1))  # the baseline hazard plays the role of an intercept
    for col in cols:
        trial = np.column_stack([design, df[col].to_numpy(float)])
        if np.linalg.matrix_rank(trial) > np.linalg.matrix_rank(design):
            kept.append(col)
            design = trial
    return kept


def fit_cox(df: pd.DataFrame, time: str, event: str, columns: list[str],
            strata: list[str] | None = None, formula: str | None = None) -> CoxPHFitter:
    keep = list(dict.fromkeys(columns + (strata or [])))
    data = df[[time, event] + keep]
    cph = CoxPHFitter()
    if formula:
        cph.fit(data, duration_col=time, event_col=event, formula=formula, strata=strata)
    else:
        cph.fit(data, duration_col=time, event_col=event, strata=strata)
    return cph


def hr_row(cph: CoxPHFitter, term: str = "gpnmb_z") -> dict:
    s = cph.summary.loc[term]
    return {
        "hr": float(s["exp(coef)"]),
        "ci_low": float(s["exp(coef) lower 95%"]),
        "ci_high": float(s["exp(coef) upper 95%"]),
        "p": float(s["p"]),
    }


def ph_test(cph: CoxPHFitter, df: pd.DataFrame, time: str, event: str, columns: list[str],
            strata: list[str] | None = None) -> pd.DataFrame:
    """Schoenfeld-residual test of proportional hazards (rank-transformed time)."""
    keep = list(dict.fromkeys(columns + (strata or [])))
    res = proportional_hazard_test(cph, df[[time, event] + keep], time_transform="rank")
    return res.summary[["test_statistic", "p"]].rename_axis("term").reset_index()


def lrt(full: CoxPHFitter, reduced: CoxPHFitter) -> tuple[float, int, float]:
    """Likelihood-ratio test between nested Cox models."""
    stat = 2.0 * (full.log_likelihood_ - reduced.log_likelihood_)
    dof = len(full.params_) - len(reduced.params_)
    return float(stat), int(dof), float(stats.chi2.sf(stat, dof))


def c_index_gain(df: pd.DataFrame, time: str, event: str, base: list[str], strata: list[str] | None,
                 reps: int, rng: np.random.Generator) -> dict:
    """Apparent and optimism-corrected gain in Harrell's C from adding GPNMB (Harrell bootstrap)."""
    def gain(train: pd.DataFrame, test: pd.DataFrame) -> float:
        m0 = fit_cox(train, time, event, base, strata)
        m1 = fit_cox(train, time, event, base + ["gpnmb_z"], strata)
        return m1.score(test, scoring_method="concordance_index") - m0.score(test, scoring_method="concordance_index")

    m0 = fit_cox(df, time, event, base, strata)
    m1 = fit_cox(df, time, event, base + ["gpnmb_z"], strata)
    apparent = m1.concordance_index_ - m0.concordance_index_
    optimism = []
    for _ in range(reps):
        boot = df.iloc[rng.integers(0, len(df), len(df))]
        try:
            optimism.append(gain(boot, boot) - gain(boot, df))
        except Exception:  # rare non-convergence in a resample
            continue
    optimism = np.asarray(optimism)
    return {
        "c_base": float(m0.concordance_index_),
        "c_with_gpnmb": float(m1.concordance_index_),
        "gain_apparent": float(apparent),
        "optimism_mean": float(optimism.mean()),
        "gain_corrected": float(apparent - optimism.mean()),
        "bootstrap_reps_used": int(len(optimism)),
    }
