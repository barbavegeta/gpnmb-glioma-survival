import numpy as np
import pandas as pd

from gpnmb_surv.data import analysis_set, grade_from_cdr, molecular_class, one_sample_per_patient
from gpnmb_surv.models import add_model_columns, expand

MAPPING = {"Codel": "IDHmut-codel", "G-CIMP-high": "IDHmut-noncodel", "Classic-like": "IDHwt"}


def test_gbm_is_grade_4_and_discrepancy_is_missing():
    cancer = pd.Series(["GBM", "LGG", "LGG", "LGG"])
    grade = pd.Series([None, "G2", "G3", "[Discrepancy]"])
    out = grade_from_cdr(cancer, grade)
    assert out.iloc[:3].tolist() == [4, 2, 3]
    assert pd.isna(out.iloc[3])


def test_molecular_class_strips_prefix_and_maps():
    out = molecular_class(pd.Series(["GBM_LGG.Codel", "GBM_LGG.Classic-like", "GBM_LGG.NA", None]), MAPPING)
    assert out.iloc[0] == "IDHmut-codel"
    assert out.iloc[1] == "IDHwt"
    assert out.iloc[2:].isna().all()


def test_one_primary_sample_per_patient():
    samples = pd.Index(["TCGA-AA-0001-01", "TCGA-AA-0001-02", "TCGA-AA-0002-11", "TCGA-AA-0003-01"])
    assert one_sample_per_patient(samples, "01").tolist() == ["TCGA-AA-0001-01", "TCGA-AA-0003-01"]


def test_expand_drops_dummies_collinear_with_baseline():
    # No codeleted tumours: the two class dummies sum to one, so one must go.
    df = add_model_columns(pd.DataFrame({
        "grade": [4, 4, 4, 4],
        "molecular_class": ["IDHwt", "IDHwt", "IDHmut-noncodel", "IDHwt"],
        "age_10y": [5.0, 6.0, 4.0, 7.0],
        "male": [1, 0, 1, 1],
    }))
    cols = expand(["age_10y", "male", "grade", "molecular_class"], df)
    assert "grade_4" not in cols and "grade_3" not in cols
    assert len({"idhmut_noncodel", "idhwt"} & set(cols)) == 1


def test_analysis_set_standardises_within_complete_cases():
    cohort = pd.DataFrame({
        "gpnmb_log2": [8.0, 9.0, 10.0, 11.0, np.nan],
        "os_months": [10.0, 0.0, 5.0, 20.0, 3.0],
        "os_event": [1, 0, 1, 0, 1],
        "age_10y": [5.0, 6.0, 4.0, 7.0, 6.0],
    })
    df = analysis_set(cohort, "OS", ["age_10y"])
    assert len(df) == 3  # missing GPNMB and zero follow-up removed
    assert abs(df["gpnmb_z"].mean()) < 1e-12
    assert abs(df["gpnmb_z"].std() - 1.0) < 1e-12
