
import pandas as pd
import numpy as np

# ── 1. CLEAN DATA ────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix impossible values and cap outliers.
    Rule: we cap, not delete — dropping rows loses real borrower signal.
    """
    df = df.copy()

    # RevolvingUtilization is a ratio (0–1). Anything above 1 is a data error.
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=1.0)

    # DebtRatio and MonthlyIncome have extreme right tails — cap at 99th pct.
    for col in ["DebtRatio", "MonthlyIncome"]:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    # Age of 0 is impossible. Replace with NaN then fill with median.
    df["age"] = df["age"].replace(0, np.nan)
    df["age"] = df["age"].fillna(df["age"].median())

    return df


# ── 2. IMPUTE MISSING VALUES ─────────────────────────────────────────────────
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute before engineering so downstream features are never built on NaNs.
    Key: create the missingness FLAG before filling — the flag is itself a feature.
    """
    df = df.copy()

    # income_missing: the fact that income is missing may predict default.
    # Create this flag BEFORE imputing, otherwise you lose the signal.
    df["income_missing"] = df["MonthlyIncome"].isna().astype(int)

    # Smarter imputation: median income grouped by age bracket.
    # A 25-year-old and a 55-year-old have very different income distributions.
    df["age_bracket"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"]
    )
    group_medians = df.groupby("age_bracket", observed=True)["MonthlyIncome"].transform("median")
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(group_medians)
    # Fallback: if any age bracket is entirely NaN, use global median
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

    # NumberOfDependents: global median is fine (no strong age interaction here)
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
        df["NumberOfDependents"].median()
    )

    # Drop the helper column — it was only needed for grouped imputation
    df = df.drop(columns=["age_bracket"])

    return df


# ── 3. ENGINEER FEATURES ─────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new columns that encode domain knowledge.
    Every feature here has a business reason — not just math.
    """
    df = df.copy()

    # DebtToIncomeRatio: the single most important metric in credit underwriting.
    # +1 avoids division by zero for zero-income rows.
    df["DebtToIncomeRatio"] = df["DebtRatio"] / (df["MonthlyIncome"] + 1)

    # IncomePerDependent: financial stress — how much income supports each dependent.
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

    # TotalLatePayments: combines all three delinquency columns into one risk score.
    late_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    df["TotalLatePayments"] = df[late_cols].sum(axis=1)

    # LatePaymentRatio: late payments relative to total credit exposure.
    df["LatePaymentRatio"] = df["TotalLatePayments"] / (
        df["NumberOfOpenCreditLinesAndLoans"] + 1
    )

    # UtilizationBin: captures nonlinear risk — 0% and 100% are both risky.
    df["UtilizationBin"] = pd.cut(
        df["RevolvingUtilizationOfUnsecuredLines"],
        bins=[-0.01, 0.01, 0.3, 0.7, 1.01],
        labels=["zero", "low", "medium", "high"]
    ).astype(str)

    # AgeGroup: age has nonlinear risk patterns.
    df["AgeGroup"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"]
    ).astype(str)

    # Log transforms for the logistic regression meta-learner in Phase 6.
    # np.log1p = log(x + 1) — handles zeros cleanly.
    for col in ["MonthlyIncome", "DebtRatio", "RevolvingUtilizationOfUnsecuredLines"]:
        df[f"log_{col}"] = np.log1p(df[col])

    return df


# ── 4. MASTER PIPELINE ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The one function every notebook calls.
    Order matters: clean → impute → engineer.
    """
    df = clean_data(df)
    df = impute_missing(df)
    df = engineer_features(df)
    return df
