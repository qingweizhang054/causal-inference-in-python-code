import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.linear_model import LogisticRegression, LinearRegression


def ipw(df: pd.DataFrame, ps_formula: str, T: str, Y: str) -> float:
    """
    Estimate Average Treatment Effect (ATE) using Inverse Propensity Weighting (IPW).

    Parameters:
        df : pd.DataFrame
            Input dataset containing treatment, outcome, and covariates.
        ps_formula : str
            Patsy formula specifying covariates for propensity score model.
        T : str
            Column name for treatment variable (binary: 0 or 1).
        Y : str
            Column name for outcome variable.

    Returns:
        float
            Estimated ATE.
    """

    # Build design matrix for propensity score model
    X = dmatrix(ps_formula, df, return_type="dataframe")

    # Extract treatment and outcome arrays
    t = df[T].values
    y = df[Y].values

    # Fit logistic regression model for propensity scores
    ps_model = LogisticRegression(penalty=None, max_iter=1000)
    ps_model.fit(X, t)

    # Predict propensity scores P(T=1 | X)
    ps = ps_model.predict_proba(X)[:, 1]

    # Clip propensity scores to avoid division by zero
    eps = 1e-6
    ps = np.clip(ps, eps, 1 - eps)

    # Compute IPW weights
    weights = (t - ps) / (ps * (1 - ps))

    # Estimate ATE
    ate = np.mean(weights * y)

    return float(ate)


def doubly_robust(df: pd.DataFrame, formula: str, T: str, Y: str) -> float:
    """
    Estimate Average Treatment Effect (ATE) using Doubly Robust (DR) estimator.

    Parameters:
        df : pd.DataFrame
            Input dataset containing treatment, outcome, and covariates.
        formula : str
            Patsy formula specifying covariates for both models.
        T : str
            Column name for treatment variable (binary: 0 or 1).
        Y : str
            Column name for outcome variable.

    Returns:
        float
            Estimated ATE.
    """

    # Build design matrix
    X = dmatrix(formula, df, return_type="dataframe")

    # Extract treatment and outcome
    t = df[T].values
    y = df[Y].values

    # Fit propensity score model
    ps_model = LogisticRegression(penalty=None, max_iter=1000)
    ps_model.fit(X, t)

    # Predict propensity scores
    ps = ps_model.predict_proba(X)[:, 1]

    # Clip to avoid instability
    eps = 1e-6
    ps = np.clip(ps, eps, 1 - eps)

    # Fit outcome model for treated group: μ1(X)
    model_treated = LinearRegression()
    model_treated.fit(X[t == 1], y[t == 1])

    # Fit outcome model for control group: μ0(X)
    model_control = LinearRegression()
    model_control.fit(X[t == 0], y[t == 0])

    # Predict outcomes for all samples
    mu1 = model_treated.predict(X)
    mu0 = model_control.predict(X)

    # Compute doubly robust components
    term_treated = t * (y - mu1) / ps + mu1
    term_control = (1 - t) * (y - mu0) / (1 - ps) + mu0

    # Estimate ATE
    ate = np.mean(term_treated - term_control)

    return float(ate)
