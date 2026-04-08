import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import norm


def calculate_ate_ci(
    data: pd.DataFrame,
    alpha: float = 0.05,
) -> Tuple[
    float,
    float,
    float,
]:
    # split groups
    treatment = data[data["T"] == 1]["Y"]
    control = data[data["T"] == 0]["Y"]

    # sample sizes
    n1 = len(treatment)
    n0 = len(control)

    # means
    mean1 = treatment.mean()
    mean0 = control.mean()

    # ATE
    ate = mean1 - mean0

    # variances (unbiased)
    var1 = treatment.var(ddof=1)
    var0 = control.var(ddof=1)

    # standard error
    se = np.sqrt(var1 / n1 + var0 / n0)

    # z critical value
    z = norm.ppf(1 - alpha / 2)

    # confidence interval
    ci_lower = ate - z * se
    ci_upper = ate + z * se

    return ate, ci_lower, ci_upper


def calculate_ate_pvalue(
    data: pd.DataFrame,
) -> Tuple[
    float,
    float,
    float,
]:
    # split groups
    treatment = data[data["T"] == 1]["Y"]
    control = data[data["T"] == 0]["Y"]

    # sample sizes
    n1 = len(treatment)
    n0 = len(control)

    # means
    mean1 = treatment.mean()
    mean0 = control.mean()

    # ATE
    ate = mean1 - mean0

    # variances
    var1 = treatment.var(ddof=1)
    var0 = control.var(ddof=1)

    # standard error
    se = np.sqrt(var1 / n1 + var0 / n0)

    # t statistic (normal approx)
    t_stat = ate / se

    # two-sided p-value
    p_value = 2 * (1 - norm.cdf(abs(t_stat)))

    return ate, t_stat, p_value
