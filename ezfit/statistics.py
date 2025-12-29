"""Goodness-of-fit statistics module for ezfit.

This module provides standardized calculation of all goodness-of-fit statistics
for model fitting, including handling cases where measurement errors are not available.
"""

from typing import Any

import numpy as np
from scipy.stats import pearsonr

from ezfit.model import Model


def calculate_fit_statistics(
    model: Model,
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray | None,
    popt: np.ndarray,
    pcov: np.ndarray | None,
) -> dict[str, Any]:
    """Calculate comprehensive goodness-of-fit statistics.

    Parameters
    ----------
    model : Model
        The model object containing the function and parameters.
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    sigma : np.ndarray | None
        Error on dependent variable, or None if not provided.
    popt : np.ndarray
        Optimized parameter values.
    pcov : np.ndarray | None
        Covariance matrix, or None if not available.

    Returns
    -------
    dict[str, Any]
        Dictionary containing all calculated statistics:
        - residuals: np.ndarray - Residuals (ydata - model)
        - chi2: float - Chi-squared statistic (uses sigma=1 if errors not provided)
        - rchi2: float | None - Reduced chi-squared (None if dof <= 0)
        - cor: np.ndarray | None - Correlation matrix of parameters
        - r_squared: float - R² (coefficient of determination)
        - pearson_r: float - Pearson correlation coefficient
        - rmse: float - Root Mean Square Error
        - rmsd: float - Root Mean Square Deviation (same as RMSE)
        - bic: float - Bayesian Information Criterion
        - aic: float - Akaike Information Criterion
    """
    # Calculate residuals
    y_pred = model.func(xdata, *popt)
    residuals = ydata - y_pred

    # Count fitted parameters (excluding fixed ones)
    n_params_fit = len(popt) - sum(p.fixed for p in model.params.values())  # type: ignore
    n_data = len(xdata)

    # Calculate chi-squared statistics
    # If sigma is None, assume sigma=1 for all data points
    if sigma is not None and np.all(sigma > 0):
        safe_sigma = np.where(sigma == 0, 1e-10, sigma)
    else:
        # Use sigma=1 when errors are not provided
        safe_sigma = np.ones_like(residuals)

    chi2 = float(np.sum((residuals / safe_sigma) ** 2))
    dof = n_data - n_params_fit
    if dof > 0:
        rchi2 = chi2 / dof
    else:
        rchi2 = None

    # Calculate correlation matrix if covariance is available
    cor: np.ndarray | None = None
    if pcov is not None and not np.all(np.isnan(pcov)):
        diag_sqrt = np.sqrt(np.diag(pcov))
        if not np.any(diag_sqrt == 0):
            outer_prod = np.outer(diag_sqrt, diag_sqrt)
            cor = np.divide(
                pcov, outer_prod, out=np.full_like(pcov, np.nan), where=outer_prod != 0
            )
            np.fill_diagonal(cor, 1.0)

    # Calculate R² (coefficient of determination)
    ss_res = np.sum(residuals**2)  # Sum of squares of residuals
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)  # Total sum of squares
    if ss_tot > 0:
        r_squared = float(1.0 - (ss_res / ss_tot))
    else:
        # If all y values are the same, R² is undefined
        r_squared = np.nan if ss_res > 0 else 1.0

    # Calculate Pearson correlation coefficient
    try:
        pearson_r_val, _ = pearsonr(ydata, y_pred)
        pearson_r = float(pearson_r_val) if not np.isnan(pearson_r_val) else np.nan
    except Exception:
        pearson_r = np.nan

    # Calculate RMSE (Root Mean Square Error)
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # RMSD is the same as RMSE
    rmsd = rmse

    # Calculate BIC (Bayesian Information Criterion)
    # BIC = n * ln(SSE/n) + k * ln(n)
    # where n = number of data points, k = number of parameters, SSE = sum of squared errors
    sse = ss_res
    if n_data > 0 and sse > 0:
        bic = float(n_data * np.log(sse / n_data) + n_params_fit * np.log(n_data))
    else:
        bic = np.inf

    # Calculate AIC (Akaike Information Criterion)
    # AIC = n * ln(SSE/n) + 2 * k
    # where n = number of data points, k = number of parameters, SSE = sum of squared errors
    if n_data > 0 and sse > 0:
        aic = float(n_data * np.log(sse / n_data) + 2 * n_params_fit)
    else:
        aic = np.inf

    return {
        "residuals": residuals,
        "chi2": chi2,
        "rchi2": rchi2,
        "cor": cor,
        "r_squared": r_squared,
        "pearson_r": pearson_r,
        "rmse": rmse,
        "rmsd": rmsd,
        "bic": bic,
        "aic": aic,
    }
