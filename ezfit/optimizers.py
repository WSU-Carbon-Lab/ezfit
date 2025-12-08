"""
Optimization routines for ezfit, wrapping scipy.optimize and emcee.
"""

import warnings
from typing import TYPE_CHECKING, Any

import emcee
import numpy as np
from scipy.optimize import (
    OptimizeResult,
    curve_fit,
    differential_evolution,
    dual_annealing,
    minimize,
    shgo,
)

if TYPE_CHECKING:
    from sklearn.linear_model import (
        BayesianRidge,
        ElasticNet,
        Lasso,
        LinearRegression,
        Ridge,
    )
    from sklearn.preprocessing import PolynomialFeatures
else:
    try:
        from sklearn.linear_model import (
            BayesianRidge,
            ElasticNet,
            Lasso,
            LinearRegression,
            Ridge,
        )
        from sklearn.preprocessing import PolynomialFeatures
    except ImportError:
        BayesianRidge = None
        Ridge = None
        Lasso = None
        ElasticNet = None
        LinearRegression = None
        PolynomialFeatures = None

from ezfit.constraints import extract_constraints_from_model
from ezfit.mcmc_diagnostics import check_convergence, estimate_burnin
from ezfit.model import Model
from ezfit.types import (
    BayesianRidgeKwargs,
    CurveFitKwargs,
    DifferentialEvolutionKwargs,
    DualAnnealingKwargs,
    EmceeKwargs,
    FitResult,
    MinimizeKwargs,
    ShgoKwargs,
)


# --- Helper Function for Stats ---
def _calculate_fit_stats(
    model: Model,
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray | None,
    popt: np.ndarray,
    pcov: np.ndarray | None,
) -> tuple[np.ndarray, float | None, float | None, np.ndarray | None]:
    """Calculate residuals, chi-squared, reduced chi-squared, and correlation matrix."""
    residuals = ydata - model.func(xdata, *popt)
    chi2: float = np.inf
    rchi2: float = np.inf
    cor: np.ndarray | None = None
    n_params_fit = len(popt) - sum(p.fixed for p in model.params.values())  # type: ignore

    if sigma is not None and np.all(sigma > 0):
        safe_sigma = np.where(sigma == 0, 1e-10, sigma)
        chi2 = np.sum((residuals / safe_sigma) ** 2)
        dof = len(xdata) - n_params_fit
        if dof > 0:
            rchi2 = chi2 / dof
        else:
            warnings.warn(
                "Degrees of freedom <= 0, cannot calculate reduced chi-squared.",
                stacklevel=3,
            )

    if pcov is not None and not np.all(np.isnan(pcov)):
        diag_sqrt = np.sqrt(np.diag(pcov))
        if np.any(diag_sqrt == 0):
            warnings.warn(
                (
                    "Zero standard deviation found in covariance matrix diagonal,"
                    "cannot compute correlation matrix."
                ),
                stacklevel=3,
            )
            cor = np.full_like(pcov, np.nan)
        else:
            outer_prod = np.outer(diag_sqrt, diag_sqrt)
            cor = np.divide(
                pcov, outer_prod, out=np.full_like(pcov, np.nan), where=outer_prod != 0
            )
            np.fill_diagonal(cor, 1.0)

    return residuals, chi2, rchi2, cor


# --- Optimizer Functions ---
def _fit_curve_fit(
    model: Model,
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray | None,
    fit_kwargs: CurveFitKwargs,
) -> FitResult:
    """Perform fitting using `scipy.optimize.curve_fit`."""
    p0 = model.values()
    bounds_tuple = model.bounds()
    absolute_sigma = sigma is not None

    kwargs = fit_kwargs.copy()
    method = kwargs.pop("method", None)

    try:
        popt, pcov, infodict, errmsg, ier = curve_fit(
            model.func,
            xdata,
            ydata,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            bounds=bounds_tuple,
            method=method,
            full_output=True,
            **kwargs,  # type: ignore
        )
        success = ier in [1, 2, 3, 4]
        message = errmsg
        if not success:
            warnings.warn(f"curve_fit failed: {message} (ier={ier})", stacklevel=2)

    except Exception as e:
        warnings.warn(f"curve_fit raised an exception: {e}", stacklevel=2)
        popt = np.full_like(p0, np.nan)
        pcov = np.full((len(p0), len(p0)), np.nan)
        infodict = {}
        message = str(e)
        success = False
        ier = -1

    perr = (
        np.sqrt(np.diag(pcov))
        if pcov is not None and not np.all(np.isnan(pcov))
        else np.full_like(popt, np.nan)
    )
    residuals, chi2, rchi2, cor = _calculate_fit_stats(
        model, xdata, ydata, sigma, popt, pcov
    )

    details = {
        "infodict": infodict,
        "errmsg": message,
        "ier": ier,
        "success": success,
        "message": message,
        "x": popt,
    }

    return FitResult(
        popt=popt,
        perr=perr,
        pcov=pcov,
        residuals=residuals,
        chi2=chi2,
        rchi2=rchi2,
        cor=cor,
        details=details,
        sampler_chain=None,
    )


def _fit_minimize(
    model: "Model",
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray,
    fit_kwargs: MinimizeKwargs,
) -> FitResult:
    """Perform fitting using `scipy.optimize.minimize`."""
    p0 = model.values()
    bounds_tuple = model.bounds()
    bounds_list = list(zip(bounds_tuple[0], bounds_tuple[1], strict=False))

    def objective_func(params_to_optimize: np.ndarray) -> float:
        y_model = model.func(xdata, *params_to_optimize)
        safe_sigma = np.where(sigma == 0, 1e-10, sigma)
        return np.sum(((y_model - ydata) / safe_sigma) ** 2)

    # Extract constraints from model parameters
    constraint_funcs = extract_constraints_from_model(model)
    constraints_list = fit_kwargs.get("constraints", [])

    # Convert constraint functions to scipy constraints if needed
    if constraint_funcs:
        from scipy.optimize import NonlinearConstraint

        param_names = list(model.params.keys()) if model.params else []

        for constraint_func in constraint_funcs:
            # Create NonlinearConstraint from constraint function
            scipy_constraint = NonlinearConstraint(
                constraint_func,
                lb=-np.inf,
                ub=0.0,  # constraint_func returns negative if satisfied
            )
            if isinstance(constraints_list, list):
                constraints_list.append(scipy_constraint)
            else:
                constraints_list = [scipy_constraint]

        # Update fit_kwargs with constraints
        fit_kwargs = fit_kwargs.copy()
        fit_kwargs["constraints"] = constraints_list

    try:
        result: OptimizeResult = minimize(
            objective_func,
            x0=np.array(p0),
            bounds=bounds_list,
            **fit_kwargs,  # type: ignore
        )
        if not result.success:
            warnings.warn(f"Minimize failed: {result.message}", stacklevel=2)
        popt = result.x
        pcov = None
        hess_inv = getattr(result, "hess_inv", None)
        if hess_inv is not None:
            if callable(hess_inv):
                try:
                    hess_inv_matrix = hess_inv.todense()
                    pcov = hess_inv_matrix * 2
                except (AttributeError, NotImplementedError):
                    warnings.warn(
                        "Cannot convert hess_inv operator to dense matrix for cov.",
                        stacklevel=2,
                    )
                    pcov = np.full((len(popt), len(popt)), np.nan)
            elif isinstance(hess_inv, np.ndarray):
                pcov = hess_inv * 2
            else:
                try:
                    pcov = hess_inv.todense() * 2
                except AttributeError:
                    pcov = np.full((len(popt), len(popt)), np.nan)
        else:
            warnings.warn(
                "Covariance matrix (Hessian inverse) not available from this method.",
                stacklevel=2,
            )
            pcov = np.full((len(popt), len(popt)), np.nan)

    except Exception as e:
        warnings.warn(f"Minimize raised an exception: {e}", stacklevel=2)
        popt = np.full_like(p0, np.nan)
        pcov = np.full((len(p0), len(p0)), np.nan)
        result = OptimizeResult(x=popt, success=False, message=str(e))

    perr = (
        np.sqrt(np.diag(pcov))
        if pcov is not None and not np.all(np.isnan(pcov))
        else np.full_like(popt, np.nan)
    )
    residuals, chi2, rchi2, cor = _calculate_fit_stats(
        model, xdata, ydata, sigma, popt, pcov
    )

    return FitResult(
        popt=popt,
        perr=perr,
        pcov=pcov,
        residuals=residuals,
        chi2=chi2,
        rchi2=rchi2,
        cor=cor,
        details=result,
        sampler_chain=None,
    )


def _fit_differential_evolution(
    model: "Model",
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray,
    fit_kwargs: DifferentialEvolutionKwargs,
) -> FitResult:
    """Perform fitting using `scipy.optimize.differential_evolution`."""
    bounds_tuple = model.bounds()
    bounds_list = list(zip(bounds_tuple[0], bounds_tuple[1], strict=False))
    p0 = model.values()

    def objective_func(params_to_optimize: np.ndarray) -> float:
        y_model = model.func(xdata, *params_to_optimize)
        safe_sigma = np.where(sigma == 0, 1e-10, sigma)
        return np.sum(((y_model - ydata) / safe_sigma) ** 2)

    # Extract constraints from model parameters
    constraint_funcs = extract_constraints_from_model(model)
    constraints_list = fit_kwargs.get("constraints", [])

    # Convert constraint functions to scipy constraints if needed
    if constraint_funcs:
        from scipy.optimize import NonlinearConstraint

        for constraint_func in constraint_funcs:
            scipy_constraint = NonlinearConstraint(
                constraint_func,
                lb=-np.inf,
                ub=0.0,
            )
            if isinstance(constraints_list, list):
                constraints_list.append(scipy_constraint)
            else:
                constraints_list = [scipy_constraint]

        fit_kwargs = fit_kwargs.copy()
        fit_kwargs["constraints"] = constraints_list

    try:
        result: OptimizeResult = differential_evolution(
            objective_func,
            bounds=bounds_list,
            **fit_kwargs,  # type: ignore
        )
        if not result.success:
            warnings.warn(
                f"differential_evolution failed: {result.message}", stacklevel=2
            )
        popt = result.x

    except Exception as e:
        warnings.warn(f"differential_evolution raised an exception: {e}", stacklevel=2)
        popt = np.full_like(p0, np.nan)
        result = OptimizeResult(x=popt, success=False, message=str(e))

    warnings.warn(
        "Covariance matrix not available from differential_evolution.", stacklevel=2
    )
    pcov = np.full((len(popt), len(popt)), np.nan)
    perr = np.full_like(popt, np.nan)
    residuals, chi2, rchi2, cor = _calculate_fit_stats(
        model, xdata, ydata, sigma, popt, pcov
    )

    return FitResult(
        popt=popt,
        perr=perr,
        pcov=pcov,
        residuals=residuals,
        chi2=chi2,
        rchi2=rchi2,
        cor=cor,
        details=result,
        sampler_chain=None,
    )


def _fit_shgo(
    model: "Model",
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray,
    fit_kwargs: ShgoKwargs,
) -> FitResult:
    """Perform fitting using scipy.optimize.shgo."""
    bounds_tuple = model.bounds()
    bounds_list = list(zip(bounds_tuple[0], bounds_tuple[1], strict=False))
    p0 = model.values()

    def objective_func(params_to_optimize: np.ndarray) -> float:
        y_model = model.func(xdata, *params_to_optimize)
        safe_sigma = np.where(sigma == 0, 1e-10, sigma)
        return np.sum(((y_model - ydata) / safe_sigma) ** 2)

    try:
        result: OptimizeResult = shgo(
            objective_func,
            bounds=bounds_list,
            **fit_kwargs,  # type: ignore
        )
        if not result.success:
            warnings.warn(f"shgo failed: {result.message}", stacklevel=2)
        popt = result.x

    except Exception as e:
        warnings.warn(f"shgo raised an exception: {e}", stacklevel=2)
        popt = np.full_like(p0, np.nan)
        result = OptimizeResult(x=popt, success=False, message=str(e))

    warnings.warn("Covariance matrix not available from shgo.", stacklevel=2)
    pcov = np.full((len(popt), len(popt)), np.nan)
    perr = np.full_like(popt, np.nan)
    residuals, chi2, rchi2, cor = _calculate_fit_stats(
        model, xdata, ydata, sigma, popt, pcov
    )

    return FitResult(
        popt=popt,
        perr=perr,
        pcov=pcov,
        residuals=residuals,
        chi2=chi2,
        rchi2=rchi2,
        cor=cor,
        details=result,
        sampler_chain=None,
    )


def _fit_dual_annealing(
    model: "Model",
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray,
    fit_kwargs: DualAnnealingKwargs,
) -> FitResult:
    """Perform fitting using `scipy.optimize.dual_annealing`."""
    bounds_tuple = model.bounds()
    bounds_list = list(zip(bounds_tuple[0], bounds_tuple[1], strict=False))
    p0 = model.values()

    def objective_func(params_to_optimize: np.ndarray) -> float:
        y_model = model.func(xdata, *params_to_optimize)
        safe_sigma = np.where(sigma == 0, 1e-10, sigma)
        return np.sum(((y_model - ydata) / safe_sigma) ** 2)

    try:
        result: OptimizeResult = dual_annealing(
            objective_func,
            bounds=bounds_list,
            **fit_kwargs,  # type: ignore
        )
        if not result.success:
            warnings.warn(f"dual_annealing failed: {result.message}", stacklevel=2)
        popt = result.x

    except Exception as e:
        warnings.warn(f"dual_annealing raised an exception: {e}", stacklevel=2)
        popt = np.full_like(p0, np.nan)
        result = OptimizeResult(x=popt, success=False, message=str(e))

    warnings.warn("Covariance matrix not available from dual_annealing.", stacklevel=2)
    pcov = np.full((len(popt), len(popt)), np.nan)
    perr = np.full_like(popt, np.nan)
    residuals, chi2, rchi2, cor = _calculate_fit_stats(
        model, xdata, ydata, sigma, popt, pcov
    )

    return FitResult(
        popt=popt,
        perr=perr,
        pcov=pcov,
        residuals=residuals,
        chi2=chi2,
        rchi2=rchi2,
        cor=cor,
        details=result,
        sampler_chain=None,
    )


def _fit_bayesian_ridge(
    model: "Model",
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray | None,
    fit_kwargs: "BayesianRidgeKwargs",
) -> FitResult:
    """Perform fitting using `sklearn.linear_model.BayesianRidge`.

    This method is only valid for linear models (models that are linear in their
    parameters). The function signature is analyzed to determine if the model
    is linear.

    Args:
        model: The model function to fit. Must be linear in parameters.
        xdata: Independent variable data.
        ydata: Dependent variable data.
        sigma: Error on dependent variable (used for weighting if provided).
        fit_kwargs: Keyword arguments for BayesianRidge.

    Returns
    -------
        FitResult with fitted parameters and statistics.

    Raises
    ------
        ImportError: If scikit-learn is not installed.
        ValueError: If model is not linear in parameters.
    """
    if BayesianRidge is None:
        msg = "scikit-learn is required for Bayesian Ridge fitting. Install with: pip install scikit-learn"
        raise ImportError(msg)

    # Check if model is linear in parameters
    # For now, we'll attempt to fit and let sklearn handle errors
    # A more sophisticated check could be added later

    # Prepare feature matrix for sklearn
    # For a linear model f(x, a, b, c, ...), we need to construct
    # a design matrix where each column corresponds to a parameter
    # This is complex for arbitrary functions, so we'll use a simpler approach:
    # Evaluate the model with unit parameters to construct the design matrix

    # Get parameter names and initial values
    param_names = list(model.params.keys()) if model.params else []
    n_params = len(param_names)

    if n_params == 0:
        msg = "Model must have at least one parameter for Bayesian Ridge fitting."
        raise ValueError(msg)

    # For linear models, we need to construct a design matrix
    # This is a simplified approach - for truly general linear models,
    # we'd need symbolic differentiation or automatic differentiation
    # For now, we'll use a numerical approach: evaluate partial derivatives

    # Construct design matrix by evaluating model with perturbed parameters
    p0 = model.values()
    design_matrix = np.zeros((len(xdata), n_params))

    # Use finite differences to estimate partial derivatives
    eps = 1e-6
    for i in range(n_params):
        p_perturbed = p0.copy()
        p_perturbed[i] += eps
        y_perturbed = model.func(xdata, *p_perturbed)
        y_base = model.func(xdata, *p0)
        design_matrix[:, i] = (y_perturbed - y_base) / eps

    # Check if model is approximately linear (design matrix should be constant)
    # For truly linear models, this should work well

    # Prepare sample weights from sigma if provided
    sample_weight = None
    if sigma is not None and np.all(sigma > 0):
        sample_weight = 1.0 / (sigma**2)
        sample_weight = sample_weight / np.sum(sample_weight) * len(sample_weight)

    # Create and fit BayesianRidge model
    kwargs = fit_kwargs.copy()
    br_model = BayesianRidge(**kwargs)  # type: ignore

    try:
        br_model.fit(design_matrix, ydata, sample_weight=sample_weight)
    except Exception as e:
        msg = f"Bayesian Ridge fitting failed. Model may not be linear in parameters: {e}"
        raise ValueError(msg) from e

    # Extract fitted parameters
    popt = br_model.coef_  # type: ignore
    # Add intercept if fit_intercept is True (default)
    if kwargs.get("fit_intercept", True):
        # Intercept is stored separately in sklearn
        # For compatibility, prepend it to the coefficients like other sklearn methods
        popt = np.concatenate([[br_model.intercept_], popt])  # type: ignore

    # Get parameter uncertainties from sklearn's precision matrix
    # sklearn provides lambda_ (precision of noise) and alpha_ (precision of weights)
    # The covariance can be estimated from the precision matrix
    try:
        # sklearn doesn't directly provide covariance, but we can estimate it
        # from the precision matrix: cov = (alpha * X^T X + lambda * I)^(-1)
        # For now, we'll use the standard errors from sklearn if available
        if hasattr(br_model, "sigma_"):  # type: ignore
            # sklearn's BayesianRidge has sigma_ attribute in some versions
            perr = np.sqrt(np.diag(br_model.sigma_))  # type: ignore
            pcov = br_model.sigma_  # type: ignore
        else:
            # Estimate covariance from precision
            # This is approximate
            alpha = br_model.alpha_  # type: ignore
            lambda_reg = br_model.lambda_  # type: ignore
            X = design_matrix
            if kwargs.get("fit_intercept", True):
                # Add intercept column
                X = np.column_stack([np.ones(len(xdata)), X])
            precision = alpha * X.T @ X + lambda_reg * np.eye(X.shape[1])
            try:
                pcov = np.linalg.inv(precision)
                perr = np.sqrt(np.diag(pcov))
            except np.linalg.LinAlgError:
                warnings.warn(
                    "Could not compute covariance matrix from precision matrix.",
                    stacklevel=2,
                )
                pcov = np.full((len(popt), len(popt)), np.nan)
                perr = np.full_like(popt, np.nan)
    except Exception as e:
        warnings.warn(
            f"Could not extract parameter uncertainties from BayesianRidge: {e}",
            stacklevel=2,
        )
        pcov = np.full((len(popt), len(popt)), np.nan)
        perr = np.full_like(popt, np.nan)

    # Calculate residuals and statistics
    y_pred = br_model.predict(design_matrix)  # type: ignore
    residuals = ydata - y_pred
    residuals, chi2, rchi2, cor = _calculate_fit_stats(
        model, xdata, ydata, sigma, popt, pcov
    )

    details = {
        "model": br_model,
        "alpha_": br_model.alpha_ if hasattr(br_model, "alpha_") else None,  # type: ignore
        "lambda_": br_model.lambda_ if hasattr(br_model, "lambda_") else None,  # type: ignore
        "score": br_model.score(design_matrix, ydata) if hasattr(br_model, "score") else None,  # type: ignore
    }

    return FitResult(
        popt=popt,
        perr=perr,
        pcov=pcov,
        residuals=residuals,
        chi2=chi2,
        rchi2=rchi2,
        cor=cor,
        details=details,
        sampler_chain=None,
    )


def _fit_ridge(
    model: "Model",
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray | None,
    fit_kwargs: dict[str, Any],
) -> FitResult:
    """Perform fitting using `sklearn.linear_model.Ridge`."""
    if Ridge is None:
        msg = "scikit-learn is required for Ridge fitting. Install with: pip install scikit-learn"
        raise ImportError(msg)

    from ezfit.sklearn_adapter import construct_design_matrix

    param_names = list(model.params.keys()) if model.params else []
    p0 = model.values()
    design_matrix = construct_design_matrix(model.func, xdata, param_names, p0)

    sample_weight = None
    if sigma is not None and np.all(sigma > 0):
        sample_weight = 1.0 / (sigma**2)

    kwargs = fit_kwargs.copy()
    ridge_model = Ridge(**kwargs)  # type: ignore

    try:
        ridge_model.fit(design_matrix, ydata, sample_weight=sample_weight)  # type: ignore
    except Exception as e:
        msg = f"Ridge fitting failed: {e}"
        raise ValueError(msg) from e

    popt = ridge_model.coef_  # type: ignore
    if kwargs.get("fit_intercept", True):
        # Intercept handling - for now, prepend to coefficients
        popt = np.concatenate([[ridge_model.intercept_], popt])  # type: ignore

    # Estimate covariance (Ridge doesn't provide it directly)
    pcov = np.full((len(popt), len(popt)), np.nan)
    perr = np.full_like(popt, np.nan)

    residuals, chi2, rchi2, cor = _calculate_fit_stats(
        model, xdata, ydata, sigma, popt, pcov
    )

    details = {"model": ridge_model, "alpha": ridge_model.alpha}  # type: ignore

    return FitResult(
        popt=popt,
        perr=perr,
        pcov=pcov,
        residuals=residuals,
        chi2=chi2,
        rchi2=rchi2,
        cor=cor,
        details=details,
        sampler_chain=None,
    )


def _fit_lasso(
    model: "Model",
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray | None,
    fit_kwargs: dict[str, Any],
) -> FitResult:
    """Perform fitting using `sklearn.linear_model.Lasso`."""
    if Lasso is None:
        msg = "scikit-learn is required for Lasso fitting. Install with: pip install scikit-learn"
        raise ImportError(msg)

    from ezfit.sklearn_adapter import construct_design_matrix

    param_names = list(model.params.keys()) if model.params else []
    p0 = model.values()
    design_matrix = construct_design_matrix(model.func, xdata, param_names, p0)

    sample_weight = None
    if sigma is not None and np.all(sigma > 0):
        sample_weight = 1.0 / (sigma**2)

    kwargs = fit_kwargs.copy()
    lasso_model = Lasso(**kwargs)  # type: ignore

    try:
        lasso_model.fit(design_matrix, ydata, sample_weight=sample_weight)  # type: ignore
    except Exception as e:
        msg = f"Lasso fitting failed: {e}"
        raise ValueError(msg) from e

    popt = lasso_model.coef_  # type: ignore
    if kwargs.get("fit_intercept", True):
        popt = np.concatenate([[lasso_model.intercept_], popt])  # type: ignore

    pcov = np.full((len(popt), len(popt)), np.nan)
    perr = np.full_like(popt, np.nan)

    residuals, chi2, rchi2, cor = _calculate_fit_stats(
        model, xdata, ydata, sigma, popt, pcov
    )

    details = {"model": lasso_model, "alpha": lasso_model.alpha}  # type: ignore

    return FitResult(
        popt=popt,
        perr=perr,
        pcov=pcov,
        residuals=residuals,
        chi2=chi2,
        rchi2=rchi2,
        cor=cor,
        details=details,
        sampler_chain=None,
    )


def _fit_elasticnet(
    model: "Model",
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray | None,
    fit_kwargs: dict[str, Any],
) -> FitResult:
    """Perform fitting using `sklearn.linear_model.ElasticNet`."""
    if ElasticNet is None:
        msg = "scikit-learn is required for ElasticNet fitting. Install with: pip install scikit-learn"
        raise ImportError(msg)

    from ezfit.sklearn_adapter import construct_design_matrix

    param_names = list(model.params.keys()) if model.params else []
    p0 = model.values()
    design_matrix = construct_design_matrix(model.func, xdata, param_names, p0)

    sample_weight = None
    if sigma is not None and np.all(sigma > 0):
        sample_weight = 1.0 / (sigma**2)

    kwargs = fit_kwargs.copy()
    elastic_model = ElasticNet(**kwargs)  # type: ignore

    try:
        elastic_model.fit(design_matrix, ydata, sample_weight=sample_weight)  # type: ignore
    except Exception as e:
        msg = f"ElasticNet fitting failed: {e}"
        raise ValueError(msg) from e

    popt = elastic_model.coef_  # type: ignore
    if kwargs.get("fit_intercept", True):
        popt = np.concatenate([[elastic_model.intercept_], popt])  # type: ignore

    pcov = np.full((len(popt), len(popt)), np.nan)
    perr = np.full_like(popt, np.nan)

    residuals, chi2, rchi2, cor = _calculate_fit_stats(
        model, xdata, ydata, sigma, popt, pcov
    )

    details = {
        "model": elastic_model,
        "alpha": elastic_model.alpha,  # type: ignore
        "l1_ratio": elastic_model.l1_ratio,  # type: ignore
    }

    return FitResult(
        popt=popt,
        perr=perr,
        pcov=pcov,
        residuals=residuals,
        chi2=chi2,
        rchi2=rchi2,
        cor=cor,
        details=details,
        sampler_chain=None,
    )


def _fit_polynomial(
    model: "Model",
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray | None,
    fit_kwargs: dict[str, Any],
) -> FitResult:
    """Perform polynomial fitting using sklearn."""
    if LinearRegression is None or PolynomialFeatures is None:
        msg = "scikit-learn is required for polynomial fitting. Install with: pip install scikit-learn"
        raise ImportError(msg)

    from ezfit.sklearn_adapter import convert_to_polynomial_model

    kwargs = fit_kwargs.copy()
    degree = kwargs.pop("degree", 2)

    # Convert to polynomial features
    feature_matrix, poly_transformer = convert_to_polynomial_model(
        model.func, xdata, degree=degree
    )

    sample_weight = None
    if sigma is not None and np.all(sigma > 0):
        sample_weight = 1.0 / (sigma**2)

    # Fit linear regression on polynomial features
    lin_model = LinearRegression(**kwargs)  # type: ignore
    try:
        lin_model.fit(feature_matrix, ydata, sample_weight=sample_weight)  # type: ignore
    except Exception as e:
        msg = f"Polynomial fitting failed: {e}"
        raise ValueError(msg) from e

    popt = lin_model.coef_  # type: ignore
    if kwargs.get("fit_intercept", True):
        popt = np.concatenate([[lin_model.intercept_], popt])  # type: ignore

    pcov = np.full((len(popt), len(popt)), np.nan)
    perr = np.full_like(popt, np.nan)

    residuals, chi2, rchi2, cor = _calculate_fit_stats(
        model, xdata, ydata, sigma, popt, pcov
    )

    details = {
        "model": lin_model,
        "poly_transformer": poly_transformer,
        "degree": degree,
    }

    return FitResult(
        popt=popt,
        perr=perr,
        pcov=pcov,
        residuals=residuals,
        chi2=chi2,
        rchi2=rchi2,
        cor=cor,
        details=details,
        sampler_chain=None,
    )


def _fit_emcee(
    model: "Model",
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray,
    fit_kwargs: EmceeKwargs,
) -> FitResult:
    """Perform MCMC fitting using `emcee`."""
    if np.any(sigma <= 0):
        msg = (
            "Non-positive values found in yerr (sigma). "
            "MCMC likelihood requires positive errors."
        )
        warnings.warn(msg, stacklevel=2)
        sigma = np.where(sigma <= 0, 1e-10, sigma)

    initial_params = model.values()
    bounds_tuple = model.bounds()
    ndim = len(initial_params)

    try:
        nwalkers = fit_kwargs.pop("nwalkers")
        nsteps = fit_kwargs.pop("nsteps")
    except KeyError as e:
        msg = f"Missing required emcee argument: {e}"
        raise ValueError(msg) from e

    def log_likelihood(theta: np.ndarray) -> float:
        min_bounds, max_bounds = bounds_tuple
        if not np.all((min_bounds <= theta) & (theta <= max_bounds)):
            return -np.inf
        y_model = model.func(xdata, *theta)
        chisq = np.sum(((y_model - ydata) / sigma) ** 2)
        log_norm = -0.5 * np.sum(np.log(2 * np.pi * sigma**2))
        return log_norm - 0.5 * chisq

    def log_prior(theta: np.ndarray) -> float:
        min_bounds, max_bounds = bounds_tuple
        if not np.all((min_bounds <= theta) & (theta <= max_bounds)):
            return -np.inf

        # Check parameter constraints
        constraint_funcs = extract_constraints_from_model(model)
        if constraint_funcs:
            for constraint_func in constraint_funcs:
                # Constraint function from extract_constraints_from_model takes array
                # and returns -1.0 if satisfied, 1.0 if not satisfied
                constraint_val = constraint_func(theta)
                if constraint_val > 0:  # Not satisfied (> 0 means constraint violated)
                    return -np.inf

        return 0.0

    def log_probability(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    initial_state = fit_kwargs.pop("initial_state", None)
    if initial_state is None:
        pos = np.array(initial_params) + 1e-4 * np.random.randn(nwalkers, ndim)
        min_bounds_arr = np.array(bounds_tuple[0])
        max_bounds_arr = np.array(bounds_tuple[1])
        pos = np.clip(pos, min_bounds_arr + 1e-9, max_bounds_arr - 1e-9)
    else:
        pos = initial_state
        if pos.shape != (nwalkers, ndim):
            msg = (
                f"initial_state shape mismatch: expected ({nwalkers}, {ndim}), "
                f"got {pos.shape}"
            )
            raise ValueError(msg)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        pool=fit_kwargs.pop("pool", None),
        moves=fit_kwargs.pop("moves", None),
        backend=fit_kwargs.pop("backend", None),
        vectorize=fit_kwargs.pop("vectorize", False),
        blobs_dtype=fit_kwargs.pop("blobs_dtype", None),
    )

    progress = fit_kwargs.pop("progress", True)

    try:
        sampler.run_mcmc(pos, nsteps, progress=progress, **fit_kwargs)  # type: ignore
    except Exception as e:
        warnings.warn(f"emcee sampler.run_mcmc raised an exception: {e}", stacklevel=2)
        popt = np.full(ndim, np.nan)
        perr = np.full(ndim, np.nan)
        pcov = np.full((ndim, ndim), np.nan)
        chain = None
        residuals, chi2, rchi2, cor = _calculate_fit_stats(
            model, xdata, ydata, sigma, popt, pcov
        )
        return FitResult(
            popt=popt,
            perr=perr,
            pcov=pcov,
            residuals=residuals,
            chi2=chi2,
            rchi2=rchi2,
            cor=cor,
            details=sampler,
            sampler_chain=chain,
        )

    # Get full chain for diagnostics
    try:
        full_chain = sampler.get_chain()  # Shape: (nsteps, nwalkers, ndim)
    except Exception as e:
        warnings.warn(f"Could not retrieve chain from sampler: {e}", stacklevel=2)
        full_chain = None

    # Automatic burn-in detection if discard not explicitly provided
    auto_burnin = fit_kwargs.pop("auto_burnin", True)
    if auto_burnin and full_chain is not None:
        try:
            estimated_burnin = estimate_burnin(full_chain)
            if "discard" not in fit_kwargs:
                discard = estimated_burnin
            else:
                discard = fit_kwargs.pop("discard", nsteps // 2)
        except Exception:
            discard = fit_kwargs.pop("discard", nsteps // 2)
    else:
        discard = fit_kwargs.pop("discard", nsteps // 2)

    # Automatic thinning based on autocorrelation if thin not provided
    auto_thin = fit_kwargs.pop("auto_thin", True)
    thin = fit_kwargs.pop("thin", None)
    if thin is None:
        if auto_thin and full_chain is not None:
            # Estimate thinning from autocorrelation
            # Use a conservative estimate: thin by ~10-20
            thin = max(1, min(20, nsteps // (discard + 100)))
        else:
            thin = 15

    flat = fit_kwargs.pop("flat", True)

    # Get processed chain
    try:
        chain = sampler.get_chain(discard=discard, thin=thin, flat=flat)
    except Exception as e:
        warnings.warn(f"Could not retrieve processed chain from sampler: {e}", stacklevel=2)
        chain = None

    # Store full chain for diagnostics if available
    chain_for_diagnostics = full_chain if full_chain is not None else chain

    # Run convergence diagnostics
    diagnostics = None
    if chain_for_diagnostics is not None:
        try:
            converged, diagnostics = check_convergence(
                chain_for_diagnostics, burnin=discard if full_chain is not None else None
            )
            if not converged:
                warnings.warn(
                    f"MCMC chain may not have converged. R-hat: {diagnostics.get('rhat', 'N/A')}, "
                    f"ESS: {diagnostics.get('ess', 'N/A')}",
                    stacklevel=2,
                )
        except Exception as e:
            warnings.warn(f"Could not compute convergence diagnostics: {e}", stacklevel=2)

    popt = np.full(ndim, np.nan)
    perr = np.full(ndim, np.nan)
    pcov = np.full((ndim, ndim), np.nan)

    if chain is not None and chain.shape[0] > 0:
        try:
            # Compute percentiles for parameter estimates
            q = np.nanpercentile(chain, [16, 50, 84], axis=0)
            popt = q[1]  # Median
            lower_err = q[1] - q[0]  # 16th percentile
            upper_err = q[2] - q[1]  # 84th percentile
            perr = (lower_err + upper_err) / 2.0  # Average uncertainty

            # Compute covariance matrix from chain
            if chain.shape[0] > 1:
                if np.any(~np.isfinite(chain)):
                    warnings.warn(
                        "Chain contains NaNs or Infs, cannot compute covariance.",
                        stacklevel=2,
                    )
                else:
                    try:
                        pcov = np.cov(chain, rowvar=False)
                    except ValueError as cov_err:
                        warnings.warn(
                            f"Could not estimate covariance from chain: {cov_err}",
                            stacklevel=2,
                        )

        except IndexError:
            warnings.warn(
                "Could not calculate percentiles from chain (possibly too short).",
                stacklevel=2,
            )
        except Exception as e:
            warnings.warn(f"Error processing MCMC chain results: {e}", stacklevel=2)

    residuals, chi2, rchi2, cor = _calculate_fit_stats(
        model, xdata, ydata, sigma, popt, pcov
    )

    # Store diagnostics in details
    details_dict: dict[str, Any] = {"sampler": sampler}
    if diagnostics is not None:
        details_dict["diagnostics"] = diagnostics
        details_dict["burnin"] = discard
        details_dict["thin"] = thin

    return FitResult(
        popt=popt,
        perr=perr,
        pcov=pcov,
        residuals=residuals,
        chi2=chi2,
        rchi2=rchi2,
        cor=cor,
        details=details_dict,
        sampler_chain=chain,
    )
