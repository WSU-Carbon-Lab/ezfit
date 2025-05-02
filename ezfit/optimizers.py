"""
Optimization routines for ezfit, wrapping scipy.optimize and emcee.
"""

import warnings

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

from ezfit.model import Model
from ezfit.types import (
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
        if np.all((min_bounds <= theta) & (theta <= max_bounds)):
            return 0.0
        return -np.inf

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

    discard = fit_kwargs.pop("discard", nsteps // 2)
    thin = fit_kwargs.pop("thin", 15)
    flat = fit_kwargs.pop("flat", True)

    try:
        chain = sampler.get_chain(discard=discard, thin=thin, flat=flat)
    except Exception as e:
        warnings.warn(f"Could not retrieve chain from sampler: {e}", stacklevel=2)
        chain = None

    popt = np.full(ndim, np.nan)
    perr = np.full(ndim, np.nan)
    pcov = np.full((ndim, ndim), np.nan)

    if chain is not None and chain.shape[0] > 0:
        try:
            q = np.nanpercentile(chain, [16, 50, 84], axis=0)
            popt = q[1]
            lower_err = q[1] - q[0]
            upper_err = q[2] - q[1]
            perr = (lower_err + upper_err) / 2.0

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
