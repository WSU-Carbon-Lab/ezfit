"""Type hints and type definitions for ezfit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

import emcee  # Add import for Sampler type hint
from scipy.optimize import OptimizeResult

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

FitMethod = Literal[
    "curve_fit",
    "minimize",
    "differential_evolution",
    "shgo",
    "dual_annealing",
    "bayesian_ridge",
    "emcee",
    "ridge",
    "lasso",
    "elasticnet",
    "polynomial",
]

type FitKwargs = (
    CurveFitKwargs
    | MinimizeKwargs
    | DifferentialEvolutionKwargs
    | ShgoKwargs
    | DualAnnealingKwargs
    | BayesianRidgeKwargs
    | EmceeKwargs
)


class CurveFitKwargs(TypedDict, total=False):
    """Keyword arguments for `scipy.optimize.curve_fit`."""

    jac: Callable[..., Any] | str | None
    method: Literal["lm", "trf", "dogbox"]
    bounds: tuple[list[float], list[float]]
    full_output: Literal[True]  # Required by our implementation
    check_finite: bool
    nan_policy: Literal["raise", "omit"] | None
    # p0, sigma, absolute_sigma are handled by the calling function


class MinimizeKwargs(TypedDict, total=False):
    """Keyword arguments for `scipy.optimize.minimize`."""

    method: str | None  # e.g., 'Nelder-Mead', 'BFGS', 'L-BFGS-B', 'SLSQP', etc.
    jac: Callable | str | Literal["2-point", "3-point", "cs"] | bool | None
    hess: (
        Callable | str | Literal["2-point", "3-point", "cs"] | Any | None
    )  # OptimizeResult, HessianUpdateStrategy
    hessp: Callable | None
    bounds: list[tuple[float, float]] | Any  # Bounds, Sequence
    constraints: dict | list[dict] | Any  # LinearConstraint, NonlinearConstraint
    tol: float | None
    callback: Callable[[np.ndarray], Any] | None
    options: dict[str, Any] | None
    # fun, x0 are handled by the calling function


class DifferentialEvolutionKwargs(TypedDict, total=False):
    """Keyword arguments for `scipy.optimize.differential_evolution`."""

    strategy: Literal[
        "best1bin",
        "best1exp",
        "rand1exp",
        "randtobest1exp",
        "currenttobest1exp",
        "best2exp",
        "rand2exp",
        "randtobest1bin",
        "currenttobest1bin",
        "best2bin",
        "rand2bin",
        "rand1bin",
    ]
    maxiter: int
    popsize: int
    tol: float
    mutation: float | tuple[float, float]
    recombination: float
    seed: int | np.random.Generator | np.random.RandomState | None
    callback: (
        Callable[[np.ndarray, ...], Any] | None  # type: ignore
    )  # intermediate_result= OptimizeResult
    disp: bool
    polish: bool
    init: Literal["latinhypercube", "random", "sobol"] | np.ndarray
    atol: float
    updating: Literal["immediate", "deferred"]
    workers: int | Any  # map-like callable
    constraints: Any  # NonlinearConstraint, LinearConstraint, Bounds
    x0: np.ndarray | None
    integrality: np.ndarray | list[bool] | None
    vectorized: bool
    # func, bounds are handled by the calling function


class ShgoKwargs(TypedDict, total=False):
    """Keyword arguments for `scipy.optimize.shgo`."""

    constraints: dict | list[dict] | None
    n: int
    iters: int
    callback: Callable[[np.ndarray], Any] | None
    minimizer_kwargs: dict | None
    options: dict | None
    sampling_method: Literal["simplicial", "sobol", "halton"] | Callable
    # func, bounds are handled by the calling function


class DualAnnealingKwargs(TypedDict, total=False):
    """Keyword arguments for `scipy.optimize.dual_annealing`."""

    maxiter: int
    local_search_options: dict
    initial_temp: float
    restart_temp_ratio: float
    visit: float
    accept: float
    maxfun: int
    seed: int | np.random.Generator | np.random.RandomState | None
    no_local_search: bool
    callback: Callable[[np.ndarray, float, int], Any] | None  # x, f, context
    x0: np.ndarray | None
    # func, bounds are handled by the calling function


class BayesianRidgeKwargs(TypedDict, total=False):
    """Keyword arguments for `sklearn.linear_model.BayesianRidge`."""

    n_iter: int
    tol: float
    alpha_1: float
    alpha_2: float
    lambda_1: float
    lambda_2: float
    alpha_init: float | None
    lambda_init: float | None
    compute_score: bool
    fit_intercept: bool
    copy_X: bool
    verbose: bool


class EmceeKwargs(TypedDict, total=False):
    """Keyword arguments for `emcee.EnsembleSampler` and `run_mcmc`."""

    # Sampler args
    nwalkers: int  # Moved from calling function to here
    pool: Any | None  # e.g., multiprocessing.Pool
    moves: Any | None  # list of (Move, float) tuples or emcee.moves.Move instance
    backend: Any | None  # emcee.backends.Backend
    vectorize: bool
    blobs_dtype: Any | None  # numpy.dtype or list of dtypes
    # run_mcmc args
    initial_state: Any | None  # State, ndarray
    nsteps: int  # Required
    tune: bool
    skip_initial_state_check: bool
    thin_by: int
    thin: int  # alias for thin_by
    store: bool
    progress: bool
    progress_kwargs: dict | None
    # ndim, log_prob_fn are handled by the calling function


# --- Result Type Hint ---
class FitResult(TypedDict):
    """Structured dictionary holding the results of a fit."""

    popt: np.ndarray
    perr: np.ndarray | None
    pcov: np.ndarray | None
    residuals: np.ndarray | None
    chi2: float | None
    rchi2: float | None
    cor: np.ndarray | None
    details: (
        OptimizeResult | emcee.EnsembleSampler | dict[str, Any] | None
    )  # More specific type for details
    sampler_chain: np.ndarray | None  # For MCMC methods
