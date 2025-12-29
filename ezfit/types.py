"""Type hints and type definitions module for ezfit.

This module provides all TypeScript-like type definitions for the ezfit package,
including fit method types, keyword argument types, and result structures.
These types ensure type safety throughout the codebase and provide clear
documentation of expected function signatures.

Features
--------
- FitMethod Literal type for all supported optimization methods
- TypedDict classes for type-safe keyword arguments per optimizer
- FitResult TypedDict for structured fit outputs
- Comprehensive type coverage for all public APIs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable

    import emcee  # Add import for Sampler type hint
    import numpy as np
    from scipy.optimize import OptimizeResult

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
    """Keyword arguments for `scipy.optimize.curve_fit`.

    All fields are optional. Parameters p0, sigma, and absolute_sigma are handled
    automatically by the calling function and should not be included here.

    Attributes
    ----------
    jac : Callable[..., Any] | str | None, optional
        Method for computing the Jacobian matrix.
    method : Literal["lm", "trf", "dogbox"], optional
        Algorithm to use for optimization.
    bounds : tuple[list[float], list[float]], optional
        Lower and upper bounds on parameters.
    full_output : Literal[True], optional
        If True, return full output (required by our implementation).
    check_finite : bool, optional
        Whether to check that input arrays contain only finite numbers.
    nan_policy : Literal["raise", "omit"] | None, optional
        How to handle NaN values in input.
    """

    jac: Callable[..., Any] | str | None
    method: Literal["lm", "trf", "dogbox"]
    bounds: tuple[list[float], list[float]]
    full_output: Literal[True]  # Required by our implementation
    check_finite: bool
    nan_policy: Literal["raise", "omit"] | None
    # p0, sigma, absolute_sigma are handled by the calling function


class MinimizeKwargs(TypedDict, total=False):
    """Keyword arguments for `scipy.optimize.minimize`.

    All fields are optional. Parameters fun and x0 are handled automatically
    by the calling function.

    Attributes
    ----------
    method : str | None, optional
        Type of solver (e.g., 'Nelder-Mead', 'BFGS', 'L-BFGS-B', 'SLSQP').
    jac : Callable | str | Literal["2-point", "3-point", "cs"] | bool | None, optional
        Method for computing the gradient vector.
    hess : Callable | str | Literal["2-point", "3-point", "cs"] | Any | None, optional
        Method for computing the Hessian matrix.
    hessp : Callable | None, optional
        Hessian of objective function times an arbitrary vector p.
    bounds : list[tuple[float, float]] | Any, optional
        Bounds on variables.
    constraints : dict | list[dict] | Any, optional
        Constraints definition (LinearConstraint, NonlinearConstraint).
    tol : float | None, optional
        Tolerance for termination.
    callback : Callable[[np.ndarray], Any] | None, optional
        Called after each iteration.
    options : dict[str, Any] | None, optional
        A dictionary of solver options.
    """

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
    """Keyword arguments for `scipy.optimize.differential_evolution`.

    All fields are optional. Parameters func and bounds are handled automatically
    by the calling function.

    Attributes
    ----------
    strategy : Literal[..., optional
        The differential evolution strategy to use.
    maxiter : int, optional
        Maximum number of iterations.
    popsize : int, optional
        Population multiplier.
    tol : float, optional
        Relative tolerance for convergence.
    mutation : float | tuple[float, float], optional
        Mutation constant.
    recombination : float, optional
        Recombination constant.
    seed : int | np.random.Generator | np.random.RandomState | None, optional
        Random number generator seed.
    callback : Callable[[np.ndarray, ...], Any] | None, optional
        Function called after each iteration.
    disp : bool, optional
        Print evaluated func at every iteration.
    polish : bool, optional
        Apply a polisher to the result.
    init : Literal["latinhypercube", "random", "sobol"] | np.ndarray, optional
        Specify how the population initialization is performed.
    atol : float, optional
        Absolute tolerance for convergence.
    updating : Literal["immediate", "deferred"], optional
        If 'immediate', the best solution vector is continuously updated.
    workers : int | Any, optional
        Number of workers.
    constraints : Any, optional
        Constraint handling.
    x0 : np.ndarray | None, optional
        Initial guess.
    integrality : np.ndarray | list[bool] | None, optional
        Indicates which decision variables are constrained to integer values.
    vectorized : bool, optional
        If True, func is sent a vector of solutions.
    """

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
    """Keyword arguments for `scipy.optimize.shgo`.

    All fields are optional. Parameters func and bounds are handled automatically
    by the calling function.

    Attributes
    ----------
    constraints : dict | list[dict] | None, optional
        Constraints definition.
    n : int, optional
        Number of sampling points.
    iters : int, optional
        Number of iterations.
    callback : Callable[[np.ndarray], Any] | None, optional
        Called after each iteration.
    minimizer_kwargs : dict | None, optional
        Extra keyword arguments to be passed to the minimizer.
    options : dict | None, optional
        A dictionary of solver options.
    sampling_method : Literal["simplicial", "sobol", "halton"] | Callable, optional
        Sampling method to use.
    """

    constraints: dict | list[dict] | None
    n: int
    iters: int
    callback: Callable[[np.ndarray], Any] | None
    minimizer_kwargs: dict | None
    options: dict | None
    sampling_method: Literal["simplicial", "sobol", "halton"] | Callable
    # func, bounds are handled by the calling function


class DualAnnealingKwargs(TypedDict, total=False):
    """Keyword arguments for `scipy.optimize.dual_annealing`.

    All fields are optional. Parameters func and bounds are handled automatically
    by the calling function.

    Attributes
    ----------
    maxiter : int, optional
        Maximum number of iterations.
    local_search_options : dict, optional
        Options for local search algorithm.
    initial_temp : float, optional
        Initial temperature.
    restart_temp_ratio : float, optional
        Temperature restart ratio.
    visit : float, optional
        Parameter for visit distribution.
    accept : float, optional
        Parameter for acceptance distribution.
    maxfun : int, optional
        Soft limit for the number of function evaluations.
    seed : int | np.random.Generator | np.random.RandomState | None, optional
        Random number generator seed.
    no_local_search : bool, optional
        If True, disable local search.
    callback : Callable[[np.ndarray, float, int], Any] | None, optional
        Called after each iteration with (x, f, context).
    x0 : np.ndarray | None, optional
        Initial guess.
    """

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
    """Keyword arguments for `sklearn.linear_model.BayesianRidge`.

    All fields are optional and use sklearn defaults if not provided.

    Attributes
    ----------
    n_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Stop the algorithm if w has converged.
    alpha_1 : float, optional
        Hyper-parameter for the Gamma distribution prior over the alpha parameter.
    alpha_2 : float, optional
        Hyper-parameter for the Gamma distribution prior over the alpha parameter.
    lambda_1 : float, optional
        Hyper-parameter for the Gamma distribution prior over the lambda parameter.
    lambda_2 : float, optional
        Hyper-parameter for the Gamma distribution prior over the lambda parameter.
    alpha_init : float | None, optional
        Initial value for alpha.
    lambda_init : float | None, optional
        Initial value for lambda.
    compute_score : bool, optional
        If True, compute the log marginal likelihood at each iteration.
    fit_intercept : bool, optional
        Whether to calculate the intercept for this model.
    copy_X : bool, optional
        If True, X will be copied; else, it may be overwritten.
    verbose : bool, optional
        Verbose mode when fitting the model.
    """

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
    """Keyword arguments for `emcee.EnsembleSampler` and `run_mcmc`.

    Attributes
    ----------
    nwalkers : int
        Number of walkers (required).
    nsteps : int
        Number of steps to run (required).
    pool : Any | None, optional
        Pool for parallel execution (e.g., multiprocessing.Pool).
    moves : Any | None, optional
        List of (Move, float) tuples or emcee.moves.Move instance.
    backend : Any | None, optional
        Backend for storing chain (emcee.backends.Backend).
    vectorize : bool, optional
        If True, use vectorized likelihood evaluation.
    blobs_dtype : Any | None, optional
        numpy.dtype or list of dtypes for blob data.
    initial_state : Any | None, optional
        Initial state for walkers (State, ndarray).
    tune : bool, optional
        If True, tune the sampler.
    skip_initial_state_check : bool, optional
        If True, skip checking initial state.
    thin_by : int, optional
        Thinning factor.
    thin : int, optional
        Alias for thin_by.
    store : bool, optional
        If True, store chain in backend.
    progress : bool, optional
        If True, show progress bar.
    progress_kwargs : dict | None, optional
        Additional arguments for progress bar.
    """

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
    """Structured dictionary holding the results of a fit.

    All fields are required. This structure is returned by all fitting methods
    and contains the optimized parameters, statistics, and optional diagnostic
    information.

    Attributes
    ----------
    popt : np.ndarray
        Optimized parameter values.
    perr : np.ndarray | None
        Parameter errors (standard deviations). None if not available.
    pcov : np.ndarray | None
        Covariance matrix of parameter estimates. None if not available.
    residuals : np.ndarray | None
        Residuals (ydata - model(xdata, *popt)). None if not computed.
    chi2 : float | None
        Chi-squared statistic. Calculated using sigma=1 when errors are not provided.
        None only if degrees of freedom <= 0.
    rchi2 : float | None
        Reduced chi-squared (chi2 / degrees_of_freedom). None if degrees of freedom <= 0.
    cor : np.ndarray | None
        Correlation matrix of parameters. None if cannot be computed.
    r_squared : float | None
        R² (coefficient of determination). Always calculable when residuals are available.
    pearson_r : float | None
        Pearson correlation coefficient between observed and predicted values. Always calculable.
    rmse : float | None
        Root Mean Square Error. Always calculable.
    rmsd : float | None
        Root Mean Square Deviation (same as RMSE). Always calculable.
    bic : float | None
        Bayesian Information Criterion. Always calculable.
    aic : float | None
        Akaike Information Criterion. Always calculable.
    details : OptimizeResult | emcee.EnsembleSampler | dict[str, Any] | None
        Additional details specific to the fitting method. For MCMC methods,
        this may contain the sampler object or diagnostics dictionary.
    sampler_chain : np.ndarray | None
        MCMC chain array for MCMC methods. Shape (n_walkers, n_steps, n_params)
        or (n_samples, n_params) if flattened. None for non-MCMC methods.
    """

    popt: np.ndarray
    perr: np.ndarray | None
    pcov: np.ndarray | None
    residuals: np.ndarray | None
    chi2: float | None
    rchi2: float | None
    cor: np.ndarray | None
    r_squared: float | None
    pearson_r: float | None
    rmse: float | None
    rmsd: float | None
    bic: float | None
    aic: float | None
    details: (
        OptimizeResult | emcee.EnsembleSampler | dict[str, Any] | None
    )  # More specific type for details
    sampler_chain: np.ndarray | None  # For MCMC methods
