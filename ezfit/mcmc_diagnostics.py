"""MCMC diagnostics and convergence analysis module for ezfit.

This module provides comprehensive tools for analyzing MCMC chains, checking
convergence, and computing diagnostic statistics. These functions are used
internally by the emcee fitting method to ensure reliable parameter estimation.

Features
--------
- Effective Sample Size (ESS) computation using autocorrelation analysis
- Gelman-Rubin R-hat statistic for convergence assessment
- Automatic burn-in period estimation using multiple methods
- Comprehensive convergence diagnostics combining multiple metrics
- Integration with arviz for advanced statistical analysis

The functions in this module operate on MCMC chains in various formats:
- 2D arrays: (n_samples, n_params) for flattened chains
- 3D arrays: (n_walkers, n_steps, n_params) for multi-walker chains
"""

import numpy as np

try:
    import arviz as az
except ImportError:
    az = None


def compute_ess(chain: np.ndarray, axis: int = 0) -> float:
    """Compute effective sample size (ESS) of MCMC chain.

    The ESS estimates how many independent samples the chain represents.
    Higher ESS indicates better mixing and more reliable estimates.

    Parameters
    ----------
    chain : np.ndarray
        MCMC chain array of shape (n_samples, n_params) or
        (n_walkers, n_steps, n_params).
    axis : int, optional
        Axis along which to compute ESS (currently unused, reserved for future use),
        by default 0.

    Returns
    -------
    float
        Effective sample size (minimum across all parameters).
    """
    if chain is None or chain.size == 0:
        return 0.0

    if chain.ndim == 3:
        # Reshape to (n_samples, n_params) by flattening walkers
        chain = chain.reshape(-1, chain.shape[-1])

    if chain.ndim != 2:
        msg = f"Chain must be 2D or 3D, got shape {chain.shape}"
        raise ValueError(msg)

    n_samples, n_params = chain.shape

    if n_samples == 0:
        return 0.0

    # Compute ESS for each parameter and return minimum (bottleneck)
    ess_values = []
    for i in range(n_params):
        param_chain = chain[:, i]

        # Compute autocorrelation
        # Simple approach: use integrated autocorrelation time
        # More sophisticated: use arviz if available
        if az is not None:
            try:
                ess = az.ess(param_chain)
                ess_values.append(ess)
                continue
            except Exception:
                pass

        # Fallback: compute autocorrelation manually
        # Remove mean
        centered = param_chain - np.mean(param_chain)

        # Compute autocorrelation function
        n = len(centered)
        autocorr = np.correlate(centered, centered, mode="full")[n - 1 :]
        autocorr = autocorr / autocorr[0]  # Normalize

        # Find where autocorrelation drops below threshold
        threshold = 0.05
        tau = 1.0
        for lag in range(1, min(n // 2, 1000)):  # Limit search
            if abs(autocorr[lag]) < threshold:
                tau = lag
                break
        else:
            # If we didn't find a drop, estimate from integrated autocorrelation
            tau = 1.0 + 2.0 * np.sum(np.abs(autocorr[1 : min(n // 2, 100)]))

        # ESS = n / (1 + 2 * tau)
        ess = n / (1.0 + 2.0 * tau)
        ess_values.append(ess)

    return float(np.min(ess_values))


def gelman_rubin(chain: np.ndarray) -> float:
    """Compute Gelman-Rubin R-hat statistic for MCMC convergence.

    R-hat measures convergence by comparing within-chain and between-chain variance.
    Values close to 1.0 indicate good convergence. Values > 1.1 suggest lack of
    convergence.

    Parameters
    ----------
    chain : np.ndarray
        MCMC chain array of shape (n_walkers, n_steps, n_params) or
        (n_samples, n_params) if already flattened. For 2D chains, returns 1.0
        (cannot compute R-hat without multiple chains).

    Returns
    -------
    float
        R-hat statistic (maximum over all parameters). Returns 1.0 for single chains
        or np.nan for invalid chains.
    """
    if chain is None or chain.size == 0:
        return np.nan

    if chain.ndim == 2:
        # Assume single chain, can't compute R-hat
        return 1.0

    if chain.ndim != 3:
        msg = f"Chain must be 2D or 3D for R-hat, got shape {chain.shape}"
        raise ValueError(msg)

    n_walkers, n_steps, n_params = chain.shape

    if n_walkers < 2 or n_steps == 0:
        return 1.0  # Need at least 2 chains and some steps

    # Compute R-hat for each parameter
    rhat_values = []

    for param_idx in range(n_params):
        param_chains = chain[:, :, param_idx]  # (n_walkers, n_steps)

        # Within-chain variance (W)
        chain_means = np.mean(param_chains, axis=1)  # Mean for each walker
        chain_vars = np.var(param_chains, axis=1, ddof=1)  # Variance for each walker
        W = np.mean(chain_vars)  # Average within-chain variance

        # Between-chain variance (B)
        overall_mean = np.mean(param_chains)
        B = (n_steps / (n_walkers - 1.0)) * np.sum((chain_means - overall_mean) ** 2)

        # Pooled variance estimate
        var_hat = ((n_steps - 1) / n_steps) * W + (1 / n_steps) * B

        # R-hat
        if W > 0:
            rhat = np.sqrt(var_hat / W)
        else:
            rhat = np.inf

        rhat_values.append(rhat)

    return float(np.max(rhat_values))


def estimate_burnin(
    chain: np.ndarray, method: str = "autocorr", threshold: float = 0.05
) -> int:
    """Estimate burn-in period for MCMC chain.

    Parameters
    ----------
    chain : np.ndarray
        MCMC chain array of shape (n_samples, n_params) or
        (n_walkers, n_steps, n_params).
    method : str, optional
        Method to use: "autocorr" (autocorrelation), "rolling" (rolling mean),
        or "integrated" (integrated autocorrelation time), by default "autocorr".
    threshold : float, optional
        Threshold for convergence (for autocorr method), by default 0.05.

    Returns
    -------
    int
        Estimated burn-in period (number of samples to discard).

    Raises
    ------
    ValueError
        If method is unknown or chain shape is invalid.
    """
    if chain is None or chain.size == 0:
        return 0

    if chain.ndim == 3:
        # Use first parameter as proxy, or average over parameters
        chain = chain.reshape(-1, chain.shape[-1])

    if chain.ndim != 2:
        msg = f"Chain must be 2D or 3D, got shape {chain.shape}"
        raise ValueError(msg)

    n_samples, n_params = chain.shape

    if n_samples == 0:
        return 0

    if method == "autocorr":
        # Find where autocorrelation drops below threshold
        # Use first parameter as representative
        param_chain = chain[:, 0]

        centered = param_chain - np.mean(param_chain)
        autocorr = np.correlate(centered, centered, mode="full")[n_samples - 1 :]
        autocorr = autocorr / autocorr[0]

        # Find first lag where autocorr < threshold
        for lag in range(1, min(n_samples // 2, 1000)):
            if abs(autocorr[lag]) < threshold:
                return min(lag * 2, n_samples // 2)  # Conservative estimate

        return n_samples // 4  # Default: discard first quarter

    elif method == "rolling":
        # Use rolling mean/variance to detect stabilization
        window = min(100, n_samples // 10)
        if window < 10:
            return 0

        # Compute rolling variance
        rolling_var = []
        for i in range(window, n_samples):
            window_data = chain[i - window : i, 0]
            rolling_var.append(np.var(window_data))

        # Find where variance stabilizes
        rolling_var = np.array(rolling_var)
        var_change = np.abs(np.diff(rolling_var))
        threshold_var = np.std(var_change) * 2

        for i in range(len(var_change)):
            if var_change[i] < threshold_var:
                return i + window

        return n_samples // 4

    elif method == "integrated":
        # Use integrated autocorrelation time
        if az is not None:
            try:
                # Compute ESS and convert to autocorrelation time
                # arviz.ess returns effective sample size
                ess = az.ess(chain[:, 0])  # type: ignore[attr-defined]
                # Convert ESS to autocorrelation time: tau = n / ESS
                if ess > 0:
                    tau = n_samples / ess
                else:
                    tau = n_samples / 4  # Fallback if ESS is invalid
                burnin = int(5 * tau)  # Conservative: 5x autocorrelation time
                return min(burnin, n_samples // 2)
            except Exception:
                pass

        # Fallback: simple estimate
        return n_samples // 4

    else:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)


def check_convergence(
    chain: np.ndarray,
    rhat_threshold: float = 1.1,
    ess_min: float = 100.0,
    burnin: int | None = None,
) -> tuple[bool, dict[str, float | int]]:
    """Check MCMC chain convergence using multiple diagnostics.

    Combines R-hat statistic and Effective Sample Size (ESS) to determine
    if an MCMC chain has converged. Automatically handles burn-in estimation
    and filters invalid samples.

    Parameters
    ----------
    chain : np.ndarray
        MCMC chain array of shape (n_walkers, n_steps, n_params) or
        (n_samples, n_params).
    rhat_threshold : float, optional
        Maximum R-hat value for convergence, by default 1.1.
    ess_min : float, optional
        Minimum effective sample size for convergence, by default 100.0.
    burnin : int | None, optional
        Burn-in period to discard. If None, estimated automatically, by default None.

    Returns
    -------
    tuple[bool, dict[str, float | int]]
        Tuple of (converged, diagnostics_dict).
        converged is True if all diagnostics indicate convergence.
        diagnostics_dict contains: rhat, ess, burnin, n_effective_samples, converged.
    """
    diagnostics: dict[str, float | int] = {}

    # Validate chain
    if chain is None or chain.size == 0:
        diagnostics["rhat"] = np.nan
        diagnostics["ess"] = 0.0
        diagnostics["burnin"] = 0
        diagnostics["n_effective_samples"] = 0
        diagnostics["converged"] = False
        return False, diagnostics

    # Check for NaN or Inf values
    if np.any(~np.isfinite(chain)):
        # Try to work with finite values only
        if chain.ndim == 3:
            finite_mask = np.all(np.isfinite(chain), axis=(0, 2))
            if np.sum(finite_mask) == 0:
                diagnostics["rhat"] = np.nan
                diagnostics["ess"] = 0.0
                diagnostics["burnin"] = 0
                diagnostics["n_effective_samples"] = 0
                diagnostics["converged"] = False
                return False, diagnostics
            chain = chain[:, finite_mask, :]
        else:
            finite_mask = np.all(np.isfinite(chain), axis=1)
            if np.sum(finite_mask) == 0:
                diagnostics["rhat"] = np.nan
                diagnostics["ess"] = 0.0
                diagnostics["burnin"] = 0
                diagnostics["n_effective_samples"] = 0
                diagnostics["converged"] = False
                return False, diagnostics
            chain = chain[finite_mask, :]

    # Estimate burn-in if not provided
    if burnin is None:
        burnin = estimate_burnin(chain)

    # Ensure burn-in doesn't remove all samples
    if chain.ndim == 3:
        max_burnin = chain.shape[1] - 10  # Keep at least 10 samples
    else:
        max_burnin = chain.shape[0] - 10
    burnin = min(burnin, max_burnin)
    burnin = max(0, burnin)  # Ensure non-negative

    diagnostics["burnin"] = burnin

    # Discard burn-in
    if chain.ndim == 3:
        chain_post_burnin = chain[:, burnin:, :]
        if chain_post_burnin.shape[1] == 0:
            diagnostics["rhat"] = np.nan
            diagnostics["ess"] = 0.0
            diagnostics["n_effective_samples"] = 0
            diagnostics["converged"] = False
            return False, diagnostics
    else:
        chain_post_burnin = chain[burnin:, :]
        if chain_post_burnin.shape[0] == 0:
            diagnostics["rhat"] = np.nan
            diagnostics["ess"] = 0.0
            diagnostics["n_effective_samples"] = 0
            diagnostics["converged"] = False
            return False, diagnostics

    # Compute R-hat (requires multiple chains)
    if chain.ndim == 3:
        try:
            rhat = gelman_rubin(chain_post_burnin)
        except Exception:
            rhat = np.nan
    else:
        rhat = 1.0  # Can't compute R-hat for single chain
    diagnostics["rhat"] = rhat

    # Compute ESS
    try:
        ess = compute_ess(chain_post_burnin)
    except Exception:
        ess = 0.0
    diagnostics["ess"] = ess

    # Effective number of samples after burn-in
    if chain.ndim == 3:
        n_effective = int(ess * chain.shape[0])  # ESS per walker
    else:
        n_effective = int(ess)
    diagnostics["n_effective_samples"] = n_effective

    # Check convergence
    converged = True
    if chain.ndim == 3 and (np.isnan(rhat) or rhat > rhat_threshold):
        converged = False
    if np.isnan(ess) or ess < ess_min:
        converged = False

    diagnostics["converged"] = converged

    return converged, diagnostics
