"""MCMC diagnostics and convergence analysis tools.

This module provides functions for analyzing MCMC chains, checking convergence,
and computing diagnostic statistics like effective sample size and Gelman-Rubin statistic.
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

    Args:
        chain: MCMC chain array of shape (n_samples, n_params) or (n_walkers, n_steps, n_params).
        axis: Axis along which to compute ESS (0 for samples, 1 for walkers if 3D).

    Returns
    -------
        Effective sample size.
    """
    if chain.ndim == 3:
        # Reshape to (n_samples, n_params) by flattening walkers
        chain = chain.reshape(-1, chain.shape[-1])

    if chain.ndim != 2:
        msg = f"Chain must be 2D or 3D, got shape {chain.shape}"
        raise ValueError(msg)

    n_samples, n_params = chain.shape

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
            tau = 1.0 + 2.0 * np.sum(np.abs(autocorr[1:min(n // 2, 100)]))

        # ESS = n / (1 + 2 * tau)
        ess = n / (1.0 + 2.0 * tau)
        ess_values.append(ess)

    return float(np.min(ess_values))


def gelman_rubin(chain: np.ndarray) -> float:
    """Compute Gelman-Rubin R-hat statistic for MCMC convergence.

    R-hat measures convergence by comparing within-chain and between-chain variance.
    Values close to 1.0 indicate good convergence. Values > 1.1 suggest lack of convergence.

    Args:
        chain: MCMC chain array of shape (n_walkers, n_steps, n_params) or
               (n_samples, n_params) if already flattened.

    Returns
    -------
        R-hat statistic (maximum over all parameters).
    """
    if chain.ndim == 2:
        # Assume single chain, can't compute R-hat
        return 1.0

    if chain.ndim != 3:
        msg = f"Chain must be 2D or 3D for R-hat, got shape {chain.shape}"
        raise ValueError(msg)

    n_walkers, n_steps, n_params = chain.shape

    if n_walkers < 2:
        return 1.0  # Need at least 2 chains

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
        B = (n_steps / (n_walkers - 1.0)) * np.sum(
            (chain_means - overall_mean) ** 2
        )

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

    Args:
        chain: MCMC chain array of shape (n_samples, n_params) or
               (n_walkers, n_steps, n_params).
        method: Method to use: "autocorr" (autocorrelation), "rolling" (rolling mean),
                or "integrated" (integrated autocorrelation time).
        threshold: Threshold for convergence (for autocorr method).

    Returns
    -------
        Estimated burn-in period (number of samples to discard).
    """
    if chain.ndim == 3:
        # Use first parameter as proxy, or average over parameters
        chain = chain.reshape(-1, chain.shape[-1])

    if chain.ndim != 2:
        msg = f"Chain must be 2D or 3D, got shape {chain.shape}"
        raise ValueError(msg)

    n_samples, n_params = chain.shape

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
                # Use arviz for better estimate
                tau = az.ess(chain[:, 0])
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

    Args:
        chain: MCMC chain array of shape (n_walkers, n_steps, n_params) or
               (n_samples, n_params).
        rhat_threshold: Maximum R-hat value for convergence (default 1.1).
        ess_min: Minimum effective sample size for convergence.
        burnin: Burn-in period to discard. If None, estimated automatically.

    Returns
    -------
        Tuple of (converged, diagnostics_dict).
        converged is True if all diagnostics indicate convergence.
        diagnostics_dict contains: rhat, ess, burnin, n_effective_samples.
    """
    diagnostics: dict[str, float | int] = {}

    # Estimate burn-in if not provided
    if burnin is None:
        burnin = estimate_burnin(chain)
    diagnostics["burnin"] = burnin

    # Discard burn-in
    if chain.ndim == 3:
        chain_post_burnin = chain[:, burnin:, :]
    else:
        chain_post_burnin = chain[burnin:, :]

    # Compute R-hat (requires multiple chains)
    if chain.ndim == 3:
        rhat = gelman_rubin(chain_post_burnin)
    else:
        rhat = 1.0  # Can't compute R-hat for single chain
    diagnostics["rhat"] = rhat

    # Compute ESS
    ess = compute_ess(chain_post_burnin)
    diagnostics["ess"] = ess

    # Effective number of samples after burn-in
    if chain.ndim == 3:
        n_effective = int(ess * chain.shape[0])  # ESS per walker
    else:
        n_effective = int(ess)
    diagnostics["n_effective_samples"] = n_effective

    # Check convergence
    converged = True
    if chain.ndim == 3 and rhat > rhat_threshold:
        converged = False
    if ess < ess_min:
        converged = False

    diagnostics["converged"] = converged

    return converged, diagnostics
