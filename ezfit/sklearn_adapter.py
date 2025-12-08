"""Adapter layer for scikit-learn integration.

This module provides utilities for converting function-based models to
sklearn-compatible format and detecting linear models.
"""

import inspect
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from sklearn.base import BaseEstimator

try:
    from sklearn.linear_model import (
        ElasticNet,
        Lasso,
        LinearRegression,
        Ridge,
    )
    from sklearn.preprocessing import PolynomialFeatures
except ImportError:
    Ridge = None
    Lasso = None
    ElasticNet = None
    LinearRegression = None
    PolynomialFeatures = None


class SklearnModelWrapper:
    """Wrapper to convert sklearn estimators to ezfit-compatible interface.

    This class adapts sklearn's fit/predict API to work with ezfit's function-based
    model interface.
    """

    def __init__(
        self,
        estimator: "BaseEstimator",
        feature_matrix: np.ndarray,
        param_names: list[str],
    ) -> None:
        """Initialize the wrapper.

        Args:
            estimator: sklearn estimator instance.
            feature_matrix: Design matrix (n_samples, n_features).
            param_names: List of parameter names corresponding to features.
        """
        self.estimator = estimator
        self.feature_matrix = feature_matrix
        self.param_names = param_names

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict using the wrapped estimator.

        Args:
            x: Input data (n_samples,).

        Returns
        -------
            Predicted values.
        """
        # For sklearn, we need to construct the feature matrix for new x values
        # This is model-specific and may need to be overridden
        return self.estimator.predict(self.feature_matrix)  # type: ignore


def is_linear_model(func: "Callable") -> bool:
    """Check if a function represents a linear model.

    A linear model is one where the function is linear in its parameters.
    This is a heuristic check - it's not always possible to determine this
    statically.

    Args:
        func: Function to check.

    Returns
    -------
        True if function appears to be linear in parameters, False otherwise.
    """
    # Get function signature
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    if len(param_names) < 2:
        return False  # Need at least x and one parameter

    # Heuristic: check function source code for nonlinear operations
    try:
        import ast

        source = inspect.getsource(func)
        tree = ast.parse(source)

        # Look for nonlinear operations in the function body
        # This is a simple heuristic - could be improved
        for node in ast.walk(tree):
            if isinstance(node, ast.Pow) and not isinstance(
                node.right, ast.Constant
            ):  # Parameter raised to variable power
                return False
            if isinstance(node, ast.Call):
                # Check for nonlinear functions (exp, log, sin, etc.)
                if isinstance(node.func, ast.Name):
                    nonlinear_funcs = ["exp", "log", "sin", "cos", "tan", "sqrt"]
                    if node.func.id in nonlinear_funcs:
                        # Check if any parameters are used in these calls
                        for arg in ast.walk(node):
                            if isinstance(arg, ast.Name) and arg.id in param_names[1:]:
                                return False
    except Exception:
        # If we can't analyze, assume it might be linear
        pass

    # Default: assume it could be linear (user should know)
    return True


def construct_design_matrix(
    func: "Callable",
    xdata: np.ndarray,
    param_names: list[str],
    p0: list[float],
    eps: float = 1e-6,
) -> np.ndarray:
    """Construct design matrix for linear model using finite differences.

    For a linear model f(x, a, b, c, ...), the design matrix has columns
    corresponding to partial derivatives with respect to each parameter.

    Args:
        func: Model function.
        xdata: Independent variable data.
        param_names: List of parameter names.
        p0: Initial parameter values.
        eps: Step size for finite differences.

    Returns
    -------
        Design matrix of shape (n_samples, n_params).
    """
    n_samples = len(xdata)
    n_params = len(param_names)
    design_matrix = np.zeros((n_samples, n_params))

    # Compute partial derivatives using finite differences
    for i in range(n_params):
        p_perturbed = p0.copy()
        p_perturbed[i] += eps
        y_perturbed = func(xdata, *p_perturbed)
        y_base = func(xdata, *p0)
        design_matrix[:, i] = (y_perturbed - y_base) / eps

    return design_matrix


def convert_to_polynomial_model(
    func: "Callable",
    xdata: np.ndarray,
    degree: int = 2,
) -> tuple[np.ndarray, "PolynomialFeatures | None"]:
    """Convert a function to polynomial features for sklearn.

    This is useful for polynomial regression where we want to fit
    y = a0 + a1*x + a2*x^2 + ...

    Args:
        func: Model function (may be ignored if using polynomial features directly).
        xdata: Independent variable data.
        degree: Polynomial degree.

    Returns
    -------
        Tuple of (feature_matrix, PolynomialFeatures transformer).
    """
    if PolynomialFeatures is None:
        msg = "sklearn.preprocessing.PolynomialFeatures is required."
        raise ImportError(msg)

    # Reshape xdata to 2D if needed
    if xdata.ndim == 1:
        xdata_2d = xdata.reshape(-1, 1)
    else:
        xdata_2d = xdata

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    feature_matrix = poly.fit_transform(xdata_2d)

    return feature_matrix, poly
