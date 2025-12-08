"""Constraint utilities for parameter relationships and bounds.

This module provides utilities for specifying and handling parameter constraints,
including inequality constraints and parameter relationships.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from scipy.optimize import LinearConstraint, NonlinearConstraint
else:
    try:
        from scipy.optimize import LinearConstraint, NonlinearConstraint
    except ImportError:
        LinearConstraint = None
        NonlinearConstraint = None


class Constraint:
    """Base class for parameter constraints.

    A constraint represents a relationship between parameters that must be
    satisfied during optimization. Constraints can be linear or nonlinear.
    """

    def __init__(
        self,
        constraint_func: Callable[[np.ndarray], float],
        constraint_type: str = "ineq",
        name: str | None = None,
    ) -> None:
        """Initialize a constraint.

        Args:
            constraint_func: Function that takes parameter array and returns
                constraint value. For inequality constraints, should return
                value <= 0 for satisfied constraint.
            constraint_type: Type of constraint: "ineq" (inequality) or "eq" (equality).
            name: Optional name for the constraint (for debugging).
        """
        self.constraint_func = constraint_func
        self.constraint_type = constraint_type
        self.name = name

    def __call__(self, params: np.ndarray) -> float:
        """Evaluate the constraint function."""
        return self.constraint_func(params)

    def to_scipy_constraint(
        self, param_names: list[str], param_indices: dict[str, int]
    ) -> "NonlinearConstraint":
        """Convert to scipy NonlinearConstraint.

        Args:
            param_names: List of parameter names in order.
            param_indices: Mapping from parameter name to index.

        Returns
        -------
            scipy NonlinearConstraint object.
        """
        if NonlinearConstraint is None:
            msg = (
                "scipy.optimize.NonlinearConstraint is required "
                "but scipy is not installed."
            )
            raise ImportError(msg)

        def constraint_func(x: np.ndarray) -> np.ndarray:
            """Convert parameter dict to array."""
            return np.array([self.constraint_func(x)])

        return NonlinearConstraint(
            constraint_func,
            lb=-np.inf if self.constraint_type == "ineq" else 0.0,
            ub=0.0,
        )


def less_than(param1: str, param2: str) -> Callable[[dict[str, float]], bool]:
    """Create a constraint function: param1 < param2.

    Args:
        param1: Name of first parameter.
        param2: Name of second parameter.

    Returns
    -------
        Constraint function that returns True if param1 < param2.
    """
    return lambda p: p[param1] < p[param2]


def greater_than(param1: str, param2: str) -> Callable[[dict[str, float]], bool]:
    """Create a constraint function: param1 > param2.

    Args:
        param1: Name of first parameter.
        param2: Name of second parameter.

    Returns
    -------
        Constraint function that returns True if param1 > param2.
    """
    return lambda p: p[param1] > p[param2]


def sum_less_than(
    params: list[str], value: float
) -> Callable[[dict[str, float]], bool]:
    """Create a constraint function: sum(params) < value.

    Args:
        params: List of parameter names to sum.
        value: Maximum value for the sum.

    Returns
    -------
        Constraint function that returns True if sum < value.
    """
    return lambda p: sum(p[param] for param in params) < value


def sum_greater_than(
    params: list[str], value: float
) -> Callable[[dict[str, float]], bool]:
    """Create a constraint function: sum(params) > value.

    Args:
        params: List of parameter names to sum.
        value: Minimum value for the sum.

    Returns
    -------
        Constraint function that returns True if sum > value.
    """
    return lambda p: sum(p[param] for param in params) > value


def product_equals(
    params: list[str], value: float, tolerance: float = 1e-6
) -> Callable[[dict[str, float]], bool]:
    """Create a constraint function: product(params) == value.

    Args:
        params: List of parameter names to multiply.
        value: Target value for the product.
        tolerance: Tolerance for equality check.

    Returns
    -------
        Constraint function that returns True if product ≈ value.
    """
    return lambda p: abs(np.prod([p[param] for param in params]) - value) < tolerance


def parse_constraint_string(
    constraint_str: str, param_names: list[str]
) -> Callable[[dict[str, float]], bool]:
    """Parse a string constraint into a constraint function.

    Supports simple constraints like:
    - "param1 < param2"
    - "param1 + param2 < 1.0"
    - "param1 * param2 == 2.0"

    Args:
        constraint_str: String representation of constraint.
        param_names: List of valid parameter names.

    Returns
    -------
        Constraint function.

    Raises
    ------
        ValueError: If constraint string cannot be parsed.
    """
    # Simple parser for basic constraints
    # This is a basic implementation - can be extended for more complex cases

    constraint_str = constraint_str.strip()

    # Check for comparison operators
    if " < " in constraint_str:
        parts = constraint_str.split(" < ", 1)
        left = parts[0].strip()
        right = parts[1].strip()

        # Try to parse as number
        try:
            right_val = float(right)
            # Left side is a parameter or expression
            if left in param_names:
                return lambda p: p[left] < right_val
            # Check for sum
            if left.startswith("sum(") and left.endswith(")"):
                params_str = left[4:-1]
                params = [p.strip() for p in params_str.split(",")]
                return sum_less_than(params, right_val)
        except ValueError:
            # Right side is a parameter
            if right in param_names:
                if left in param_names:
                    return less_than(left, right)
                # Check for sum
                if left.startswith("sum(") and left.endswith(")"):
                    params_str = left[4:-1]
                    params = [p.strip() for p in params_str.split(",")]
                    return lambda p: sum(p[param] for param in params) < p[right]

    elif " > " in constraint_str:
        parts = constraint_str.split(" > ", 1)
        left = parts[0].strip()
        right = parts[1].strip()

        try:
            right_val = float(right)
            if left in param_names:
                return lambda p: p[left] > right_val
            if left.startswith("sum(") and left.endswith(")"):
                params_str = left[4:-1]
                params = [p.strip() for p in params_str.split(",")]
                return sum_greater_than(params, right_val)
        except ValueError:
            if right in param_names:
                if left in param_names:
                    return greater_than(left, right)
                if left.startswith("sum(") and left.endswith(")"):
                    params_str = left[4:-1]
                    params = [p.strip() for p in params_str.split(",")]
                    return lambda p: sum(p[param] for param in params) > p[right]

    elif " == " in constraint_str:
        parts = constraint_str.split(" == ", 1)
        if len(parts) != 2:
            msg = f"Could not parse constraint: {constraint_str}"
            raise ValueError(msg)
        left = parts[0].strip()
        right = parts[1].strip()

        try:
            right_val = float(right)
            if left.startswith("product(") and left.endswith(")"):
                params_str = left[8:-1]
                params = [p.strip() for p in params_str.split(",")]
                return product_equals(params, right_val)
        except ValueError:
            pass

    msg = f"Could not parse constraint string: {constraint_str}"
    raise ValueError(msg)


def extract_constraints_from_model(
    model: Any,  # Model type - using Any to avoid circular import
) -> list[Callable[[np.ndarray], float]]:
    """Extract constraint functions from a Model's parameters.

    Args:
        model: Model object with parameters that may have constraints.

    Returns
    -------
        List of constraint functions that take parameter array
        and return constraint value.
    """
    constraints = []

    if model.params is None:
        return constraints

    param_names = list(model.params.keys())

    for param_name, param in model.params.items():
        if param.constraint is not None:
            # Convert dict-based constraint to array-based constraint
            def make_array_constraint(
                name: str, constraint_func: Callable[[dict[str, float]], bool]
            ) -> Callable[[np.ndarray], float]:
                def array_constraint(x: np.ndarray) -> float:
                    param_dict = {param_names[i]: x[i] for i in range(len(param_names))}
                    satisfied = constraint_func(param_dict)
                    # Return negative if satisfied (scipy: <= 0 means satisfied)
                    return -1.0 if satisfied else 1.0

                return array_constraint

            constraints.append(make_array_constraint(param_name, param.constraint))

    return constraints


def validate_constraints(
    model: Any,  # Model type - using Any to avoid circular import
    initial_values: dict[str, float] | None = None,
) -> tuple[bool, str | None]:
    """Validate that constraints are satisfiable with given parameter values.

    Args:
        model: Model with parameters and constraints.
        initial_values: Initial parameter values to test.
            If None, uses parameter values.

    Returns
    -------
        Tuple of (is_valid, error_message).
        is_valid is True if constraints are satisfiable.
    """
    if model.params is None:
        return True, None

    if initial_values is None:
        initial_values = {name: param.value for name, param in model.params.items()}

    # Check each constraint
    for param_name, param in model.params.items():
        if param.constraint is not None:
            try:
                satisfied = param.constraint(initial_values)
                if not satisfied:
                    return (
                        False,
                        f"Constraint on parameter '{param_name}' is not satisfied "
                        f"with initial values: {initial_values}",
                    )
            except KeyError as e:
                return (
                    False,
                    f"Constraint on parameter '{param_name}' "
                    f"references unknown parameter: {e}",
                )
            except Exception as e:
                return (
                    False,
                    f"Error evaluating constraint on parameter '{param_name}': {e}",
                )

    return True, None
