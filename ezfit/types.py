from typing import Callable, Literal, TypedDict


class FitKwargs(TypedDict):
    check_finite: bool
    method: Literal["lm", "trf", "dogbox"]
    jac: Callable | str | None
    nan_policy: Literal["raise", "omit"] | None
