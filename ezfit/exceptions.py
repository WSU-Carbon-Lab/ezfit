"""Exceptions for the ezfit package."""


class ColumnNotFoundError(Exception):
    """ColumnNotFoundError is raised when a column is not found in a DataFrame."""

    def __init__(self, column):
        self.column = column
        self.message = f"Column '{column}' not found in DataFrame."
