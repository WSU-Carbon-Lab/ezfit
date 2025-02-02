class ColumnNotFoundError(Exception):
    def __init__(self, column):
        self.column = column
        self.message = f"Column '{column}' not found in DataFrame."
