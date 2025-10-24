class Parameter:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Parameter = {self.value}"

    def __add__(self, other):
        return Parameter(self.value + other.value)

    def __mul__(self, other):
        return Parameter(self.value * other.value)
