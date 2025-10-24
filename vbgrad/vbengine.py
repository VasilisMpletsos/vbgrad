class Parameter:
    def __init__(self, value, _children={}, _operation=""):
        self.value = value
        self._children = _children
        self._operation = _operation

    def __repr__(self):
        return f"Parameter = {self.value}"

    def __add__(self, other):
        return Parameter(self.value + other.value, {self, other}, "+")

    def __mul__(self, other):
        return Parameter(self.value * other.value, {self, other}, "*")
