from .visualization_utils import draw_graph


class Parameter:
    def __init__(self, value, _children={}, _operation="", variable_name=""):
        self.value = value
        self._children = _children
        self._operation = _operation
        self.variable_name = variable_name

    def __repr__(self):
        return f"{self.variable_name} | Parameter = {self.value}"

    def __add__(self, other):
        return Parameter(self.value + other.value, {self, other}, "+")

    def __mul__(self, other):
        return Parameter(self.value * other.value, {self, other}, "*")

    def plot_graph(self):
        return draw_graph(self)
