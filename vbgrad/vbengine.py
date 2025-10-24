from math import exp

from .graph_utils import extract_topological_map
from .visualization_utils import draw_graph


class Parameter:
    def __init__(self, value, _children={}, _operation="", variable_name=""):
        self.value = value
        self._children = _children
        self._operation = _operation
        self._backward = lambda: None
        self.grad = 0.0
        self.variable_name = variable_name

    def __repr__(self):
        return f"{self.variable_name} | Parameter = {self.value}"

    def __add__(self, other):
        if not isinstance(other, Parameter):
            other = Parameter(other)

        out = Parameter(self.value + other.value, {self, other}, "+")

        def _backward():
            self.grad = 1 * out.grad
            other.grad = 1 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        if not isinstance(other, Parameter):
            other = Parameter(other)

        out = Parameter(self.value * other.value, {self, other}, "*")

        def _backward():
            self.grad = other.value * out.grad
            other.grad = self.value * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        value = self.value
        tanh = (exp(2 * value) - 1) / (exp(2 * value) + 1)
        out = Parameter(
            tanh,
            {
                self,
            },
            "tanh_activation",
            variable_name="tanh",
        )

        def _backward():
            self.grad = (1 - tanh**2) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        sorted_nodes = extract_topological_map(self)
        self.grad = 1
        for node in sorted_nodes:
            node._backward()

    def plot_graph(self):
        return draw_graph(self)
