import random
from random import randint

from .vbengine import Parameter


class Neuron:
    def __init__(self, input):
        self.w = [
            Parameter(random.uniform(-1, 1), variable_name=f"weight_{i}")
            for i in range(input)
        ]
        self.b = Parameter(random.uniform(-1, 1), variable_name="bias")

    def __call__(self, x):
        if len(x) != len(self.w):
            raise ValueError(f"x must be same length of w = {len(self.w)}")

        summation = sum([self.w[i] * x[i] for i in range(len(x))])
        return summation + self.b

    def __repr__(self):
        return f"Neuron | {len(self.w)} inputs"


class Layer:
    def __init__(self, input, output):
        self.neurons = [Neuron(input) for _ in range(output)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def __repr__(self):
        return f"Layer | {len(self.neurons)} neurons"


class NeuralNetwork:
    def __init__(self, input, layers):
        stacked_outputs = [input] + layers
        self.layers = [
            # The iterations is one less of stacked output so it will not overshoot the end
            Layer(stacked_outputs[i], stacked_outputs[i + 1])
            for i in range(len(layers))
        ]

    def __call__(self, x):
        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"Layer | {len(self.neurons)} neurons"
