import numpy as np
import math

class Model:
    def __init__(self):
        self.linear1 = np.random.rand(2, 4)
        self.cons1 = np.random.rand(1, 4)
        self.linear2 = np.random.rand(4, 1)

    def tanh(self, input):
        return [(math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)) for x in input]

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def relu(self, input):
        return np.maximum(0, input)

    def forward(self, input):
        input = np.array(input).reshape(1, -1)
        input = np.dot(input, self.linear1)
        input += self.cons1
        input = self.tanh(input)
        input = np.dot(input, self.linear2)
        return input


model = Model()

input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]

for i in input_data:
    output = model.forward(i)
    print("Output:", output)
