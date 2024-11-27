import numpy as np

np.random.seed(42)

class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)


class Linear:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)
        self.dW = None
        self.db = None
        if momentum:
            self.momentum = 0.9
            self.vW = np.zeros_like(self.W)
            self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.input = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad):
        self.input = self.input.reshape(1, -1)
        grad = grad.reshape(1, -1)

        self.dW = np.dot(self.input.T, grad)
        self.db = np.sum(grad, axis=0)
        if momentum:
            self.vW = self.momentum*self.vW - learning_rate * self.dW
            self.vb = self.momentum*self.vb - learning_rate * self.db
            self.W += self.vW
            self.b += self.vb
        else:
            self.W -= learning_rate * self.dW
            self.b -= learning_rate * self.db
        return np.dot(grad, self.W.T)


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad):
        return grad * self.output * (1 - self.output)


class Tanh:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.input = x
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad):
        return grad * (1 - self.output ** 2)


class Relu:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * (self.input > 0)


class MSELoss:
    def __init__(self):
        self.predicted = None
        self.target = None

    def forward(self, predicted, target):
        self.predicted = predicted
        self.target = target
        return np.mean((predicted - target) ** 2)

    def backward(self):
        return 2 * (self.predicted - self.target) / self.predicted.size



input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = np.array([[0], [1], [1], [0]])

loss_fn = MSELoss()
learning_rate = 0.1
epoch_number = 50000
momentum = False

model = Model([Linear(2, 4), Tanh(), Linear(4, 1), Tanh()])

losses = []
for epoch in range(epoch_number):
    epoch_loss = 0
    for i in range(len(input_data)):
        prediction = model.forward(input_data[i])

        loss = loss_fn.forward(prediction, predicted_output[i])
        losses.append(loss)
        grad = loss_fn.backward()
        model.backward(grad)

        epoch_loss += loss

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss {epoch_loss}')

print(f'Epoch {epoch}, Loss {loss}')
for i in range(len(input_data)):
    prediction = model.forward(input_data[i])
    print(input_data[i], prediction)


