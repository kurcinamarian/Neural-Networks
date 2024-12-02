import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self, layers):
        self.layers = layers

    #for layers compute its forward function and use output as input for next layer
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        #output of the model (prediction)
        return x

    #for layers in reversed compute gradiant and use it as input for the next layer
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

#Linear layer
class Linear:
    def __init__(self, input_size, output_size):
        #weights matrix
        self.W = np.random.randn(input_size, output_size)
        #bias matrix
        self.b = np.random.randn(1,output_size)
        #gradient weights matrix
        self.dW = None
        #gradient bias matrix
        self.db = None
        #if momentum add momentum related variables
        if momentum:
            #momentum rate
            self.momentum = momentum_rate
            #weight velocity matrix
            self.vW = np.zeros_like(self.W)
            #bias velocity matrix
            self.vb = np.zeros_like(self.b)

    #matrix multiplication of input and weight matrix and adding biases
    def forward(self, x):
        self.input = x
        return np.dot(x, self.W) + self.b


    def backward(self, grad):
        #compute gradient weights matrix, reshape so diameters align
        self.dW = np.dot(self.input.T.reshape(-1, 1), grad.reshape(1, -1))
        #compute gradient bias matrix
        self.db = grad
        #if momentum update acoring to momentum rate
        if momentum:
            #compute weight velocity matrix
            self.vW = self.momentum*self.vW - learning_rate * self.dW
            #compute bias velocity matrix
            self.vb = self.momentum*self.vb - learning_rate * self.db
            #change weight matrix according to velocity
            self.W += self.vW
            #change bias matrix according to velocity
            self.b += self.vb
        else:
            #change weight matrix according to learning rate
            self.W -= learning_rate * self.dW
            #change bias matrix according to velocity
            self.b -= learning_rate * self.db
        #return gradient for previous layer
        return np.dot(grad, self.W.T)

#Sigmoid activation function
class Sigmoid:
    def __init__(self):
        self.output = None

    #Sigmoid function
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    #Sigmoid backward function
    def backward(self, grad):
        #return gradient for previous layer
        return grad * self.output * (1 - self.output)

#Tanh activation function
class Tanh:
    def __init__(self):
        self.output = None

    #Tanh function
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    #Tanh backward function
    def backward(self, grad):
        #return gradient for previous layer
        return grad * (1 - self.output ** 2)

#RELU activation function (if larger the 0 output is x otherwise 0)
class Relu:
    def __init__(self):
        self.input = None

    #Relu function
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    #Relu backward function
    def backward(self, grad):
        #return gradient for previous layer
        return grad * (self.input > 0)

#MSELoss (Mean Squared Error Loss)
class MSELoss:
    def __init__(self):
        self.predicted = None
        self.target = None

    #return average difference squared for predictions and true values
    def forward(self, predicted, target):
        self.predicted = predicted
        self.target = target
        return np.mean((predicted - target) ** 2)

    #derivated MSELoss function
    def backward(self):
        return 2 * (self.predicted - self.target) / self.predicted.size

########################################################################################################################
#input for testing different parameters
learning_rate = float(input("Learning rate: "))
epoch_number = int(input("Number of epochs: "))
momentum = input("Momentum(y/n): ")
if momentum == "y":
    momentum = True
    momentum_rate = float(input("Momentum: "))
else:
    momentum = False
function = input("Function: ")

#definition of network model and MSELoss
model = Model([
    Linear(2, 4),
    Tanh(),
    Linear(4, 1),
    Tanh(),
])
loss_fn = MSELoss()

#definition of inputs and predicted outputs
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
if function == 'XOR':
    output_data = np.array([[0], [1], [1], [0]])
elif function == 'OR':
    output_data = np.array([[0], [1], [1], [1]])
elif function == 'AND':
    output_data = np.array([[0], [0], [0], [1]])
else:
    print("Invalid function")
    exit()

data = np.hstack((input_data, output_data))

#training process
losses = []
for epoch in range(1,epoch_number+1):
    epoch_loss = 0
    #shuffle data
    np.random.shuffle(data)
    input_data = data[:, :2]
    output_data = data[:, 2:]
    #try all inputs
    for i in range(4):
        #get predicted result by running input forward
        prediction = model.forward(input_data[i])
        #get MSELoss (difference) between prediction and real output for the input
        loss = loss_fn.forward(prediction, output_data[i])
        #get gradient based on the model's prediction
        grad = loss_fn.backward()
        #go backward and change linear layers (weights and bias)
        model.backward(grad)
        epoch_loss += loss
    #get average loss for inputs and add to losses list
    avg_epoch_loss = epoch_loss / 4
    losses.append(avg_epoch_loss)
    if epoch % 10 == 0:
        #average loss for epoch
        print(f'Epoch: {epoch}, Loss: {avg_epoch_loss/4}')

#test of the model
for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    #run model forward and get model's prediction
    prediction = model.forward(i)
    print(f"Input: {i} Predicted output: {prediction}")

#plot the losses over epochs
plt.plot(range(1, epoch_number + 1), losses, label='Training Loss', color='blue', marker='o',markersize=1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(function)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()