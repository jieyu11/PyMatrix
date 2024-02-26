import numpy as np
class Layer:
    def __init__(self, prev=None, next=None):
        self._prev = prev
        self._next = next
    
    @property
    def prev_layer(self):
        return self._prev
    
    @prev_layer.setter
    def prev_layer(self, layer):
        self._prev = layer
    
    @property
    def next_layer(self):
        return self._next
    
    @next_layer.setter
    def next_layer(self, layer):
        self._next = layer
        
class Linear(Layer):
    def __init__(self, d_input, d_output):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self._weights = 2 * (np.random.rand(d_input, d_output) - 0.5)
        self._bias = 2 * (np.random.rand(1, d_output) - 0.5)
        self.inputs = None
        self.outputs = None

    def forward(self, X):
        self.inputs = X
        self.outputs = X * self._weights + self._bias
        return self.outputs
    
    def backward(self, y_errors, learning_rate):
        """Given errors of the output layer and the learning rate,
        calculate the error of the input layer. Meanwhile, update
        the weights and bias parameters given the gradients.
        Value vectors:
            Y: vector dimension of N_y (1 row of N_y columns)
            B: vector dimension of N_y (1 row of N_y columns)
            X: vector dimension of N_x (1 row of N_x columns)
            W: matrix dimension of N_x * N_y (N_x rows and N_y columns)
            E: loss, which is a scalar.
            dE/dY, dE/dB: vector dimensions same as Y
            dY/dW: vector dimension same as X
            dE/dW: matrix dimension same as W

        Calcuations:
            Y = X * W + B
            dE/dB = dE/dY
            dE/dW = dY/dW * dE/dY, where dY/dW = X
            dE/dX = dE/dY * dY/dX, where dY/dX = W

        Args:
            y_errors (array): errors on the output vector.
            learning_rate (float): learning rate for the parameters

        Returns:
            array: errors on the input vector.
        """
        x_errors = np.dot(y_errors, self.weights.T)
        weight_errors = np.dot(self.input.T, y_errors)

        self.weights -= learning_rate * weight_errors
        self.bias -= learning_rate * y_errors
        return x_errors

    @property
    def weights(self):
        return self._weights
    
    @property
    def bias(self):
        return self._bias

class Relu:
    def __init__(self):
        pass

    def forward(self, X):
        return np.maximum(0., X)
