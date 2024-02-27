from nn import Layer, Linear, Relu
from loss import MSELoss
from module import Module
import numpy as np
import logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuralNet:
    def __init__(self):
        self._module = Module()
        self._loss = None
        self._lr = 0.0001
    
    def add_layer(self, layer):
        self._module.add(layer)

    @property
    def module(self):
        self._module
        
    @property
    def loss(self):
        return self._loss
    
    @loss.setter
    def loss(self, loss_function):
        self._loss = loss_function
    
    @property
    def learning_rate(self):
        return self._lr
    
    @learning_rate.setter
    def learning_rate(self, lr):
        self._lr = lr
        
    def train(self, X_train, y_train, n_epoch):
        assert self.loss is not None, "Please set the loss function!"
        N_data = len(y_train)
        for iep in range(n_epoch):
            logger.info("Running epoch: %4d" % iep)
            total_error = 0
            for idt, X in enumerate(X_train):
                y_true = y_train[idt]
                # forward propagation:
                y_pred = self.module(X)
                total_error += self.loss(y_pred, y_true)

                # backward propagation
                d_error = self.loss.backward(y_pred, y_true)
                self.module.backward(d_error, self.learning_rate)
            logger.info("Loss: %12.3f" % total_error / N_data)
        logger.info("Trained model!")
    
    def test(self, X_test, y_test):
        N_data = len(y_test)
        logger.info("Number of test data: %d" % N_data)
        total_error = 0
        predicted = np.array()
        for idt, X in enumerate(X_test):
            y_true = y_test[idt]
            y_pred = self.module(X)
            predicted.append(y_pred)
            total_error += self.loss(y_pred, y_true)
        error = total_error / N_data
        logger.info("Error of test data: %12.3f" % error)
        return predicted
