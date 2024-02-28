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
    
    def train(self, X_train, y_train, n_epoch):
        assert self.loss is not None, "Please set the loss function!"
        N_data = len(y_train)
        for iep in range(n_epoch):
            logger.info("Running epoch: %4d" % iep)
            total_error = 0
            for idt, X in enumerate(X_train):
                y_true = y_train[idt]
                # forward propagation:
                # input X needs to be 2D array like: [[1., 2., 3.]]
                y_pred = self._module.forward(X.reshape((1, len(X))))
                # loss functions takes array inputs
                y_true = np.array([[y_true]])
                #print("true", y_true)
                #print("pred", y_pred)
                err = self._loss.forward(y_pred, y_true)
                # print("event error", err, "predicted", y_pred, "true", y_true)
                total_error += err[0][0]

                #print("total error", total_error)
                # backward propagation
                d_error = self._loss.backward(y_pred, y_true)
                #print("d_error", d_error)
                self._module.backward(d_error)

            logger.info("Epoch: %d, Loss: %12.3f\n" % (iep, total_error / N_data))
        logger.info("Trained model!")
    
    def test(self, X_test, y_test):
        N_data = len(y_test)
        logger.info("Number of test data: %d" % N_data)
        total_error = 0
        predicted = np.array()
        for idt, X in enumerate(X_test):
            y_true = y_test[idt]
            y_pred = self._module.forward(X)[0]
            predicted.append(y_pred[0])
            y_true = np.array([y_true])
            total_error += self._loss.forward(y_pred, y_true)
        error = total_error / N_data
        logger.info("Error of test data: %12.3f" % error)
        return predicted
