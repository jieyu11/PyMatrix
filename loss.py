import numpy as np
class MSELoss:
    def __init__(self):
        pass

    @staticmethod
    def forward(output, target):
        return np.mean(np.power(output - target, 2))
    
    @staticmethod
    def backward(output, target):
        """The derivative of the MSE:
        EY_i = Y_i^pred - Y_i^true
        Loss = 1/n * sum_i(EY_i^2)
        dLoss / dEY_i = 1/n * 2 * EY_i
        """
        diff = output - target
        return 2 * diff / len(output)

class BinaryCrossEntropyLoss:
    MINI = 1.e-12
    def __init__(self):
        pass
    
    def forward(self, output, target):
        """For binary classification, targets are 0's and 1's. For each data point, output
        is a single value between 0 and 1, represented as a 2D matrix, e.g.: [[0.136]]
        The fuction is:
            Loss = - 1./N * sum(y_i^true * log(y_i^pred) + (1- y_i^true)*log(1 - y_i^pred))
        Args:
            output (array): ML model prediction.
            target (array): Truth values.
        """
        N = len(output)
        assert len(target) == N, "target and output vectors must have same dimensions."
        cost = np.dot(np.log(output+self.MINI), target.T) + np.dot(np.log(1.0-output+self.MINI), (1.0-target).T)
        cost *= -1 / len(output)
        return cost
    
    def backward(self, output, target):
        """Backward probagation of cross entropy loss function.
        EY_i = y_i^pred - y_i^true
        Loss = - 1./N * sum(y_i^true * log(y_i^pred) + (1- y_i^true)*log(1 - y_i^pred))
        dLoss / dEY_i = - 1./N * y_i^true / y_i^pred            # if y_i^true = 1
        dLoss / dEY_i = 1./N * (1 - y_i^true) / (1 - y_i^pred)  # if y_i^true = 0
        
        Args:
            output (array): ML model prediction.
            target (array): Truth values.
        """
        N = len(output)
        assert len(target) == N, "target and output vectors must have same dimensions."
        error = target / (output+self.MINI) * (-1./N) + (1-target) / (1-output+self.MINI) * (1./N)
        return np.mean(error)

    def __repr__(self):
        return f"Binary Classification Cross Entropy Loss."


class MultiCrossEntropyLoss:
    MINI = 1.e-12
    def __init__(self):
        pass
    
    def forward(self, output, target):
        """For multi-class classification, targets are 0's and 1's. The output of each data
        point is a vector of N values, given N is the number of classes. For N = 3, it shall
        look like: [[0.35, 0.25, 0.4]]
        The fuction is:
            Loss = - sum_i (y_true_i * log(y_pred_i))
        Args:
            output (array): ML model prediction.
            target (array): Truth values.
        """
        assert target.shape == output.shape, f"target {target.shape} and output {output.shape} not same dimensions."
        # element wise multiplication with: target * log(output)
        cost = np.sum(target * np.log(output+self.MINI))
        cost *= -1
        return cost
    
    def backward(self, output, target):
        """Backward probagation of multi class cross entropy loss function.
        EY_i = y_i^pred - y_i^true
        Loss = - sum(y_i^true * log(y_i^pred))
        dLoss / dEY_i = - y_i^true / y_i^pred
        
        Args:
            output (array): ML model prediction.
            target (array): Truth values.
        """
        assert target.shape == output.shape, f"target {target.shape} and output {output.shape} not same dimensions."
        error = - target / (output+self.MINI)
        return error

    def __repr__(self):
        return f"Multi-class Classification Cross Entropy Loss."