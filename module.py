from nn import Layer, Linear, Relu
from loss import MSELoss
import logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Module:
    def __init__(self):
        self.head = Layer()
        self.tail = Layer()
        self.head.next_layer = self.tail
        self.tail.prev_layer = self.head
        self.n_layers = 0
    
    def add(self, layer):
        # add layer before tail
        curr = self.tail.prev_layer
        curr.next_layer = layer
        layer.prev_layer = curr
        layer.next_layer = self.tail
        self.tail.prev_layer = layer
        self.n_layers += 1
    
    def forward(self, inputs):
        if self.n_layers == 0:
            logger.error("No layer is added to the module.")
            return
        curr = self.head.next_layer
        values = inputs
        while curr != self.tail:
            # print("Starting forward with layer, ", curr)
            # print("Layer inputs", values)
            values = curr.forward(values)
            # print("Layer outputs", values)
            curr = curr.next_layer
        return values

    def backward(self, errors):
        if self.n_layers == 0:
            logger.error("No layer is added to the module.")
            return
        curr = self.tail.prev_layer
        values = errors
        while curr != self.head:
            #print("Starting backward with layer, ", curr)
            values = curr.backward(values)
            curr = curr.prev_layer
        return values
        