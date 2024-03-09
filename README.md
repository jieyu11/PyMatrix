# PyMatrix: Neural Network Implementation

PyMatrix is a neural network implementation leveraging the simplicity of NumPy,
with PyTorch and/or TensorFlow utilized primarily for validation purposes. The
primary objective of this project is not to compete with established packages
but to deepen understanding of neural network mechanics. By meticulously
implementing forward/backward propagations for Linear, Sigmoid, ReLU, and
various loss functions, the inner workings of neural networks are explored. This
endeavor fosters a profound appreciation for those who have developed
comprehensive functionalities in well-known machine learning packages like
PyTorch, TensorFlow, scikit-learn, Keras, XGBoost, and more, often taken for
granted.

## Challenges Encountered during Implementation

### Matrix Operations Accuracy
- Ensure correctness of matrix operations.  - Validate dimensions of inputs,
outputs, weights, etc.  - Utilize 2D matrices for weights consistently, even for
scalar outputs like `[[0.1]]`.

### Implementation of Loss and Activation Functions
- Rigorously implement loss functions and activation functions.  - Address
challenges, such as implementing softmax.

### Accurate Error Calculations
- Validate correct error calculations.  - Ensure accurate computation of
derivatives and correct application of chain rules.

### Multi-class Classification Complexity
- Recognize the nuances of multi-class classification.  - Distinguish the
difference from binary classification, where one value between 0 and 1
represents a binary classification score.  - Understand the use of N values for
N-class classification; while it's theoretically possible to use N-1 values, it
is not a common practice.

### Side Effects
- Gain insights into PyTorch implementation methods.  - Understand the usage of
`loss.backward()` in PyTorch, which calculates the loss and derivatives behind
the scenes.  - Appreciate the advantages of a pre-defined computation graph in
PyTorch.

## Challenges Encountered during Testing

### Weight/Gradient Explosion
- Address challenges related to weight and gradient explosion during testing.

By documenting these challenges, PyMatrix aims to provide a comprehensive
learning experience in neural network implementation while highlighting the
intricacies that go into building and testing such systems.
