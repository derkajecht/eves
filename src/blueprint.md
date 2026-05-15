micro_math.py: Contains activation functions
(ReLU, Softmax) and their derivatives, plus the loss
function (MSE or Cross-Entropy).

layers.py: Contains the DenseLayer class. Defines
weight initialization (He/Xavier), the forward
method (computing Z and A), and the backward
method (calculating dW, db, and dx).

model.py: The wrapper class that manages the
sequence of layers. It orchestrates the full forward
pass and the full backward pass through the list of layers.

train.py: The main execution script. Handles
loading .npy files, data preprocessing
(flattening/normalization),
batching, the training loop (epochs), and calling
the update step.

config.py (Optional): Stores hyperparameters like
learning_rate, batch_size, and epoch_count to
keep the code clean.
