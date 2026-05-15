import numpy as np

# ReLU, Softmax & loss function (MSE or cross-entropy)
def relu(z):
    """Activation(relu) function.
    Returns 0 for negative values and z for positive values."""
    return np.maximum(0, z)

def relu_derivative(z):
    """Backprop relu function.
    Returns 1 for positive values 0 for negative values."""
    return (z > 0).astype(float)

def softmax(z):
    """Softmax function.
    Returns probabilities that sum to 1 for each row (image)."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot(file, num_classes=10):
    """Loads the labels .npy file and converts labels into a one-hot matrix."""
    file = np.load(file).astype(int).flatten()
    return np.eye(num_classes)[file]

def ce_loss(y_true, y_pred, batch_size):
    epsilon = 1e-15  # prevents log(0) = -inf
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred))
    return loss / batch_size

# if __name__ == "__main__":
#     print("Testing math tools...")
