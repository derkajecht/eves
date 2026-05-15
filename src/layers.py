import micro_math as mm
import numpy as np

def i_layer(image_data):
    """Takes the image data as input and reshapes it, ready for sending onto the hidden layers."""
    input_layer = image_data.reshape(image_data.shape[0], -1) # (10000, 3072)
    return input_layer

def dense_layer(a_prev, W, b):
    """
    A_prev: Input data from the previous layer
    W: Weight matrix for this layer
    b: Bias vector for this layer
    """
    Z = np.dot(a_prev, W) + b
    return Z, mm.relu(Z)

def o_layer(a_prev, W, b):
    """Output layer. Just calls the softmax func."""
    Z = np.dot(a_prev, W) + b
    return Z, mm.softmax(Z)
