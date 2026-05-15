import numpy as np
import sys
import layers as lay
import config as cf
import micro_math as mm

def initialize_parameters(layer_dims):
    """
    layer_dims: list containing [input_features, hidden1, hidden2, ..., output]
    Example: [3072, 512, 256, 10]
    """
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        # He Initialization
        parameters["W" + str(l)] = np.random.randn(layer_dims[l-1],
                                                   layer_dims[l]) * np.sqrt(2.0 / layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((1, layer_dims[l]))

    return parameters

def forward_prop(image_data, parameters):
    cache = {}
    A = lay.i_layer(image_data)
    cache["A0"] = A

    L = len(parameters) // 2

    for l in range(1, L):
        # input layer function
        a_prev = A
        # weights and bias' in the params table - init'd from initialize_parameters func
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]

        A, Z = lay.dense_layer(a_prev, W, b)
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A

    # OUTPUT
    W_last = parameters["W" + str(L)]
    b_last = parameters["b" + str(L)]
    ZL, probs = lay.o_layer(A, W_last, b_last)
    cache[f"Z{L}"] = ZL
    cache[f"A{L}"] = probs

    return probs, cache

def back_prop(Y_true, AL, cache, parameters):
    """
    Y_true:   One-hot labels, shape (batch_size, num_classes)
    AL:       Softmax output predictions, shape (batch_size, num_classes)
    cache:    Dict with A0, Zl, Al for each layer
    parameters: Dict with W1, b1, W2, b2, etc.
    """
    L = len(parameters) // 2
    gradients = {}

    # output layer
    dZL = AL - Y_true
    gradients[f"dW{L}"] = cache[f"A{L-1}"].T @ dZL
    gradients[f"db{L}"] = np.sum(dZL, axis=0, keepdims=True)

    for l in range(L - 1, 0, -1):
        # dZ[l] = (dZ[l+1] @ W[l+1].T) * relu_derivative(Z[l])
        # Propagate error backward through weights, then mask dead ReLUs
        gradients[f"dZ{l}"] = (gradients[f"dZ{l+1}"] @ parameters[f"W{l+1}"].T) * \
                              mm.relu_derivative(cache[f"Z{l}"])

        # dW[l] = A[l-1].T @ dZ[l]
        gradients[f"dW{l}"] = cache[f"A{l-1}"].T @ gradients[f"dZ{l}"]

        # db[l] = sum(dZ[l]) across batch
        gradients[f"db{l}"] = np.sum(gradients[f"dZ{l}"], axis=0, keepdims=True)

    return gradients

def update_parameters(parameters, gradients, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * gradients[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * gradients[f"db{l}"]
    return parameters

# images = np.load(cf.images_processed_path)
# labels = np.load(cf.labels_processed_path)
# oh_labels = mm.one_hot(labels)
# params = initialize_parameters(cf.LAYER_DIMS)
#
# result = forward_prop(images, params)
# print(result[0])

