import cupy as np
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
    L = len(parameters) // 2
    gradients = {}
    m = Y_true.shape[0]

    # Output layer error
    dZL = AL - Y_true
    gradients[f"dZ{L}"] = dZL

    gradients[f"dW{L}"] = (1/m) * (cache[f"A{L-1}"].T @ dZL)
    gradients[f"db{L}"] = (1/m) * np.sum(dZL, axis=0, keepdims=True)

    for l in range(L - 1, 0, -1):
        # Backprop through ReLU
        gradients[f"dZ{l}"] = (gradients[f"dZ{l+1}"] @ parameters[f"W{l+1}"].T) * mm.relu_derivative(cache[f"Z{l}"])

        gradients[f"dW{l}"] = (1/m) * (cache[f"A{l-1}"].T @ gradients[f"dZ{l}"])
        gradients[f"db{l}"] = (1/m) * np.sum(gradients[f"dZ{l}"], axis=0, keepdims=True)

    return gradients

def update_parameters(parameters, gradients, v, learning_rate, momentum, weight_decay):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        gradients[f"dW{l}"] = np.clip(gradients[f"dW{l}"], -1.0, 0.1)
        gradients[f"db{l}"] = np.clip(gradients[f"db{l}"], -1.0, 0.1)
        for prop in ['W', 'b']:
            key = f"{prop}{l}"
            grad_key = f"d{key}"

            # Weight Decay (L2) only on Weights, not Biases
            reg_grad = np.clip(gradients[grad_key], -1.0, 1.0)
            if prop == 'W':
                reg_grad += weight_decay * parameters[key]

            # Update Velocity: v = (momentum * v) - (lr * grad)
            v[key] = (momentum * v[key]) - (learning_rate * reg_grad)

            # Update Parameters: W = W + v
            parameters[key] += v[key]


    return parameters, v

def initialise_velocity(parameters):
    velocity = {}
    for key, value in parameters.items():
        velocity[key] = np.zeros_like(value)
    return velocity

images = np.load(cf.images_processed_path)
labels_raw = np.load(cf.labels_processed_path)
params = initialize_parameters(cf.LAYER_DIMS)

probs, cache = forward_prop(images, params)
# probs = (probs, probs.shape)
# One hot labels
oh_labels = mm.one_hot(labels_raw)
# oh_labels_ = (oh_labels, oh_labels_shape)

gradients = back_prop(oh_labels, probs, cache, params)
# print(probs)
# print(oh_labels)
# print(gradients)
