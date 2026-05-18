# import numpy as np
import cupy as np
import model as md
import config as cf
import timeit

def get_batches(X, y, batch_size):
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    for i in range(0, m, batch_size):
        yield X_shuffled[i:i + batch_size], y_shuffled[i:i + batch_size]

def train(X, y, layer_dims, epochs, batch_size, learning_rate, weight_decay):
    # Initialize both params AND velocity
    params = md.initialize_parameters(layer_dims)
    v = md.initialise_velocity(params)

    for i in range(epochs):
        epoch_loss = 0
        correct_predictions = 0

        for X_batch, y_batch in get_batches(X, y, batch_size):
            # 1. Forward
            probs, cache = md.forward_prop(X_batch, params)

            # 2. Track Accuracy
            preds = np.argmax(probs, axis=1)
            targets = np.argmax(y_batch, axis=1)
            correct_predictions += np.sum(preds == targets)

            # 3. Loss
            batch_loss = -np.mean(np.sum(y_batch * np.log(probs + 1e-8), axis=1))
            epoch_loss += batch_loss

            # 4. Backward
            grads = md.back_prop(y_batch, probs, cache, params)

            # 5. Update (Now passing 'v' and receiving the updated 'v' back!)
            params, v = md.update_parameters(params, grads, v, learning_rate, cf.MOMENTUM, weight_decay)

        # Learning Rate Decay Logic
        # if (i + 1) % 100 == 0:
        #     learning_rate = max(learning_rate * cf.LR_DECAY_RATE, cf.MIN_LR)
        #     # Use the local learning_rate variable for the print!
        #     print(f"--- Learning rate decreased to {learning_rate} ---")

        if i % 10 == 0:
            avg_loss = epoch_loss / (X.shape[0] / batch_size)
            accuracy = (correct_predictions / X.shape[0]) * 100
            print(f"Epoch {i} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    return params

batch_size = 2048
trained_params = train(md.images, md.oh_labels, cf.LAYER_DIMS, cf.EPOCHS, batch_size, cf.LEARNING_RATE, cf.WEIGHT_DECAY)
print( trained_params )
