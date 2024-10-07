import numpy as np

# Initialize parameters
W_a = np.random.randn(hidden_size, input_size)  # Weights for activations
W_y = np.random.randn(output_size, hidden_size)  # Weights for output
b_a = np.zeros((hidden_size, 1))  # Bias for activations
b_y = np.zeros((output_size, 1))  # Bias for output

# Forward pass
def forward_pass(inputs):
    activations = []
    predictions = []
    for x in inputs:
        a = np.tanh(np.dot(W_a, x) + b_a)  # Activation
        y_hat = softmax(np.dot(W_y, a) + b_y)  # Prediction
        activations.append(a)
        predictions.append(y_hat)
    return activations, predictions

# Backward pass
def backward_pass(inputs, targets, activations, predictions):
    dW_a, dW_y = np.zeros_like(W_a), np.zeros_like(W_y)
    db_a, db_y = np.zeros_like(b_a), np.zeros_like(b_y)

    # Compute loss and gradients
    for t in reversed(range(len(inputs))):
        # Calculate gradients for output layer
        loss_gradient = predictions[t] - targets[t]
        dW_y += np.dot(loss_gradient, activations[t].T)
        db_y += loss_gradient

        # Calculate gradients for hidden layer
        hidden_gradient = (1 - activations[t] ** 2) * np.dot(W_y.T, loss_gradient)
        dW_a += np.dot(hidden_gradient, inputs[t].T)
        db_a += hidden_gradient

    return dW_a, dW_y, db_a, db_y

# Update parameters
def update_parameters(dW_a, dW_y, db_a, db_y, learning_rate):
    global W_a, W_y, b_a, b_y
    W_a -= learning_rate * dW_a
    W_y -= learning_rate * dW_y
    b_a -= learning_rate * db_a
    b_y -= learning_rate * db_y

# Example usage
inputs = [...]  # Your input sequences
targets = [...]  # Your target sequences
activations, predictions = forward_pass(inputs)
dW_a, dW_y, db_a, db_y = backward_pass(inputs, targets, activations, predictions)
update_parameters(dW_a, dW_y, db_a, db_y, learning_rate=0.01)