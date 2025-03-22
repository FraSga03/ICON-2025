import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def leaky_relu(x, alpha=0.01):
    return np.maximum(x, alpha * x)

def relu_derivative(x):
    return x > 0

def sigmoid_derivative(x):
    return x * (1 - x)

def swish(x):
    return x / (1 + np.exp(-x))

def swish_derivative(x):
    sig = sigmoid(x)
    return sig + x * sig * (1 - sig)

def layer(val, weight, bias, activate_function):
    inter_1 = np.dot(val, weight) + bias
    activate_1 = activate_function(inter_1)

    return inter_1, activate_1

def apply_dropout(layer_output, drop_prob):
    mask = np.random.binomial(1, 1 - drop_prob, size=layer_output.shape)
    return layer_output * mask

np.random.seed(42)

print("Scelta esempi...")
scaler = StandardScaler()
data_train = pd.read_csv("./datasets/doncic_ref_train.csv")
data_test = pd.read_csv("./datasets/doncic_ref_test.csv")

xs = scaler.fit_transform(data_train.drop(columns=["WIN"]))
y = np.array(data_train["WIN"]).reshape(-1, 1)

input_layer_size = xs.shape[1]
hidden_layer_size_1 = 4
hidden_layer_size_2 = 3
output_layer_size = 1

print("Defizione pesi e bias iniziali...")

w1 = np.random.randn(input_layer_size, hidden_layer_size_1)
b1 = np.random.randn(1, hidden_layer_size_1)

w2 = np.random.randn(hidden_layer_size_1, hidden_layer_size_2)
b2 = np.random.randn(1, hidden_layer_size_2)

w3 = np.random.randn(hidden_layer_size_2, output_layer_size)
b3 = np.random.randn(1, output_layer_size)

min_loss = 0.1
prev_loss = 0
learning_rate = 0.002
epochs = 50000
dropout_rate = 0.05

print("Backtracking...")

for epoch in range(epochs):
    inter_1, activate_1 = layer(xs, w1, b1, leaky_relu)

    activate_1 = apply_dropout(activate_1, dropout_rate)
    inter_2, activate_2 = layer(activate_1, w2, b2, swish)

    inter_3, activate_3 = layer(activate_2, w3, b3, sigmoid)

    error = y - activate_3
    loss = np.mean(np.square(error))

    # Backpropagation
    d_inter_3 = error * sigmoid_derivative(activate_3)  # Derivative of the sigmoid
    d_weight_3 = np.dot(activate_2.T, d_inter_3)  # Gradient for W2
    d_bias_3 = np.sum(d_inter_3, axis=0, keepdims=True)  # Gradient for b2

    d_activate_2 = np.dot(d_inter_3, w3.T)  # Propagate error to hidden layer
    d_inter_2 = d_activate_2 * swish_derivative(inter_2)  # Derivative of swish
    d_weight_2 = np.dot(activate_1.T, d_inter_2)  # Gradient for W2
    d_bias_2 = np.sum(d_inter_2, axis=0, keepdims=True)  # Gradient for b2

    d_activate_1 = np.dot(d_inter_2, w2.T)  # Propagate error to hidden layer
    d_inter_1 = d_activate_1 * relu_derivative(inter_1)  # Derivative of ReLU
    d_weight_1 = np.dot(xs.T, d_inter_1)  # Gradient for W1
    d_bias_1 = np.sum(d_inter_1, axis=0, keepdims=True)  # Gradient for b1

    w1 += learning_rate * d_weight_1
    b1 += learning_rate * d_bias_1
    w2 += learning_rate * d_weight_2
    b2 += learning_rate * d_bias_2
    w3 += learning_rate * d_weight_3
    b3 += learning_rate * d_bias_3

    if loss <= min_loss:
        print(f"Break due to goal achieved ")
        break

    if (np.absolute(loss - prev_loss) < 0.000001) and epoch > (epochs/2) :
        print(f"Delta raggiunto")
        break

    prev_loss = loss

    if epoch % 1000 == 0:
        print(f"Epoca {epoch}/{epochs} - Loss: {loss:.5f}")


print("Calcolo su test_set...")

x_test = scaler.fit_transform(data_test.drop(columns=["WIN"]))
y_tests = np.array(data_test["WIN"]).reshape(-1, 1)

inter_1, activate_1 = layer(x_test, w1, b1, leaky_relu)
inter_2, activate_2 = layer(activate_1, w2, b2, swish)
inter_, y_predictions = layer(activate_2, w3, b3, sigmoid)

avg_log_loss = log_loss(y_tests, y_predictions)
print(f"Average Log Loss: {avg_log_loss:.5f}")