import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load and preprocess the MNIST dataset
def load_data():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"].astype(np.int)
    X = (X / 255.0).astype(np.float32)  # Normalize
    encoder = OneHotEncoder(categories='auto')
    y = encoder.fit_transform(y.reshape(-1, 1)).toarray()
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()

# Define the ReLU activation function and its derivative
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return z > 0

# Define the softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Initialize parameters
def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

# Forward propagation
def forward_propagation(X, parameters):
    caches = []
    A = X.T
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        A = relu(Z)
        caches.append((A_prev, W, b, Z))
        
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    Z = np.dot(W, A) + b
    AL = softmax(Z)
    caches.append((A, W, b, Z))
    
    return AL, caches

# Compute cost
def compute_cost(AL, Y):
    m = Y.shape[0]
    cost = -np.mean(Y * np.log(AL.T + 1e-8))
    return cost

# Backward propagation
def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    
    dZL = AL - Y.T
    current_cache = caches[L-1]
    A_prev, W, b, Z = current_cache
    m = A_prev.shape[1]
    
    grads["dW" + str(L)] = np.dot(dZL, A_prev.T) / m
    grads["db" + str(L)] = np.sum(dZL, axis=1, keepdims=True) / m
    grads["dA" + str(L-1)] = np.dot(W.T, dZL)
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        A_prev, W, b, Z = current_cache
        dZ = np.multiply(grads["dA" + str(l+1)], relu_derivative(Z))
        grads["dW" + str(l+1)] = np.dot(dZ, A_prev.T) / m
        grads["db" + str(l+1)] = np.sum(dZ, axis=1, keepdims=True) / m
        if l > 0:
            grads["dA" + str(l)] = np.dot(W.T, dZ)
            
    return grads

# Update parameters with gradient descent
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    return parameters

# Predict function
def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    predictions = np.argmax(AL, axis=0)
    return predictions

# Model accuracy
def accuracy(predictions, labels):
    return np.mean(predictions == np.argmax(labels, axis=1))

# Compile and train the model
def model(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate=0.0075, num_iterations=2500, print_cost=True):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters(layers_dims)
    
    for i in range(num_iterations):
        AL, caches = forward_propagation(X_train, parameters)
        cost = compute_cost(AL, Y_train)
        grads = backward_propagation(AL, Y_train, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
    
    predictions_train = predict(X_train, parameters)
    predictions_test = predict(X_test, parameters)
    
    print("Train Accuracy:", accuracy(predictions_train, Y_train))
    print("Test Accuracy:", accuracy(predictions_test, Y_test))
    
    return parameters

layers_dims = [784, 128, 64, 10] # 4-layer model
parameters = model(X_train, y_train, X_test, y_test, layers_dims, learning_rate=0.0075, num_iterations=2000, print_cost=True)