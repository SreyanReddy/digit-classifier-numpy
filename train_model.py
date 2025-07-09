import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

train_data = pd.read_csv(r"D:\Git Repos\digit-classifier-numpy\mnist_csv\mnist_train.csv")
test_data = pd.read_csv(r"D:\Git Repos\digit-classifier-numpy\mnist_csv\mnist_test.csv")

train_data = np.array(train_data)
test_data = np.array(test_data)
m, n = train_data.shape
M, N = test_data.shape
np.random.shuffle(train_data)

# Preparing Data 
test_data = test_data.T
Y_test = test_data[0]
X_test = test_data[1:N] 

data_train = train_data.T
Y_train = data_train[0]
X_train = data_train[1:n]

X_train = X_train / 255
X_test = X_test / 255


def init_params() :
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5 
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)
def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0))  
    return expZ / np.sum(expZ, axis=0)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T
def deriv_ReLU(Z):
    return Z>0

def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * (dZ2.dot(A1.T))
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = (W2.T).dot(dZ2) * deriv_ReLU(Z1)
    dW1 = (1 / m) * (dZ1.dot(X.T))
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)
    
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, epochs):
    W1, b1, W2, b2 = init_params()
    for i in range(epochs):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0 or i == epochs - 1:
            print("Iteration: " , i)
            print("Accuracy ", get_accuracy(get_predictions(A2), Y) * 100, "%")
    return W1, b1, W2, b2

def train_and_save():
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 500)
    np.save("weights/W1.npy", W1)
    np.save("weights/b1.npy", b1)
    np.save("weights/W2.npy", W2)
    np.save("weights/b2.npy", b2)
    print("Model trained and saved.")

if __name__ == "__main__" :
    train_and_save()