import numpy as np
import matplotlib.pyplot as plt
import train_model as tm

W1 = np.load("weights/W1.npy")
b1 = np.load("weights/b1.npy")
W2 = np.load("weights/W2.npy")
b2 = np.load("weights/b2.npy")

def make_predictions(X, W1, b1, W2, b2):
    Z1, A1, Z2, A2 = tm.forward_prop(W1, b1, W2, b2, X)
    predictions = tm.get_predictions(A2)
    return predictions

# Show Predictions
def show_train_prediction(index, W1, b1, W2, b2):
    current_image = tm.X_train[:, index, None]
    prediction = make_predictions(tm.X_train[:, index, None], W1, b1, W2, b2)
    label = tm.Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation = 'nearest')
    plt.show()

def show_test_prediction(index, W1, b1, W2, b2):
    current_image = tm.X_test[:, index, None]
    prediction = make_predictions(tm.X_test[:, index, None], W1, b1, W2, b2)
    label = tm.Y_test[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_predictions = make_predictions(tm.X_test, W1, b1, W2, b2)
test_accuracy = tm.get_accuracy(test_predictions, tm.Y_test)
print("Test Accuracy:", test_accuracy * 100, "%")
show_test_prediction(200, W1, b1, W2, b2) 