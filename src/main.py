#Importing the data and numpy library
from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

# Defining variables
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.01

#Sigmoid functions for the output layer
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def d_sigmoid(z):
    return (sigmoid(z) * (1-sigmoid(z)))

#ReLU and ReLU derivative activation functions
def ReLU(z):
   return np.maximum(0, z)

def d_ReLU(z):
    return np.where(z>0, 1, 0)

#Mean Squared Error for loss calculation
def MSE(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def d_MSE(predictions, targets):
    return 2 * (predictions - targets) / targets.size

#Initialising the neural network
class NeuralNetwork():
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        self.weight_input_h1 = np.random.rand(input_size, hidden1_size) - 0.5
        self.bias_hidden1 = np.random.rand(hidden1_size) - 0.5

        self.weight_h1_h2 = np.random.rand(hidden1_size, hidden2_size) - 0.5
        self.bias_hidden2 = np.random.rand(hidden2_size) - 0.5

        self.weight_h2_out = np.random.rand(hidden2_size, output_size) - 0.5
        self.bias_out = np.random.rand(output_size) - 0.5

    def softmax(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        epsilon = 1e-15 # log(0) elkerülése
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def d_cross_entropy_loss(self, y_pred, y_true):
        return y_pred - y_true

    #Forward propagation function
    def forward(self, x):
        self.in_h1 = np.dot(x, self.weight_input_h1) + self.bias_hidden1

        self.h1_h2_in = ReLU(self.in_h1)
        self.h1_h2 = np.dot(self.h1_h2_in, self.weight_h1_h2) + self.bias_hidden2

        self.h2_out_in = ReLU(self.h1_h2)
        self.h2_out = np.dot(self.h2_out_in, self.weight_h2_out) + self.bias_out

        self.output = self.softmax(self.h2_out)
        return self.output

    #Backward propagation function
    def backward(self, x, y, learning_rate = 0.01):
        #Implemetation for MSE  and sigmoid
        '''
        #Calculating loss and it's derivative
        loss = MSE(self.output, y)
        d_loss = d_MSE(self.output, y)
        #loss = self.cross_entropy_loss(self.output, y)
        #d_loss = self.d_cross_entropy_loss(self.output, y)

        #Calculating the gradients for the layers
        output_gradient = d_loss * d_sigmoid(self.output)
        #output_gradient = self.output - y
        weight_hidden2_to_output_gradient = np.dot(self.h2_out_in.T, output_gradient)
        bias_output_gradient = np.sum(output_gradient)

        hidden2_gradient = np.dot(output_gradient, self.weight_h2_out.T) * d_ReLU(self.h1_h2)
        weight_hidden1_to_hidden2_gradient = np.dot(self.h1_h2_in.T, hidden2_gradient)
        bias_hidden2_gradient = np.sum(hidden2_gradient)

        hidden1_gradient = np.dot(hidden2_gradient, self.weight_h1_h2.T) * d_ReLU(self.in_h1)
        weight_input_to_hidden1_gradient = np.dot(x.T, hidden1_gradient)
        bias_hidden1_gradient = np.sum(hidden1_gradient)
        '''

        # Implemetation for
        batch_size = x.shape[0]

        loss = self.cross_entropy_loss(self.output, y)

        output_gradient = (self.output - y) / batch_size
        weight_hidden2_to_output_gradient = np.dot(self.h2_out_in.T, output_gradient)
        bias_output_gradient = np.sum(output_gradient, axis=0)

        hidden2_gradient = np.dot(output_gradient, self.weight_h2_out.T) * d_ReLU(self.h1_h2)
        weight_hidden1_to_hidden2_gradient = np.dot(self.h1_h2_in.T, hidden2_gradient)
        bias_hidden2_gradient = np.sum(hidden2_gradient, axis=0)

        hidden1_gradient = np.dot(hidden2_gradient, self.weight_h1_h2.T) * d_ReLU(self.in_h1)
        weight_input_to_hidden1_gradient = np.dot(x.T, hidden1_gradient)
        bias_hidden1_gradient = np.sum(hidden1_gradient, axis=0)

        #Updating the weights (gradient descent)
        self.weight_h2_out += - learning_rate * weight_hidden2_to_output_gradient
        self.bias_out += - learning_rate * bias_output_gradient
        self.weight_h1_h2 += - learning_rate * weight_hidden1_to_hidden2_gradient
        self.bias_hidden2 += - learning_rate * bias_hidden2_gradient
        self.weight_input_h1 += - learning_rate * weight_input_to_hidden1_gradient
        self.bias_hidden1 += - learning_rate * bias_hidden1_gradient

        return loss

    #For calculating the accuracy of the model, at the end or for epochs
    def evaluate(self, correct, total_samples, train_loss, batch_number, epoch = 0):
        accuracy = (correct / total_samples) * 100
        loss = train_loss / batch_number
        print(f"Epoch {epoch + 1}/{EPOCHS} - Acc: {round(accuracy, 2)}%, Loss: {round(loss, 5)}")
        return accuracy, loss

#Loading the mini batches for learning
def batch_loader(X, y, batch_size):
    n_samples = int(X.shape[0])
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        yield X[begin:end], y[begin: end]

def plot_performance(train_acc, test_acc, train_loss, test_loss):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 20,  # Base font
        "axes.titlesize": 24,  # Title
        "axes.labelsize": 22,  # Axis labels
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.figsize": (8.3 * 0.8, 11.7 * 0.8 / 1.4),  # A4 width × scaled height
        "figure.dpi": 300
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x_axis = np.arange(len(train_loss))+1
    # Plot Accuracy
    ax1.plot(x_axis, train_acc, linewidth=2, label='Train Accuracy', color='blue')
    ax1.plot(x_axis, test_acc, linewidth=2, label='Test Accuracy', color='orange')
    max_acc_val = max(test_acc)
    max_acc_idx = test_acc.index(max_acc_val)
    ax1.plot(max_acc_idx, max_acc_val, 'r.', markersize=12, label='Max Test Acc')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy [$\%$]')
    ax1.legend()
    ax1.grid(True)

    # Plot Loss
    ax2.plot(x_axis, train_loss, linewidth=2, label='Train Loss', color='blue')
    ax2.plot(x_axis, test_loss, linewidth=2, label='Test Loss', color='orange')
    min_loss_val = min(test_loss)
    min_loss_idx = test_loss.index(min_loss_val)
    ax2.plot(min_loss_idx, min_loss_val, 'r.', markersize=12, label='Min Test Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("figures/100_epochs.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def main() -> NeuralNetwork:
    #Object Instantiation
    neur = NeuralNetwork(784, 128, 64,10)

    #Reading in the data
    images, labels = get_mnist()

    #Splitting the data into training and testing sets
    split_by = 0.8
    split_number = int(split_by * images.shape[0])
    X_train = images[:split_number]
    Y_train = labels[:split_number]
    X_test = images[split_number:]
    Y_test = labels[split_number:]

    #index = 0
    nr_correct = 0
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    print("Starting the learning")

    #Going through the data epoch times
    for epoch in range(EPOCHS):
        total_samples = 0
        train_correct = 0
        train_loss = 0
        batch_number = 0

        # --- TRAIN ---
        print("\nTrain")
        #Going through the mini-batches
        for x_batch, y_batch in batch_loader(X_train, Y_train, BATCH_SIZE):
            #Forward pass
            pred = neur.forward(x_batch)

            #Backward pass and loss calculation, adding it to list for visualization
            loss = neur.backward(x_batch, y_batch)

            #Adding the correct predictions
            train_loss += loss
            train_correct += np.sum(np.argmax(pred, axis=1) == np.argmax(y_batch, axis=1))

            #For calculating accuracy
            total_samples += x_batch.shape[0]
            batch_number += 1

        avg_tran_acc, avg_train_loss = neur.evaluate(train_correct, total_samples, train_loss, batch_number, epoch)
        train_acc_list.append(avg_tran_acc)
        train_loss_list.append(avg_train_loss)

        # --- TEST ---
        print("Test")
        pred = neur.forward(X_test)

        test_loss = neur.cross_entropy_loss(pred, Y_test)
        test_correct = np.sum(np.argmax(pred, axis=1) == np.argmax(Y_test, axis=1))
        test_samples = int(X_test.shape[0])

        avg_test_acc, avg_test_loss = neur.evaluate(test_correct, test_samples, test_loss, 1, epoch)
        test_acc_list.append(avg_test_acc)
        test_loss_list.append(test_loss)

    plot_performance(train_acc_list, test_acc_list, train_loss_list, test_loss_list)

    return neur


if __name__ == "__main__":
    main()