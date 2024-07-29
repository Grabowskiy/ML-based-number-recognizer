#Importing the data and numpy library
from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

#Sigmoid functions for the output layer
#TODO Implementing the softmax function and its derivative
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
        #normalising the data
        z = np.exp(z)
        for index, d in enumerate(z):
            d_new = d - np.max(d)
            z[index] = d_new
        #sum = np.sum(np.exp(z - np.max(z)))
        #return np.exp(z - np.max(z)) / sum
        return z / np.sum(z)

    def cross_entropy_loss(self, y_pred, y_true):
        y_pred = self.softmax(y_pred)
        loss = 0
        for i in range(len(y_pred)):
            loss += (-1 * y_true[i] * np.log(y_pred[i]))
        return loss

    def d_cross_entropy_loss(self, y_pred, y_true):
        return y_pred - y_true

    #Forvard propagation function
    def forvard(self, x):
        self.in_h1 = np.dot(x, self.weight_input_h1) + self.bias_hidden1

        self.h1_h2_in = ReLU(self.in_h1)
        self.h1_h2 = np.dot(self.h1_h2_in, self.weight_h1_h2) + self.bias_hidden2

        self.h2_out_in = ReLU(self.h1_h2)
        self.h2_out = np.dot(self.h2_out_in, self.weight_h2_out) + self.bias_out

        self.output = sigmoid(self.h2_out)
        return self.output

    #Backward propagation function
    def backward(self, x, y, learning_rate = 0.01):
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

        #Updating the weights (gradient descent)
        self.weight_h2_out += - learning_rate * weight_hidden2_to_output_gradient
        self.bias_out += - learning_rate * bias_output_gradient
        self.weight_h1_h2 += - learning_rate * weight_hidden1_to_hidden2_gradient
        self.bias_hidden2 += - learning_rate * bias_hidden2_gradient
        self.weight_input_h1 += - learning_rate * weight_input_to_hidden1_gradient
        self.bias_hidden1 += - learning_rate * bias_hidden1_gradient

        return loss

    #For calculating the accuracy of the model, at the end or for epochs
    def accuracy(self, correct, total_samples, epoch = 0, in_training = False):
        accuracy = (correct / total_samples) * 100
        if in_training: print(f"Epoch {epoch + 1}/{epochs} - Acc: {round(accuracy, 2)}%")
        return accuracy

#Plotting the data
def plot(list, list_label, title, xlabel, ylabel, color):
    plt.figure()
    plt.plot(list, label=list_label, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper left', fontsize='medium', frameon=True)
    plt.show()

#Loading the mini batches for learning
def batch_loader(X, y, batch_size):
    n_samples = int(X.shape[0])
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        yield X[begin:end], y[begin: end]

#Object Instantiation
neur = NeuralNetwork(784, 256, 128,10)

#Reading in the data
images, labels = get_mnist()

#Splitting the data into training and testing sets
split_by = 0.8
split_number = int(split_by * images.shape[0])
X_train = images[:split_number]
y_train = labels[:split_number]
X_test = images[split_number:]
y_test = labels[split_number:]

#Defining variables
epochs = 15
batch_size = 16
learning_rate = 0.01
#index = 0
nr_correct = 0
loss_list = []
acc_list = []


#Going through the data epoch times
for epoch in range(epochs):
    print("Starting the learning")
    nr_correct = 0
    total_samples = 0
    #Going through the mini-batches
    for x_batch, y_batch in batch_loader(X_train, y_train, batch_size):
        #Forvard pass
        pred = neur.forvard(x_batch)

        #Backward pass and loss calculation, adding it to list for visualization
        loss = neur.backward(x_batch, y_batch)
        loss_list.append(loss)

        #Adding the correct predictions
        nr_correct += np.sum(np.argmax(pred, axis=1) == np.argmax(y_batch, axis=1))

        #For calculating accuracy
        total_samples += x_batch.shape[0]

        #Printing out the loss after number of iteration for manual evaluation
        #index += 1
        #if index == 1000:
        #    print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss}')
        #    index = 0
    acc_list.append(neur.accuracy(nr_correct, total_samples, epoch, True))

print("Learning finished")

test_correct = 0
len_test = int(X_test.shape[0])
#Evaluating the models performance on the test data
for img, label in zip(X_test, y_test):
    np.reshape(img, (1, 784))
    np.reshape(label, (1, 10))

    pred = neur.forvard(img)
    if np.argmax(pred) == np.argmax(label):
        test_correct += 1
test_acc = neur.accuracy(test_correct, len_test)
print(f'Final accuracy of the model on the testing data (batch size:{batch_size}, learning rate:{learning_rate}) '
      f': {test_acc}')
'''
#Plottint the loss and the accuracy data throughout the learning
plot(loss_list, 'Loss','Loss over Epochs', 'Epoch', 'Loss', 'red')
plot(acc_list, 'Accuracy', 'Accuracy over Epochs','Epoch', 'Accuracy', 'blue' )

#Writing weight and biases to a file
with open("filename", 'w') as file:
    print(f"Az elso layer weightek: {neur.weight_input_h1}\nAz elso layer bias: {neur.bias_hidden1}\n\n"
          f"Az masodik layer weightek: {neur.weight_h1_h2}\nAz masodik layer bias: {neur.bias_hidden2}\n\n"
          f"Az harmadik layer weightek: {neur.weight_h2_out}\nAz harmadik layer bias: {neur.bias_out}\n\n", file=file)
'''