# This will become the basis of my neural network

import numpy as np
import mnist as data
import matplotlib.pyplot as plot
import seaborn


def prepend_bias(x_data):
    # Insert a column of 1s in the position 0 of X.
    # (axis=1 stands for: insert a column, not a row)
    return np.insert(x_data, 0, 1, axis=1)


# Returns a sigmoid of the data (a value between 0-1)
def sigmoid(z):
    return 1/(1 + np.exp(-z))


# Returns a softmax of the matrix
def softmax(logits):
    exponentials = np.exp(logits)
    return exponentials / np.sum(exponentials, axis=1).reshape(-1, 1)


# Returns the sigmoid of the sum of the input data times the given weights
def forward(x_data, w1, w2):
    h = sigmoid(np.matmul(prepend_bias(x_data), w1))
    return softmax(np.matmul(prepend_bias(h), w2)), h


# Classify the data as true or false (1 or 0)
def classify(x_data, w1, w2):
    y_hat, _ = forward(x_data, w1, w2)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)


# Returns the cross entropy formula given the ground truth and the experimental predictions
def loss(y_data, y_hat):
    # The negative average of the ground truth matrix multiplied by the log of the predictions
    return -np.sum(y_data * np.log(y_hat)) / y_data.shape[0]


# Returns the gradients of the loss with respect to each of the matrices of weights
def back(x_data, y_data, y_hat, w2, h):
    w2_gradient = np.matmul(prepend_bias(h).T, y_hat - y_data) / x_data.shape[0]
    w1_gradient = np.matmul(prepend_bias(x_data).T, np.matmul(y_hat - y_data, w2[1:].T) * np.multiply(h, 1-h)) / x_data.shape[0]
    return w1_gradient, w2_gradient


def report(iteration, x_data, y_data, x_test, y_test, w1, w2):
    y_hat, _ = forward(x_data, w1, w2)
    training_loss = loss(y_data, y_hat)
    classifications = classify(x_test, w1, w2)
    accuracy = np.average(classifications == y_test) * 100.0
    print("Iteration: %5d - Loss: %.6f, Accuracy: %.2f%%" % (iteration, training_loss, accuracy))


def init_weights(n_inputs, n_hidden_nodes, n_classes):
    w1_rows = n_inputs + 1
    w1 = np.random.randn(w1_rows, n_hidden_nodes) * np.sqrt(1 / w1_rows)

    w2_rows = n_hidden_nodes + 1
    w2 = np.random.randn(w2_rows, n_classes) * np.sqrt(1 / w2_rows)
    return w1, w2


def train(x_data, y_data, x_test, y_test, n_hidden_nodes, iterations, lr):
    n_inputs = x_data.shape[1]
    n_classes = y_data.shape[1]
    w1, w2 = init_weights(n_inputs, n_hidden_nodes, n_classes)
    for i in range(iterations):
        y_hat, h = forward(x_data, w1, w2)
        # Move the weights in the opposite direction of the slope of the loss function multiplied by the learning rate
        w1_gradient, w2_gradient = back(x_data, y_data, y_hat, w2, h)
        w1 -= w1_gradient * lr
        w2 -= w2_gradient * lr
        # Every subdivision of ten of the iterations, print out how the loss is doing
        # if i % (iterations / 10) == 0:
        report(i, x_data, y_data, x_test, y_test, w1, w2)
    report(iterations, x_data, y_data, x_test, y_test, w1, w2)
    return w1, w2


def main():
    print("Hello world!")
    # seaborn.set()  # Activates Seaborn
    # plot.axis([75, 105, 0, 40])  # Sets the range of the axes
    # plot.xticks(fontsize=15)  # Sets the font size of the numbers on the axis
    # plot.yticks(fontsize=15)
    # plot.xlabel("Temperature", fontsize=30)  # Sets the label and font size of the x-axis
    # plot.ylabel("Ice Cream Cones Sold", fontsize=30)  # Sets the label and font size of the y-axis
    # x1, x2, x3, y = np.loadtxt("police.txt", skiprows=1, unpack=True)  # Loads data from the file into 2 numpy arrays
    # Stacks a column of ones next to a col of data in a matrix
    # x_data = np.column_stack((np.ones(x1.size), x1, x2, x3))
    # The column of ones represents the bias (or 'b') in the formula

    # Then we reshape the y data into a column that can be modified later
    # y_data = y.reshape(-1, 1)
    # Train the algorithm in ten million iterations with the given learning rate
    w1, w2 = train(data.X_train, data.Y_train, data.X_test, data.Y_test, n_hidden_nodes=200, iterations=10000, lr=.01)
    # Title our plot
    # plot.title("Water Park Ice Cream Sales")
    # # Create 100 points on the x-axis to form a line for our equation we just trained.
    # x_line = np.linspace(75, 105, 100)
    # # Plot the line with the y-data being modified like so: y = w*x + b
    # # The '-' means a solid line and 'b' signifies the color blue
    # plot.plot(x_line, w[1] * x_line + w[0], "-b")
    # plot.plot(x_data, y_data, "bo")  # Plots the data on a graph with pretty circles for each data point

    # print("\nWeights: %s" % w.T)

    # Test algorithm
    # total_examples = data.X_test.shape[0]
    # correct_results = np.sum(classify(data.X_test, w) == data.Y_test)
    # success_percent = correct_results / total_examples * 100
    # print("\nSuccess %d/%d (%.2f%%)" % (correct_results, total_examples, success_percent))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
