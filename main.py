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


# Returns a softmax of the matrix. A softmax scales all the data in the array so that the data maintains the relative
# distance between each other piece of data in the array (think 2, 4, 8 into 1, 2, 4), but it also conveniently makes
# it so the sum of the data in the array is one! Very useful for probabilities.
def softmax(logits):
    exponentials = np.exp(logits)
    return exponentials / np.sum(exponentials, axis=1).reshape(-1, 1)


# Returns the softmax of the second weights matrix times the sigmoid of the input data times the first
# weights matrix. Predicts outputs based on the inputs and the current weights.
def forward(x_data, w1, w2):
    # Sigmoid the input data times w1. (and add bias in manually)
    h = sigmoid(np.matmul(prepend_bias(x_data), w1))
    # Softmax h * w2 (after adding bias to h manually)
    return softmax(np.matmul(prepend_bias(h), w2)), h


# Classify the data with one of the given labels (0-9)
def classify(x_data, w1, w2):
    y_hat, _ = forward(x_data, w1, w2)
    # Pick the label with the highest probability. np.argmax returns the column number of the highest number,
    # which conveniently also matches the labels for this algorithm.
    labels = np.argmax(y_hat, axis=1)
    # Reshape the labels to be a column instead of a simple array.
    return labels.reshape(-1, 1)


# Returns the cross entropy formula given the ground truth and the experimental predictions
def loss(y_data, y_hat):
    # The negative average of the ground truth matrix multiplied by the log of the predictions
    return -np.sum(y_data * np.log(y_hat)) / y_data.shape[0]


# Returns the gradients of the loss with respect to each of the matrices of weights
def back(x_data, y_data, y_hat, w2, h):
    # Mmmmm... calculus and matrices.
    # The pictures in the book do a great job at explaining all this math. Partial derivatives and the chain rule.
    w2_gradient = np.matmul(prepend_bias(h).T, y_hat - y_data) / x_data.shape[0]
    w1_gradient = np.matmul(prepend_bias(x_data).T, np.matmul(y_hat - y_data, w2[1:].T) * np.multiply(h, 1-h)) / x_data.shape[0]
    return w1_gradient, w2_gradient


# Report how the training loss and accuracy are doing.
def report(iteration, x_data, y_data, x_test, y_test, w1, w2):
    # Make prediction
    y_hat, _ = forward(x_data, w1, w2)
    # Calculate error of said prediction
    training_loss = loss(y_data, y_hat)
    # Make prediction on validation set
    classifications = classify(x_test, w1, w2)
    # Determine accuracy of algorithm on validation set by seeing if the algorithm predicted the correct results.
    accuracy = np.average(classifications == y_test) * 100.0
    print("Iteration: %5d - Loss: %.6f, Accuracy: %.2f%%" % (iteration, training_loss, accuracy))
    return training_loss, accuracy


# Initialize the weights of the network, by picking random small numbers scaled to the number of inputs, hidden nodes,
# and classes.
def init_weights(n_inputs, n_hidden_nodes, n_classes):
    # Number of rows in the first weights matrix is the number of inputs (plus 1 for bias)
    w1_rows = n_inputs + 1
    # Pick a random number between the number of rows of this matrix and the next matrix and multiply it by the
    # square root of the inverse of the number of rows. This will be a good random place to start.
    w1 = np.random.randn(w1_rows, n_hidden_nodes) * np.sqrt(1 / w1_rows)

    # Number of rows in the second weights matrix is the number of hidden nodes (plus 1 for bias)
    w2_rows = n_hidden_nodes + 1
    # Pick a random number between the number of rows of this matrix and the output matrix (10 classifications) and
    # multiply it by the square root of the inverse of the number of rows. This will be a good random place to start.
    w2 = np.random.randn(w2_rows, n_classes) * np.sqrt(1 / w2_rows)
    return w1, w2


# The train function takes the x and y data for both the training and validation sets (which I have plainly named
# data and test), the number of nodes hidden in the middle layer of the network, the number of iterations to run the
# network, and the learning rate for the network (lr).
def train(x_data, y_data, x_test, y_test, n_hidden_nodes, iterations, lr):
    # Determine the number of inputs (which in this case is the number of pixels in the image)
    n_inputs = x_data.shape[1]
    # Determine the number of possible classifications (which is 10, because there are 10 numbers between 0-9,
    # which is what we are classifying)
    n_classes = y_data.shape[1]
    # Initialize the weights for each layer of the network semi-randomly. It needs to be random so that the weights
    # are not uniform, it causes issues when training if the weights are identical.
    w1, w2 = init_weights(n_inputs, n_hidden_nodes, n_classes)
    # Initialize the history dict.
    history = {}
    # Loop for the number of iterations to train the algorithm.
    for i in range(iterations):
        # Make a prediction with the current weights. (Also grab h, which is needed later to determine the gradients)
        y_hat, h = forward(x_data, w1, w2)
        # Backpropogate the data and weights to determine the gradients for each of the weights in both matrices. Mathy
        w1_gradient, w2_gradient = back(x_data, y_data, y_hat, w2, h)
        # Move the weights in the opposite direction of the slope of the loss function multiplied by the learning rate
        w1 -= w1_gradient * lr
        w2 -= w2_gradient * lr
        # Every subdivision of ten of the iterations, print out how the loss and accuracy is doing
        # if i % (iterations / 10) == 0:
        history[i] = report(i, x_data, y_data, x_test, y_test, w1, w2)
    history[iterations] = report(iterations, x_data, y_data, x_test, y_test, w1, w2)
    return w1, w2, history


def main():
    # Train the algorithm in ten thousand iterations with the given learning rate
    w1, w2, history = train(data.X_train, data.Y_train, data.X_test, data.Y_test,
                            n_hidden_nodes=200,
                            iterations=100,
                            lr=.01)
    # The iterations are so low because this math heavy version of the deep neural network is extremely
    # computationally expensive. Training in this way will not be the best way to do this in the future,
    # but it is neat to see how the math actually works! My computer took a couple minutes to finish 100 iterations.

    seaborn.set()  # Activates Seaborn
    # Title our plot
    plot.title("Deep neural network history")
    # Tell Matplotlib we will be creating an array of plots with two rows and one column
    for i in range(2):
        # rows, cols, plot # to edit
        plot.subplot(2, 1, i + 1)

        plot.xlabel("Iterations", fontsize=16)  # Sets the label and font size of the x-axis
        line_color = 'g'  # Sets the color of the line to draw for later

        if i == 0:
            plot.ylabel("Loss", fontsize=16)  # Sets the label and font for the y-axis for the loss graph
        else:
            plot.ylabel("Accuracy", fontsize=16)  # Sets the label and font for the y-axis for the accuracy graph
            line_color = 'b'  # Change color for Accuracy graph

        # Plots either the loss history or accuracy history with respect to the iterations.
        # Plots the iterations as the x values, and either the loss or accuracy data as the y values using list
        # comprehension. This gets the list of values (which is a list of tuples), and then returns the appropriate data
        plot.plot(history.keys(), [h[i] for h in history.values()], line_color)

        # Sets the size of the x-axis
        plot.xlim(0, 100)

        # Set the size of the numbers ticking the axes.
        plot.xticks(fontsize=12)
        plot.yticks(fontsize=12)

    # Transpose and print the weights matrices. Transposition is to help with human-readability.
    print("\nWeight matrix 1: %s\nWeight matrix 2: %s" % (w1.T, w2.T))

    plot.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
