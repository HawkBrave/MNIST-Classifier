import numpy as np
import random

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Model:
    def __init__(self, structure):
        self.num_layers = len(structure)
        self.structure = structure
        self.biases = [np.random.randn(y, 1) for y in structure[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(structure[:-1], structure[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print("Epoch {}: {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {}: complete".format(j))

    def update_mini_batch(self, mini_batch, learning_rate):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            gradient_change_b, gradient_change_w = self.backpropagation(x, y)
            gradient_b = [gb + gcb for gb, gcb in zip(gradient_b, gradient_change_b)]
            gradient_w = [gw + gcw for gw, gcw in zip(gradient_w, gradient_change_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * gb for b, gb in zip(self.biases, gradient_b)]
        self.weights = [w - (learning_rate / len(mini_batch)) * gw for w, gw in zip(self.weights, gradient_w)]

    def backpropagation(self, x, y):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            gradient_b[-l] = delta
            gradient_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (gradient_b, gradient_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y
