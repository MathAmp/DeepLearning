import numpy as np
import random


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, x):
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x) + b)
        return x

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        n = len(training_data)
        t = len(test_data)
        print(n, t)
        if test_data:
            print(f"Before Learning. {self.evaluate(test_data)} / {t}")
        for i in range(epochs):
            # shuffle and split batch
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # gradient descent with one splited mini batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            # result
            if test_data:
                print(f"Epoch {i}: {self.evaluate(test_data)} / {t}")
            else:
                print(f"Epoch {i} complete")

    def update_mini_batch(self, mini_batch, eta):
        # gradient. initializing in zero vector for all biases and weights in ALL layer
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # x, y : data in one mini batch
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_propagation(x, y)      # get one
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - eta * nw / len(mini_batch) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta * nb / len(mini_batch) for b, nb in zip(self.biases, nabla_b)]

    def back_propagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]   # activation in each layer
        z_vectors = list()
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            z_vectors.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(z_vectors[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta.reshape(-1, 1), activations[-2].reshape(1, -1))

        for layer in range(2, self.num_layers):
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sigmoid_prime(z_vectors[-layer])
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta.reshape(-1, 1), activations[-layer - 1].reshape(1, -1))
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum([int(x == y) for x, y in test_results])

    @staticmethod
    def cost_derivative(output_activations, y):
        return output_activations - y

