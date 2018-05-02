"""
 NeuralNetworkClass.py  (authors: Sambit, Rosalin, Teja)
 Note: This code is heavily influenced by https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
 Here we will define a neural Network class for multiclass classification.
 
"""
import numpy as np
import random
from math import exp

class NeuralNetwork:
    #
    # Initialize the network variables.
    # Here "hidden_layer_size" is passed as a tuple.
    # ith element of the tuple gives the value of number of neuron in that layer
    #
    def __init__(self, input_layer_size=None, output_layer_size=None, hidden_layer_size=None):
        self.input_layer_size = input_layer_size  # number of features
        self.output_layer_size = output_layer_size  # number of classes
        self.hidden_layer_size = hidden_layer_size  # number of hidden layers
        self.network = self.create_network()

    #
    # Train the network using forward propagation and back propagation
    # We have employed SGD method without momentum for optimisation
    #
    def train(self, X_train, y_train, l_rate=0.5, n_epochs=1000):

        for epoch in range(n_epochs):
            for (x, y) in zip(X_train, y_train):
                # Forward-pass training example into network (updates node output)
                self.forward_prop(x)
                # Create target output
                y_target = np.zeros(self.output_layer_size, dtype=np.int)
                y_target[y] = 1
                # Backward-propagate error into network (updates node delta)
                self.backward_prop(y_target)
                # Update network weights (using updated node delta and node output)
                self.adjust_weights(x, l_rate=l_rate)

    #
    # Predict the most probable class
    # Note: we have as many as nodes in output layer as many as unique classes are there
    #
    def predict(self, X):

        y_predict = np.zeros(len(X), dtype=np.int)
        for i, x in enumerate(X):
            output = self.forward_prop(x)  # output by forward propagating
            y_predict[i] = np.argmax(output)  # predict highest probability class by argmax function

        return y_predict


    #
    # Build neural network by creating weights between nodes
    # Note: our network doesn't contain bias term
    #
    def create_network(self):

        # Connect input nodes with outputs nodes using weights
        def _build_layer(input_layer_size, output_layer_size):
            layer = list()
            for idx_out in range(output_layer_size):
                weights = list()
                for idx_in in range(input_layer_size):
                    weights.append(random.random())
                layer.append({"weights": weights,
                              "output": None,
                              "delta": None})
            return layer

        # random initialisation of weights: input layer -> hidden layer(s)  -> output layer
        hidden_layer_size = len(self.hidden_layer_size)
        network = list()
        if hidden_layer_size == 0:
            network.append(_build_layer(self.input_layer_size, self.output_layer_size))
        else:
            network.append(_build_layer(self.input_layer_size, self.hidden_layer_size[0]))
            for i in range(1,hidden_layer_size):
                network.append(_build_layer(self.hidden_layer_size[i-1],
                                            self.hidden_layer_size[i]))
            network.append(_build_layer(self.hidden_layer_size[hidden_layer_size-1],
                                        self.output_layer_size))

        return network

    #
    # Forward-propagate input -> output and save internal node values
    # This updates: node['output']
    #
    def forward_prop(self, x):

        # Weighted sum of inputs with no bias term for forward propagation
        def _weighted_sum(weights, inputs):
            weighted_sum = 0.0
            for i in range(len(weights)):
                weighted_sum += weights[i] * inputs[i]
            return weighted_sum

        # Perform forward-propagation through the network and update node outputs
        input = x
        for layer in self.network:
            output = list()
            for node in layer:
                # Compute weighted_sum and apply transfer to it
                weighted_sum = _weighted_sum(node['weights'], input)
                node['output'] = self.sigmoid_weighted_sum(weighted_sum)
                output.append(node['output'])
            input = output

        return input

    #
    # Backward-pass error into neural network
    # The loss function is assumed to be L2-error.
    # This updates: node['delta']
    #
    def backward_prop(self, target):

        # Perform backward-propagation through network to update node deltas
        n_layers = len(self.network)
        for i in reversed(range(n_layers)):
            layer = self.network[i]

            # Compute errors either:
            # - explicit target output difference on last layer
            # - weights sum of deltas from frontward layers
            errors = list()
            if i == n_layers - 1:
                # Last layer: errors = target output difference
                for j, node in enumerate(layer):
                    error = target[j] - node['output']
                    errors.append(error)
            else:
                # Previous layers: error = weights sum of frontward node deltas
                for j, node in enumerate(layer):
                    error = 0.0
                    for node in self.network[i + 1]:
                        error += node['weights'][j] * node['delta']
                    errors.append(error)

            # Update delta using our errors
            # The weight update will be:
            # dW = learning_rate * errors * transfer' * input
            #    = learning_rate * delta * input
            for j, node in enumerate(layer):
                node['delta'] = errors[j] * self.sigmoid_gradient(node['output'])

    #
    # Update network weights with error
    # This updates: node['weights']
    #
    def adjust_weights(self, x, l_rate=0.3):

        # Update weights forward layer by layer
        for i_layer, layer in enumerate(self.network):

            # Choose previous layer output to update current layer weights
            if i_layer == 0:
                inputs = x
            else:
                inputs = np.zeros(len(self.network[i_layer - 1]))
                for i_node, node in enumerate(self.network[i_layer - 1]):
                    inputs[i_node] = node['output']

            # Update weights using delta rule for single layer neural network
            # The weight update will be:
            # dW = learning_rate * errors * transfer' * input
            #    = learning_rate * delta * input
            for node in layer:
                for j, input in enumerate(inputs):
                    dW = l_rate * node['delta'] * input
                    node['weights'][j] += dW

    # Activation function (sigmoid)
    def sigmoid_weighted_sum(self, x):
        return 1.0/(1.0+exp(-x))

    # Activation function derivative (sigmoid)
    def sigmoid_gradient(self, transfer):
	return transfer*(1.0-transfer)
