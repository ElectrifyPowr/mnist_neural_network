#!/usr/local/bin/python3
# Copyright 2019-02-24 Powr
#
# All rights reserved
#
# Author: Powr
#
#==================================================================
"""

    3-layer neural network for learning MNIST dataset

"""


import numpy as np
from scipy.special import expit # sigmoid function
from scipy.special import logit # inverse sigmoid function
import matplotlib.pyplot as plt
import sys
import json




# neural network class definition
class neuralNetwork:

    # init neural net
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # set number of nodes in each input, hidden, output layer
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # link weight matrices, wih (weight input->hidden) and who (weight hidden->output)
        # weights inside the arrays are w_i_j, where link is from node i to node j in next layer
        # w11 w21
        # w12 w22 etc
        self.weight_input_hidden = np.random.normal(0.0,
                                                    pow(self.input_nodes, -0.5),
                                                    (self.hidden_nodes, self.input_nodes))
        self.weight_hidden_output = np.random.normal(0.0,
                                                    pow(self.hidden_nodes, -0.5),
                                                    (self.output_nodes, self.hidden_nodes))

        self.learning_rate = learning_rate

        # activation function is the sigmoid function
        self.activation_function = lambda x: expit(x)
        self.inverse_activation_function = lambda x: logit(x)


    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.weight_input_hidden, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.weight_hidden_output, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # error is the (target - actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.weight_hidden_output.T, output_errors)

        # updating weight for link between node j and node k in next layer formula:
        #
        #       Δw_jk = α * E_k * sigmoid(o_k) * (1 - sigmoid(o_k)) · o^{T}_j
        #
        # where o^{T}_j is transposed outputs from previous layer
        #
        # update the weights for the links between the hidden and output layers
        self.weight_hidden_output += self.learning_rate * np.dot((output_errors * final_outputs * (1.0-final_outputs)),
                                                                 np.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.weight_input_hidden += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0-hidden_outputs)),
                                                                 np.transpose(inputs))



    # query the neural network
    def query(self, input_list):
        # convert inputs list to 2d array
        inputs = np.array(input_list, ndmin=2).T

        # calculate signals into hidden layers
        hidden_inputs = np.dot(self.weight_input_hidden, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.weight_hidden_output, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # backquery the neural network
    # we'll use the same termnimology to each item,
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = np.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = np.dot(self.weight_hidden_output.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer
        inputs = np.dot(self.weight_input_hidden.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)



def print_progress(current_index, total_number):
    if current_index % 1000 == 0:
        progress = current_index / total_number
        progress_percentage = round((progress * 100), 3)
        progressBar(progress, 1)
        #print("Progress: {}%".format(progress_percentage))

# prints: Percent: [------------->      ] 69%
def progressBar(value, endvalue, bar_length=20):
        percent = float(value) / endvalue
        arrow = '=' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rProgress: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()



# run the network backwards, given a label, see what image it produces
def run_nn_backwards(neural_net, label, epochs, accuracy):
    # create the output signals for this label
    targets = np.zeros(output_nodes) + 0.01
    # all_values[0] is the target label for this record
    targets[label] = 0.99
    print(targets)

    # get image data
    image_data = neural_net.backquery(targets)

    # plot image data
    fig, axeslist = plt.subplots(ncols=5, nrows=2)

    for i in range(10):
        # create the output signals for this label
        targets = np.zeros(output_nodes) + 0.01
        # all_values[i] is the target label for this record
        targets[i] = 0.99

        # get image data
        image_data = neural_net.backquery(targets)

        axeslist.ravel()[i].imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None')
        axeslist.ravel()[i].set_title(str(i))
        axeslist.ravel()[i].set_axis_off()

    # display plot
    textstr = "Hidden_Nodes: {}, Learning_Rate: {}, Epochs: {}, Accuracy: {}".format(
                neural_net.hidden_nodes, neural_net.learning_rate, epochs, accuracy)
    # x,y -> (0,0) bottom-left, (1,1) top right
    plt.gcf().text(0.02, 0.9, textstr, fontsize=14)
    plt.tight_layout() # optional
    plt.show()




"""
    ------------------------------------------------------------------
    training & testing of NN
    ------------------------------------------------------------------
"""

if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    learning_rate = 0.1

    # epochs is the number of times the training dataset is used for training
    epochs = 1

    print("Neural Network with hidden_nodes: {}, learning_rate: {}, epochs: {}".format(
        hidden_nodes, learning_rate, epochs))


    nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    dir_name = "mnist_dataset/"
    #train_name = "mnist_train.csv"
    #test_name = "mnist_test.csv"
    train_name = "rotated_mnist_train.csv"
    test_name = "rotated_mnist_test.csv"

    # full_path will hold path to either train or test file
    full_path = dir_name + train_name

    # load training data
    training_data_file = open(full_path, 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # train the neural network
    print("Training neural network...")

    for e in range(epochs):
        print("Epoch: {} of {}".format((e+1), epochs))
        # go through all records in the training data set
        for i, record in enumerate(training_data_list):
            # split the record by the ','
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            nn.train(inputs, targets)
            print_progress(i, len(training_data_list))
        print_progress(len(training_data_list), len(training_data_list))
        print()


    # test the neural network
    print("Testing neural network")

    # load test data
    full_path = dir_name + test_name
    test_data_file = open(full_path, 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the records in the test data set
    for i, record in enumerate(test_data_list):
        # split the record by the ','
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        #print("correct label: {}".format(correct_label))
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = nn.query(inputs)
        # index of highest value corresponds to the label
        label = np.argmax(outputs)
        #print("Network's answer: {}".format(label))
        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)

        print_progress(i, len(test_data_list))
    print_progress(len(test_data_list), len(test_data_list))

    # calculate the performance score, the fraction of correct answers
    scorecard_array = np.asarray(scorecard)
    accuracy = (scorecard_array.sum() / scorecard_array.size)
    print()
    print("performance = {}".format(accuracy))

    # reverse-net -> give net label to get image
    label = 0
    run_nn_backwards(nn, label, epochs, accuracy)





    with open("self_made_number1_edit.csv", "r") as f:
        nummer = f.readlines()

    temp = nummer[0]
    all_v = temp.split(',')
    inputs = (np.asfarray(all_v[1:]) / 255.0 * 0.99) + 0.01
    output = nn.query(inputs)
    label = np.argmax(outputs)
    print(label)
