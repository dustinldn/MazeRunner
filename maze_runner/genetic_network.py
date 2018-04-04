import tensorflow as tf
import random

#genetic algorithm defines
population_count = 20
mutation_chance = 0.05
crossover_chance = 0.5


#neural network defines
n_inputs = 5
n_nodes_hl1 = 5
n_nodes_hl2 = 5

n_classes = 4

class GeneticAlgo:

    def __init__(self):
        #initialize random population with fitness 0
        self.population = [(NeuralNetwork(), 0) for i in range(0,population_count)]

    def run(self):
        pass

class NeuralNetwork:
    '''
    Baseclass for the neural networks wich will be used for the task.
    '''

    def __init__(self):
        '''
        Base constructor.
        Initializes the neural network with its layers, weights and biases randomly.
        '''
        self.hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_inputs, n_nodes_hl1])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

        self.hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

        self.output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                        'biases': tf.Variable(tf.random_normal([n_classes]))}

    def __init__(self, hidden_1_layer, hidden_2_layer, output_layer):
        '''
        Custom constructor to assign attributes directly.
        :param hidden_1_layer: Hidden layer 1 with weights and biases
        :param hidden_2_layer: Hidden layer 2 with weights and biases
        :param output_layer: Output layer with weights and biases.
        '''
        self.hidden_1_layer = hidden_1_layer
        self.hidden_2_layer = hidden_2_layer
        self.output_layer = output_layer

    def compute(self, data):
        '''
        Runs the neural network with the given input.
        :param data: The input for the neural network.
        :return: The estimated class.
        '''
        l1 = tf.add(tf.matmul(data, self.hidden_1_layer['weights']) + self.hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, self.hidden_2_layer['weights']) + self.hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        output = tf.add(tf.matmul(l2, self.output_layer['weights']) + self.output_layer['biases'])

        return output

    def mate(self, other_nn):
        '''
        Mates with another neural network to create a child.
        :param other_nn: The other parent neural network.
        :return: Child network consisting of parts of both parent networks.
        '''

        #makes sure that the child is not identical to one of the parents
        random_bool = random.getrandbits(1)
        new_layer_1 = self.hidden_1_layer if random_bool else other_nn.hidden_1_layer
        new_layer_2 = self.hidden_2_layer if not random_bool else other_nn.hidden_2_layer
        new_output_layer = self.output_layer if random.getrandbits(1) else other_nn.output_layer

        child_network = NeuralNetwork(new_layer_1, new_layer_2, new_output_layer)
        return child_network


if __name__ == '__main__':
    algo = GeneticAlgo()