import tensorflow as tf

#genetic algorithm defines
population_count = 20


#neural network defines
n_nodes_hl1 = 5
n_nodes_hl2 = 5

n_classes = 4

class GeneticAlgo:

    def __init__(self):
        self.population = [NeuralNetwork() for i in range(0,population_count)]


class NeuralNetwork:

    def __init__(self):
        self.hidden_1_layer = {'weights': tf.Variable(tf.random_normal([5, n_nodes_hl1])),
                          'biases': tf.Variable(tf.random_normal(n_nodes_hl1))}

        self.hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                          'biases': tf.Variable(tf.random_normal(n_nodes_hl2))}

        self.output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                        'biases': tf.Variable(tf.random_normal(n_classes))}

    def compute(self, data):
        l1 = tf.add(tf.matmul(data, self.hidden_1_layer['weights']) + self.hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, self.hidden_2_layer['weights']) + self.hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        output = tf.add(tf.matmul(l2, self.output_layer['weights']) + self.output_layer['biases'])

        return output