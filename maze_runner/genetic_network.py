import tensorflow as tf
import random
import maze_runner
from PIL import Image
import glob
import operator
import time

#default folder defines
training_mazes_path = 'mazes/train/*.jpg'

#image specific
image_width = 100
image_height = 100
image_start_point = image_width/2, image_height - 1
fitness_boundary = 200

#genetic algorithm defines
population_count = 20
mutation_chance = 0.05
crossover_chance = 0.5


#neural network defines
n_inputs = 5
n_nodes_hl1 = 5
n_nodes_hl2 = 5

n_classes = 3

class GeneticAlgo:
    ''''
    Class for the genetic algorithm.

    Attributes
    ----------
    gui : instance of the gui to show the results of the network.
    population: list of tuples
                each tuple contains a network and the fitness of the network for the current task.
    '''

    def __init__(self):
        #initialize random population with fitness 0
        self.gui = maze_runner.Mazerunner(self.train_networks)
        self.population = [(NeuralNetwork(), 0) for i in range(0,population_count)]

    def train_networks(self):
        '''
        Starts the genetic learning process with the images.
        '''

        training_mazes = self.load_training_mazes()
        for image in training_mazes:
            global_fitness = 0
            n_generation = 0
            #get access to pixel values
            while global_fitness < fitness_boundary:
                for idx, (network, fitness) in enumerate(self.population):
                    pix_map = image.load()
                    #location of the current pixel
                    current_loc = image_start_point
                    #location of the last pixel
                    last_loc = current_loc
                    #value of the last pixel
                    last_val = pix_map[last_loc]
                    #fitness of the network
                    current_fitness = 0
                    current_laps = 0

                    while True:
                        # recolor last visited location
                        pix_map[last_loc] = last_val
                        #save current color
                        last_loc = current_loc
                        #case if we exceed the boundary limit. this means that we finished one lap
                        try:
                            last_val = pix_map[last_loc]
                            if last_val == (0,0,0):
                                break
                            #last_val holds our current value. if its black, we hit terrain, so exit the loop
                        except IndexError:
                            #if we are in the
                            if current_fitness >= fitness_boundary:
                                break
                            current_laps +=1
                            current_loc = image_start_point
                            last_loc = current_loc
                            last_val = pix_map[last_loc]
                            continue
                        #paint current position red
                        pix_map[current_loc] = (255,0,0)

                        #update statistics
                        current_fitness = image_height - current_loc[1] + image_height*current_laps
                        if current_fitness > global_fitness:
                            global_fitness = current_fitness
                        #if we managed to go 2 laps

                        #move the current point
                        x_loc, y_loc = current_loc
                        y_loc -= 1
                        current_loc = x_loc, y_loc
                        #update the gui with all statistics
                        self.gui.frame.update_state(image, current_fitness, current_laps, global_fitness, n_generation )
                        time.sleep(0.05)

    def calculate_distances(self, current_loc, maze):
        '''

        :param current_loc: tuple containing x and y coordinates
        :param maze: the image displaying the maze
        :return: tuple of 5 distances to the terrain in pixel: W, NW, N, NE, E
        '''
        pass

    def load_training_mazes(self):
        '''
        Loads the images used for training the networks.
        :return: List of images.
        '''
        training_mazes = [Image.open(filename) for filename in glob.glob(training_mazes_path)]
        return training_mazes

    def run_gui(self):
        self.gui.run()

class NeuralNetwork:
    '''
    Baseclass for the neural networks wich will be used for the task.
    '''

    def __init__(self, *args, **kwargs):
        '''
        Base constructor.
        Initializes the neural network with its layers, weights and biases randomly.
        Takes either 0 or 3 arguments: hidden_1_layer, hidden_2_layer, output_layer
        '''
        default_hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_inputs, n_nodes_hl1])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

        default_hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

        default_output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                        'biases': tf.Variable(tf.random_normal([n_classes]))}

        #make sure that if arguments are given, all needed arguments have to be given.
        key_hidden_1_layer = 'hidden_1_layer'
        key_hidden_2_layer = 'hidden_2_layer'
        key_output_layer = 'output_layer'
        if len(kwargs.keys()) > 0:
            for key in [key_hidden_1_layer, key_hidden_2_layer, key_output_layer]:
                try:
                    kwargs[key]
                except KeyError:
                    raise KeyError("The arguments of the constructor don't match the needed arguments.")

        self.hidden_1_layer = kwargs.get(key_hidden_1_layer, default_hidden_1_layer)
        self.hidden_2_layer = kwargs.get(key_hidden_2_layer, default_hidden_2_layer)
        self.output_layer = kwargs.get(key_output_layer, default_output_layer)

    def compute(self, data):
        '''
        Runs the neural network with the given input.
        :param data: The input for the neural network.
        :return: The estimated class.
        '''
        with tf.Session() as sess:
            l1 = tf.add(tf.matmul(data, self.hidden_1_layer['weights']) + self.hidden_1_layer['biases'])
            l1 = tf.nn.relu(l1)

            l2 = tf.add(tf.matmul(l1, self.hidden_2_layer['weights']) + self.hidden_2_layer['biases'])
            l2 = tf.nn.relu(l2)

            output = sess.run(tf.add(tf.matmul(l2, self.output_layer['weights']) + self.output_layer['biases']))

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

        child_network = NeuralNetwork(hidden_1_layer=new_layer_1, hidden_2_layer=new_layer_2, output_layer=new_output_layer)
        return child_network


if __name__ == '__main__':
    algo = GeneticAlgo()
    algo.run_gui()