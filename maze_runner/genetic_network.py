import tensorflow as tf
import random
import maze_runner
from PIL import Image
import glob
import operator
import time
import numpy as np

#default folder defines
training_mazes_path = 'mazes/train/*.png'

#image specific
image_width = 100
image_height = 100
image_start_point = image_width/2, image_height - 1
fitness_boundary = 200
terrain_sight = 5
terrain_color = (0,0,0)
player_color = (255,0,0)

#genetic algorithm defines
population_count = 20
mutation_chance = 0.05
crossover_chance = 0.5


#neural network defines
n_inputs = 5
n_nodes_hl1 = 5
n_nodes_hl2 = 5

n_classes = 3
#maps the activation of each layer to a movement command. 0->move left, 1->move_up, 2->move_right
output_mapping = {0 : (-1, 0),
                  1 : (0, -1),
                  2 : (1, 0)}

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
                            if last_val == terrain_color:
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
                        pix_map[current_loc] = player_color

                        #update statistics
                        current_fitness = image_height - current_loc[1] + image_height*current_laps
                        if current_fitness > global_fitness:
                            global_fitness = current_fitness

                        #move the current point
                        current_loc = self.compute_next_location(current_loc, pix_map, network)
                        #update the gui with all statistics
                        self.gui.frame.update_state(image, current_fitness, current_laps, global_fitness, n_generation )
                        #time.sleep(0.01)

                    #add fitness information for the network
                    self.population[idx] = network, current_fitness

                if global_fitness < fitness_boundary:
                    self.mate_population()
                    n_generation += 1

    def mate_population(self):
        fitness_weights = [fitness for nn, fitness in self.population]
        sum_fitness_weights = sum(fitness_weights)
        cumsum_weights = np.cumsum(np.array(fitness_weights))
        new_population = []
        for i in range(0, population_count):
            rnd_crossover = random.uniform(0,1)
            if rnd_crossover < crossover_chance:
                index_first_parent = np.searchsorted(cumsum_weights, random.randrange(sum_fitness_weights))
                index_second_parent = np.searchsorted(cumsum_weights, random.randrange(sum_fitness_weights))
                #draw two samples according to fitness
                # we dont want to mate a network with itself
                while(index_first_parent == index_second_parent):
                    index_second_parent = np.searchsorted(cumsum_weights, random.randrange(sum_fitness_weights))
                #get the neural networks according to the index
                first_parent = self.population[index_first_parent][0]
                second_parent = self.population[index_second_parent][0]
                child = first_parent.mate(second_parent)

                rnd_mutation = random.uniform(0,1)
                if rnd_mutation < mutation_chance:
                    child.mutate()

                new_population.append((child,0))
            else:
                new_population.append(self.population[i])

        self.population = new_population




    def compute_next_location(self, current_loc, pix_map, network):
        distances = self.calculate_distances(current_loc, pix_map)
        print('Distances: {}'.format(distances))
        result = network.compute(distances)
        # convert nd_array to list and get first entry. Those are our computed values
        result = result.tolist()[0]
        print('Result: {}'.format(result))
        winning_neuron = result.index(max(result))
        print('Winning: {}'.format(winning_neuron))
        movement_commmand = output_mapping[winning_neuron]
        print('Movement_command: {}'.format(movement_commmand))
        next_location = tuple(map(operator.add, current_loc, movement_commmand))
        print('Next location: {}'.format(next_location))
        return next_location


    def calculate_distances(self, current_loc, maze):
        '''
        Calculates the distances into each direction and converts it into a rank 2 tensor.
        :param current_loc: tuple containing x and y coordinates
        :param maze: the image displaying the maze
        :return: tuple of 5 distances to the terrain in pixel: W, NW, N, NE, E
        '''

        #tuples wich will be used to calculate the distances in each direction
        #i.e. one has to go -1 in x direction and 0 in y direction to move to W and check for the nearest black pixel
        #                    W      NW      N      NE     E
        direction_tuples = [(-1,0),(-1,-1),(0,-1),(1,-1),(1,0)]

        distances =[self.calculate_distance_for_one_direciton(current_loc, maze, direction_tuple) for direction_tuple in direction_tuples]
        return distances


    def calculate_distance_for_one_direciton(self, current_loc, maze, direction_tuple):
        '''
        Calculates the distance to the nearst black pixel in the chosen direction.
        :param current_log: current location
        :param direction_tuple: tuple to specify the direction
        :return: Distance to nearest black pixel (terrain) in pixel.
        '''

        #use float hier because the network needs float32
        distance = 0.0
        while distance < terrain_sight:
            #shift the current location in the wanted direction
            current_loc = tuple(map(operator.add, current_loc, direction_tuple))
            #if Terrain
            if maze[current_loc] == terrain_color:
                break
            else:
                distance +=1.0
        return distance


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

        self._init_layer_graph()

        with tf.Session(graph=self.init_graph) as sess:
            sess.run(tf.global_variables_initializer())
            hl1_weights = self.init_graph.get_tensor_by_name('hl1_weights:0')
            hl1_biases = self.init_graph.get_tensor_by_name('hl1_biases:0')
            hl2_weights = self.init_graph.get_tensor_by_name('hl2_weights:0')
            hl2_biases = self.init_graph.get_tensor_by_name('hl2_biases:0')
            out_weights = self.init_graph.get_tensor_by_name('out_weights:0')
            out_biases = self.init_graph.get_tensor_by_name('out_biases:0')
            default_hidden_1_layer = {'weights' :  sess.run(hl1_weights),
                                      'biases' : sess.run(hl1_biases)}
            default_hidden_2_layer = {'weights': sess.run(hl2_weights),
                                      'biases': sess.run(hl2_biases)}
            default_output_layer = {'weights': sess.run(out_weights),
                                      'biases': sess.run(out_biases)}


        self.hidden_1_layer = kwargs.get('hidden_1_layer', default_hidden_1_layer)
        self.hidden_2_layer = kwargs.get('hidden_2_layer', default_hidden_2_layer)
        self.output_layer = kwargs.get('output_layer', default_output_layer)

        self._init_computing_graph()

    def _init_computing_graph(self):
        '''
        Creates the graph for the forward pass of the network.
        :return:
        '''
        self.compute_graph = tf.Graph()
        with self.compute_graph.as_default():
            #placeholder for input
            x = tf.placeholder('float')

            l1 = tf.add(tf.matmul(x, self.hidden_1_layer['weights']), self.hidden_1_layer['weights'])
            l1 = tf.nn.relu(l1)

            l2 = tf.add(tf.matmul(l1, self.hidden_2_layer['weights']), self.hidden_2_layer['biases'])
            l2 = tf.nn.relu(l2)
            output = tf.add(tf.matmul(l2, self.output_layer['weights']), self.output_layer['biases'])




    def _init_layer_graph(self):
        '''
        Initalizes computing graph to create the layer in the neuron.
        :return:
        '''
        self.init_graph = tf.Graph()
        with self.init_graph.as_default():
            hl1_weights = tf.Variable(tf.random_normal([n_inputs, n_nodes_hl1]), name='hl1_weights')
            hl1_biases  = tf.Variable(tf.random_normal([n_nodes_hl1]), name='hl1_biases')

            hl2_weights = tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]), name='hl2_weights')
            hl2_biases  =  tf.Variable(tf.random_normal([n_nodes_hl2]), name='hl2_biases')

            out_weights = tf.Variable(tf.random_normal([n_nodes_hl2, n_classes]), name='out_weights')
            out_biases = tf.Variable(tf.random_normal([n_classes]), name='out_biases')


    def compute(self, data):
        '''
        Runs the neural network with the given input.
        :param data: The input for the neural network.
        :return: The estimated class.
        '''
        #with tf.Session(graph=self.compute_graph) as sess:
        #    sess.run(initialize_variables)
        #    result = sess.run(output, feed_dict={x : data})
        #return result
        pass

    def mutate(self):
        '''
        Mutates the first layer and initializes it random.
        '''
        self.hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_inputs, n_nodes_hl1])),
                          'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

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