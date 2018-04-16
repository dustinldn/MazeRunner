import tensorflow as tf
import random
import maze_runner
from PIL import Image
import glob
import operator
import numpy as np

#default folder defines
training_mazes_path = 'mazes/train/*.png'
test_mazes_path = 'mazes/test/*.png'
#default_best_network_folder
folder_best_network = 'best_network_attributes/'
#default layer names to save/load
file_hl1_weights = folder_best_network + 'hl1_weights.npy'
file_hl1_biases = folder_best_network + 'hl1_biases.npy'
file_hl2_weights = folder_best_network + 'hl2_weights.npy'
file_hl2_biases = folder_best_network + 'hl2_biases.npy'
file_out_weights = folder_best_network + 'out_weights.npy'
file_out_biases = folder_best_network + 'out_biases.npy'


#image specific
image_width = 100
image_height = 100
image_start_point = image_width/2, image_height - 1
fitness_boundary = 200
terrain_sight = 5
terrain_color = (0,0,0)
player_color = (255,0,0)
show_test_image = True
show_train_image = False

#genetic algorithm defines
population_count = 50
mutation_chance = 0.1
crossover_chance = 0.95
#how many entries in each layer will be mutated
mutate_layer_count = 5


#neural network defines
n_inputs = 5
n_nodes_hl1 = 3
n_nodes_hl2 = 3

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
        self.gui = maze_runner.Mazerunner(self.train_networks, self.test_networks)
        self.population = [(NeuralNetwork(), 0) for i in range(0,population_count)]

    def train_networks(self):
        '''
        Starts the genetic learning process with the images.
        '''
        self.gui.update()
        training_mazes = self.load_mazes(training_mazes_path)
        n_generation = 0
        image_index = 1
        for image in training_mazes:
            global_fitness = 0

            #get access to pixel values
            while global_fitness < fitness_boundary:
                for idx, (network, fitness) in enumerate(self.population):
                    current_fitness = self.run_through_maze(network, image, global_fitness, n_generation, show_train_image)

                    if current_fitness > global_fitness:
                        global_fitness = current_fitness
                    #add fitness information for the network
                    self.population[idx] = network, current_fitness
                    print('Image: {image}\nGeneration: {generation}\nMax_Fitness: {fitness}'.format(image=image_index, generation = n_generation, fitness = global_fitness))
                if global_fitness < fitness_boundary:
                    self.mate_population()
                    n_generation += 1
            image_index += 1
            self.save_best_network()

    def save_best_network(self):
        '''
        Saves the best network according to the fitness to load it later on for testing.
        :return:
        '''
        #get the network with the best fitness
        best_network = max(self.population, key=operator.itemgetter(1))[0]
        print(best_network.hidden_1_layer)
        print(best_network.hidden_2_layer)
        print(best_network.output_layer)
        #create configuration to define in wich file each attribute of the network is stored
        save_configuration = [(best_network.hidden_1_layer['weights'], file_hl1_weights),
                              (best_network.hidden_1_layer['biases'], file_hl1_biases),
                              (best_network.hidden_2_layer['weights'], file_hl2_weights),
                              (best_network.hidden_2_layer['biases'], file_hl2_biases),
                              (best_network.output_layer['weights'], file_out_weights),
                              (best_network.output_layer['biases'], file_out_biases)
                              ]
        for (data, filename) in save_configuration:
            np.save(filename, data)

    def load_network(self):
        '''
        Loads the saved network.
        :return: saved network.
        '''
        hidden_layer_1 = {'weights' : np.load(file_hl1_weights),
                          'biases' : np.load(file_hl1_biases)}
        hidden_layer_2 = {'weights' : np.load(file_hl2_weights),
                          'biases' : np.load(file_hl2_biases)}
        output_layer = {'weights' : np.load(file_out_weights),
                        'biases' : np.load(file_out_biases)}
        print(hidden_layer_1)
        print(hidden_layer_2)
        print(output_layer)

        network = NeuralNetwork(hidden_1_layer=hidden_layer_1, hidden_2_layer=hidden_layer_2, output_layer=output_layer)
        return network


    def test_networks(self):
        '''
        Runs the best network through all test mazes.
        :return:
        '''
        test_mazes = self.load_mazes(test_mazes_path)
        #get the network with the best fitness
        best_network = self.load_network()
        for maze in test_mazes:
            #set current laps = 1 to avoid repeating
            self.run_through_maze(best_network, maze, 0, 0, show_test_image, current_laps=1)


    def run_through_maze(self, network, image, global_fitness, n_generation, show_progress, current_laps=0):
        '''
        Runs the maze with the given neural network.
        :param network: Neural network
        :return: The fitness of the network.
        '''
        pix_map = image.load()
        # location of the current pixel
        current_loc = image_start_point
        # location of the last pixel
        last_loc = current_loc
        # value of the last pixel
        last_val = pix_map[last_loc]
        # fitness of the network
        current_fitness = 0
        # list to store every visited pixel to avoid endless looping between to position
        visited = []
        while True:
            # recolor last visited location
            pix_map[last_loc] = last_val
            # save current color
            last_loc = current_loc
            # case if we exceed the boundary limit. this means that we finished one lap
            try:
                last_val = pix_map[last_loc]
                if last_val == terrain_color:
                    break
                # last_val holds our current value. if its black, we hit terrain, so exit the loop
            except IndexError:
                # if we finished the second lap
                if current_fitness >= fitness_boundary:
                    break
                # clear visited pixel because we start again from the beginning
                visited = []
                current_laps += 1
                current_loc = image_start_point
                last_loc = current_loc
                last_val = pix_map[last_loc]
                continue

            # if we detect a endless loop, quit
            if current_loc in visited:
                break
            else:
                visited.append(current_loc)
            # paint current position red and add it to the visited pixel list
            pix_map[current_loc] = player_color

            # update statistics
            current_fitness = image_height - current_loc[1] + image_height * current_laps
            # move the current point
            current_loc = self.compute_next_location(current_loc, pix_map, network)
            # update the gui with all statistics
            if show_progress:
                self.gui.frame.update_state(image, current_fitness, current_laps, global_fitness, n_generation )
        return current_fitness

    def mate_population(self):
        '''
        Funtion wich chooses neural networks and mates them.
        :return:
        '''
        fitness_weights = [fitness for nn, fitness in self.population]
        sum_fitness_weights = sum(fitness_weights)
        cumsum_weights = np.cumsum(np.array(fitness_weights))
        new_population = []
        for i in range(0, population_count):
            rnd_crossover = random.uniform(0,1)
            if rnd_crossover < crossover_chance:
                index_first_parent = np.searchsorted(cumsum_weights, random.randrange(sum_fitness_weights))
                index_second_parent = random.randrange(population_count)#np.searchsorted(cumsum_weights, random.randrange(sum_fitness_weights))
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
        result = network.compute(distances)
        # convert nd_array to list and get first entry. Those are our computed values
        result = result.tolist()[0]
        winning_neuron = result.index(max(result))
        movement_commmand = output_mapping[winning_neuron]
        next_location = tuple(map(operator.add, current_loc, movement_commmand))
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
            #if we look over the finish, we have distance of 5 to next terrain
            try:
                # if Terrain
                if maze[current_loc] == terrain_color:
                    break
                else:
                    distance +=1.0
            except IndexError:
                distance = 5
        return distance


    def load_mazes(self, path):
        '''
        Loads the images used for training the networks.
        :return: List of images.
        '''
        mazes = [Image.open(filename) for filename in glob.glob(path)]
        return mazes

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
            #input is a list thats why we have to convert it into a matrix

            x = tf.placeholder('float', name='input_data')
            val = tf.convert_to_tensor(x)
            val = tf.reshape(val, [1,5])
            l1 = tf.add(tf.matmul(val, self.hidden_1_layer['weights']), self.hidden_1_layer['weights'])
            l1 = tf.nn.relu(l1)

            l2 = tf.add(tf.matmul(l1, self.hidden_2_layer['weights']), self.hidden_2_layer['biases'])
            l2 = tf.nn.relu(l2)
            output = tf.add(tf.matmul(l2, self.output_layer['weights']), self.output_layer['biases'], name='output_layer')




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
        with tf.Session(graph=self.compute_graph) as sess:
            sess.run(tf.global_variables_initializer())
            input_data = self.compute_graph.get_tensor_by_name('input_data:0')
            output = self.compute_graph.get_tensor_by_name('output_layer:0')
            result = sess.run(output, feed_dict={input_data : data})
        return result

    def mutate(self):
        '''
        Mutates a each layer in the network.
        '''
        with tf.Session(graph=self.init_graph) as sess:
            sess.run(tf.global_variables_initializer())
            #create a list of matrices representing the weights for each layer
            weight_list = [self.hidden_1_layer['weights'], self.hidden_2_layer['weights'], self.output_layer['weights']]
            #for each matrix, transform a random entry
            for weight_matrix in weight_list:
                i, j = weight_matrix.shape
                for k in range(0,mutate_layer_count):
                    rand_i = random.randrange(0,i)
                    rand_j = random.randrange(0,j)
                    weight_matrix[rand_i][rand_j] = np.random.normal()


    def mate(self, other_nn):
        '''
        Mates with another neural network to create a child.
        :param other_nn: The other parent neural network.
        :return: Child network consisting of parts of both parent networks.
        '''

        #TODO: Implement this cleaner! Lazy coding
        new_layer_1_weights = np.ndarray(shape=[n_inputs, n_nodes_hl1], dtype=np.float32)
        new_layer_1_biases = np.ndarray(shape=[1, n_nodes_hl1], dtype=np.float32)
        new_layer_2_weights = np.ndarray(shape=[n_nodes_hl1, n_nodes_hl2], dtype=np.float32)
        new_layer_2_biases = np.ndarray(shape=[1, n_nodes_hl2], dtype=np.float32)
        new_out_layer_weights = np.ndarray(shape=[n_nodes_hl2, n_classes], dtype=np.float32)
        new_out_layer_biases = np.ndarray(shape=[1, n_classes], dtype=np.float32)
        new_layer_attributes_list = [(new_layer_1_weights, self.hidden_1_layer['weights'], other_nn.hidden_1_layer['weights']),
                                     (new_layer_1_biases, self.hidden_1_layer['biases'].reshape([1, n_nodes_hl1]), other_nn.hidden_1_layer['biases'].reshape([1, n_nodes_hl1])),
                                     (new_layer_2_weights, self.hidden_2_layer['weights'], other_nn.hidden_2_layer['weights']),
                                     (new_layer_2_biases, self.hidden_2_layer['biases'].reshape([1, n_nodes_hl2]), other_nn.hidden_2_layer['biases'].reshape([1, n_nodes_hl2])),
                                     (new_out_layer_weights, self.output_layer['weights'], other_nn.output_layer['weights']),
                                     (new_out_layer_biases, self.output_layer['biases'].reshape([1, n_classes]), other_nn.output_layer['biases'].reshape([1, n_classes]))]

        for (layer_attributes, parent_1_attributes, parent_2_attributes) in new_layer_attributes_list:
            dim_1, dim_2 = layer_attributes.shape
            for i in range(0, dim_1):
                for j in range(0, dim_2):
                    random_bool = random.getrandbits(1)
                    layer_attributes[i][j] = parent_1_attributes[i][j] if random_bool else parent_2_attributes[i][j]
        new_layer_1 = {'weights' : new_layer_1_weights, 'biases' : new_layer_1_biases}
        new_layer_2 = {'weights': new_layer_2_weights, 'biases': new_layer_2_biases}
        new_output_layer = {'weights': new_out_layer_weights, 'biases': new_out_layer_biases}

        child_network = NeuralNetwork(hidden_1_layer=new_layer_1, hidden_2_layer=new_layer_2, output_layer=new_output_layer)
        return child_network


if __name__ == '__main__':
    algo = GeneticAlgo()
    algo.run_gui()