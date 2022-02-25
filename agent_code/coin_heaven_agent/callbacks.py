import os
import pickle
import queue
import random
import numpy as np
from collections import namedtuple, deque

import settings as s


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.D = 9
    self.num_actions = 6

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.weights = np.zeros((self.num_actions, self.D))

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.weights = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        #80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    
    #approximate Q by linear regression
    Q = np.matmul(state_to_features(game_state), self.weights.T)

    action_idx = np.argmax(Q)


    #self.logger.debug("Querying model for action.")
    return ACTIONS[action_idx]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    D = 9
    features = np.zeros(D)

    
    field = np.array(game_state['field'])
    coins = np.array(game_state['coins'])
    _, _, _, (x, y) = game_state['self']

    #print(f"{x} {y}")

    #we will normalize all features to [0,1]
    
    #our first 4 features are gonna be, what field type is around us
    features[0:4] = (np.array([field[y,x+1], field[y,x-1], field[y+1,x], field[y-1,x]]) +1) / 2

    #as a next feature we take our own position
    #features[4] = (y * s.WIDTH + x) / (s.WIDTH*s.HEIGHT)

    #the next feature is gonna be the timestep we are currently in
    features[4] = game_state['step'] / s.MAX_STEPS

    #our last feature is gonna be the direction to the nearest coin (connection vec)
    #connection_vec = (coins - np.array([x, y]))
    #shortest_vec_indx = np.argmin(np.linalg.norm(connection_vec, axis=1))

    #features[6] = (connection_vec[shortest_vec_indx][1] * s.WIDTH + connection_vec[shortest_vec_indx][0]) / (s.WIDTH*s.HEIGHT-1)

    features[5:9] = breath_first_search(field, np.array([x,y]), coins)

    return features

    '''# For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)'''

#perform a breadth first sreach to get the direction to nearest coin
def breath_first_search(field, starting_point, coins):
    """
    :param: field: Describes the current game board. 0 stays for free tile, 1 for stone walls and -1 for crates
            starting_point: The current position of the player. The position is a 1 dimensional value and the flattend version of the 2D field
    :return: np.array
    """
    
    #print("Original field")
    #print(field)
    
    #update the field so that the coins are included
    for coin in coins:
        field[coin[1],coin[0]] = 2
        
    #print("Field combined with coins")
    #print(field)
    
    #Use this list to get backtrace the path from the nearest coin to the players position to get the direction
    parent = np.ones(s.WIDTH*s.HEIGHT) * -1
    
    #print("Size of parents list")
    #print(parent.shape)

    #Queue for visiting the tiles. 
    start = starting_point[1] * s.WIDTH + starting_point[0]
    parent[start] = start
    
    #print("Start tile")
    #print(start)
    
    if field[starting_point[1],starting_point[0]] == 2:
        return np.array([0,0,0,0])
    
    path_queue = np.array([start])
    counter = 0
    
    coin_reached = False

    target = None

    #Visit tiles until a tile contains a coin
    while not coin_reached:
        current_position = path_queue[counter]
        
        #print(f"Current Position: {current_position}")

        x = current_position % s.WIDTH
        y = current_position // s.WIDTH
        
        #print(f"2D coordinate: {y} , {x}")
        
        #check if we reached a coin
        if field[y, x] == 2:
            #print("Coin reached")
            coin_reached = True
            target = current_position
        
        else:
            #left from the current position. 
            if current_position % s.WIDTH != 0 and field[y,x-1] != 1 and field[y,x-1]!= -1 and parent[current_position-1] == -1:
                path_queue = np.append(path_queue, current_position-1)
                parent[current_position-1] = current_position

            #right from the current position
            if current_position % s.WIDTH != s.WIDTH-1 and field[y,x+1]!=1 and field[y,x+1]!= -1 and parent[current_position+1] == -1:
                path_queue = np.append(path_queue, current_position+1)
                parent[current_position+1] = current_position

            #up from the current position
            if current_position >= s.WIDTH and field[y-1,x]!= 1 and field[y-1,x]!= -1 and parent[current_position- s.WIDTH] == -1:
                path_queue = np.append(path_queue,current_position- s.WIDTH)
                parent[current_position- s.WIDTH] = current_position
 
            #down from the current position
            if y < s.HEIGHT-1 and field[y+1,x]!= 1 and field[y+1,x] != -1 and parent[current_position+ s.WIDTH] == -1:
                path_queue = np.append(path_queue,current_position+ s.WIDTH)
                parent[current_position+ s.WIDTH] = current_position
                
        #print("Parent:")
        #print(parent.reshape(9,9))
        
        counter = counter + 1
    
    #print("Get Path")
    #get the path from the nearest coin to the player
    path = [target]
    tile = target
    
    #print(tile)
    while tile != start:
        tile = int(parent[tile])
        path.append(tile)
        #print(path)

    path = np.flip(path)

    #print(path)
    next_position = path[1]
    next_position_x = path[1] % s.WIDTH
    next_position_y = path[1] // s.WIDTH

    direction = [int(next_position_x < starting_point[0]), int(next_position_x > starting_point[0]), int(next_position_y < starting_point[1]), int(next_position_y > starting_point[1])]
    
    return direction





