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
    self.D = 13
    self.num_actions = 6
    self.previous_move = np.array([0,0,0,0])

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
    Q = np.matmul(state_to_features(self,game_state), self.weights.T)

    #our policy is the argmax of the approximated Q-function
    action_idx = np.argmax(Q)

    #self.logger.debug("Querying model for action.")

    #Update the previous move(We use it for the features)
    if action_idx == 0:
        self.previous_move = np.array([0,0,1,0])
    elif action_idx == 1:
        self.previous_move = np.array([0,1,0,0])
    elif action_idx == 2:
        self.previous_move = np.array([0,0,0,1])
    elif action_idx == 3:
        self.previous_move = np.array([1,0,0,0])

    return ACTIONS[action_idx]


def state_to_features(self, game_state: dict) -> np.array:
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
    
    D = 13
    features = np.zeros(D)

    field = np.array(game_state['field'])
    coins = np.array(game_state['coins'])
    _, _, _, (x, y) = game_state['self']

    #we will normalize all features to [0,1]
    #our first 4 features are gonna be, what field type is around us
    features[0:4] = (np.array([field[y,x+1], field[y,x-1], field[y+1,x], field[y-1,x]]) +1) / 2

    #the next feature is gonna be the timestep we are currently in
    features[4] = game_state['step'] / s.MAX_STEPS

    if len(coins) != 0:
        features[5:9] = breath_first_search(field.copy(), np.array([x,y]), coins)
    else:
        features[5:9] = np.array([0,0,0,0])

    features[9:13] = self.previous_move

    return features

    '''# For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)'''

#perform a breadth first sreach to get the direction to nearest coin
def breath_first_search(field: np.array, starting_point: np.array, coins:np.array)->np.array: 
    """
    :param: field: Describes the current game board. 0 stays for free tile, 1 for stone walls and -1 for crates
            starting_point: The current position of the player. The position is a 1 dimensional value and the flattend version of the 2D field
    :return: np.array
    """

    #update the field so that the coins are included
    for coin in coins:
        field[coin[1],coin[0]] = 2
    
    #Use this list to get backtrace the path from the nearest coin to the players position to get the direction
    parent = np.ones(s.WIDTH*s.HEIGHT) * -1
    

    #Queue for visiting the tiles. 
    start = starting_point[1] * s.WIDTH + starting_point[0]
    parent[start] = start

    #check if the player is already in a coin field
    if field[starting_point[1],starting_point[0]] == 2:
        return np.array([0,0,0,0])
    
    path_queue = np.array([start])
    counter = 0
    
    coin_reached = False
    target = None

    #Visit tiles until a tile contains a coin
    while not coin_reached:
        current_position = path_queue[counter]
        
        #get the 2D coordinates
        x = current_position % s.WIDTH
        y = current_position // s.WIDTH
        
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
                
        #increase counter to get the next tile from the queue
        counter = counter + 1
    
    #get the path from the nearest coin to the player by accessing the parent list
    path = [target]
    tile = target
    
    while tile != start:
        tile = int(parent[tile])
        path.append(tile)

    path = np.flip(path)

    #get the second tile of the path which indicates the direction to the nearest coin
    next_position = path[1]
    next_position_x = path[1] % s.WIDTH
    next_position_y = path[1] // s.WIDTH

    #use one-hot-encoding: [LEFT, RIGHT, UP, DOWN]
    direction = [int(next_position_x < starting_point[0]), int(next_position_x > starting_point[0]), int(next_position_y < starting_point[1]), int(next_position_y > starting_point[1])]
    
    return direction





