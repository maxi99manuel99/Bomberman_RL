import os
import pickle
import queue
import random
from types import new_class
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from collections import namedtuple, deque
from typing import Tuple
import timeit

import settings as s
import matplotlib.pyplot as plt

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

    #feature dimension
    self.D = 4

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.regression_forests = [RandomForestRegressor() for i in range(len(ACTIONS))]

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.regression_forests = pickle.load(file)

    self.obstacles = build_obstacle_indices()

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
    
    if self.train and (random.random() < random_prob or game_state['round'] < 100):
        self.logger.debug("Choosing action purely at random.")
        #80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    
    #approximate Q by regression forest
    Q =  [self.regression_forests[action_idx_to_test].predict([state_to_features(self, game_state)]) for action_idx_to_test in range(len(ACTIONS))]

    #our policy is the argmax of the approximated Q-function
    action_idx = np.argmax(Q)
    #self.logger.debug("Querying model for action.")

    #print(ACTIONS[action_idx])
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

    features = np.zeros(self.D)

    field = np.array(game_state['field'])

    coins = np.array(game_state['coins'])
    bombs = game_state['bombs']
    explosion_map = np.array(game_state['explosion_map'])
    _, _, bomb, (x, y) = game_state['self']
    others = game_state['others']
    others_position = np.zeros( (len(others), 2), dtype=int)

    for i,opponent in enumerate(others):
        others_position[i] = opponent[3]

    explosion_indices = np.array(np.where(explosion_map > 0)).T

    
    #the next feature is gonna be the direction we need to move to get to the next coin the fastest
    #if len(coins) != 0:
    #    features[0:4] =  breath_first_search(field.copy(), np.array([x,y]), coins, bombs , explosion_indices, others_position)
    #else:
    #    features[0:4] = np.array([0,0,0,0])
    
    if len(coins)>=1:
        targets = coins[0:1]

        potential_field = np.zeros((17,17))

        k_atr = 1
        k_str = 8
        k_rep = 30
        po = 3
        ps = 4


        idx = np.where(field != -1)
        k = np.where(field == -1)
        a = idx[0]
        b = idx[1]
        c = np.vstack((a,b)).T
        potential_field[k] = -10000

        dist_t = c[None,:] - targets[:, None]
        dist_o = c[None,:] - self.obstacles[:,None]

        norms_t = np.linalg.norm(dist_t, axis = 2)
        squared_norm_t = np.square(norms_t)
        summing_norms_t = np.sum(squared_norm_t, axis = 0)

        norms_o = np.linalg.norm(dist_o, axis = 2)

        norms_t_new = np.reshape(norms_t, (norms_t[0].shape[0]*len(targets)))
        norms_o_new = np.reshape(norms_o, (norms_o[0].shape[0]*len(self.obstacles)))

        dist_t = np.where(norms_t_new <= ps)[0]
        dist_o = np.where(norms_o_new <= po)[0]

        potential_field[idx] = potential_field[idx] + k_atr * 0.5 * summing_norms_t
        
        for i in dist_t:
            index = np.array([idx[0][i % norms_t[0].shape[0]], idx[1][i % norms_t[0].shape[0]]])
            potential_field[index[0], index[1]] = potential_field[index[0], index[1]] - np.square(k_str * 0.5 * (ps - norms_t_new[i]))

        for i in dist_o:
            index = np.array([idx[0][i % norms_o[0].shape[0]], idx[1][i % norms_o[0].shape[0]]])
            potential_field[index[0], index[1]] = potential_field[index[0], index[1]] + k_rep * 0.5 * np.square((1 / norms_o_new[i] - 1 / po))

        maximum_pot = np.max(potential_field)
                    
        potential_field[np.where(potential_field == -10000)] = maximum_pot + 10
        maximum_pot = maximum_pot + 10

        minimum_pot = np.min(potential_field)

        potential_field = (potential_field - minimum_pot) / (maximum_pot - minimum_pot)

        potential_field = np.round(potential_field, 3)

        potential_field[:,0]=1
        potential_field[0,:]=1
        potential_field[:,16]=1
        potential_field[16,:]=1
        
        if game_state['step'] == 10:
            # print(potential_field.T)
            plt.imshow(potential_field.T)
            plt.show()
        

        a = np.ones((5,5))
        for i in range(5):
            for j in range(5):
                if x-2+i>=0 and x-2+i < s.COLS and y-2+j>=0 and y-2+j < s.ROWS:
                    a[i,j] = potential_field[x-2+i][y-2+j]
        
        a = np.zeros(4)

        a[0] = potential_field[x-1][y]
        a[1] = potential_field[x+1][y]
        a[2] = potential_field[x][y-1]
        a[3] = potential_field[x][y+1]

        i = np.argmin(a)

        features[i] = 1

        #features = a.T.flatten()
    print(features)

    #print(features.reshape((5,5)))
    
    #print(features.reshape((3,3)))
    return features


def build_obstacle_indices():
    a = np.arange(17)
    b = np.tile(np.array([0]), 17)
    e = np.tile(np.array([16]), 17)

    c = np.vstack((a,b)).T
    d = np.vstack((b,a)).T

    f = np.vstack((a,e)).T
    g = np.vstack((e,a)).T

    h = np.vstack((c,d))
    h = np.vstack((h,f))
    h = np.vstack((h,g))
    blib = np.unique(h, axis = 0)

    x = np.arange(2,16,2)
    n_1 = np.tile(np.array([2]),7)
    n_2 = np.tile(np.array([4]),7)
    n_3 = np.tile(np.array([6]),7)
    n_4 = np.tile(np.array([8]),7)
    n_5 = np.tile(np.array([10]),7)
    n_6 = np.tile(np.array([12]),7)
    n_7 = np.tile(np.array([14]),7)


    c = np.vstack((n_1,x)).T
    d = np.vstack((n_2,x)).T
    e = np.vstack((n_3,x)).T
    f = np.vstack((n_4,x)).T
    g = np.vstack((n_5,x)).T
    h = np.vstack((n_6,x)).T
    i = np.vstack((n_7,x)).T

    new = np.vstack((c,d))
    new = np.vstack((new,e))
    new = np.vstack((new,f))
    new = np.vstack((new,g))
    new = np.vstack((new,h))
    new = np.vstack((new,i))

    obstacles = new #np.vstack((blib,new))

    return obstacles

#perform a breadth first sreach to get the direction to nearest coin
def breath_first_search(field: np.array, starting_point: np.array, targets:np.array, bombs , explosion_indices, opponents)->np.array: 
    """
    :param: field: Describes the current copied game board. 0 stays for free tile, 1 for stone walls and -1 for crates
            starting_point: The current position of the player. The position is a 1 dimensional value and the flattend version of the 2D field
    :return: np.array
    """

    #update the field so that the bombs are included
    for bomb in bombs:
        field[bomb[0][0],bomb[0][1]] = -1
    
    #update the field so that opponents are included
    for opponent in opponents:
        field[opponent[0],opponent[1]] = -1

    #update the field so that the explosions are included
    for explosion_index in explosion_indices:
        field[explosion_index[0], explosion_index[1]] = -1

    #update the field so that the coins are included
    for target in targets:
        field[target[0],target[1]] = 2
    
    #Use this list to get backtrace the path from the nearest coin to the players position to get the direction
    parent = np.ones(s.COLS*s.ROWS) * -1

    #Queue for visiting the tiles. 
    start = starting_point[1] * s.COLS + starting_point[0]
    parent[start] = start

    #check if the player is already in a coin field
    if field[starting_point[0],starting_point[1]] == 2:
        return np.array([0,0,0,0])
    
    path_queue = np.array([start])
    counter = 0
    
    coin_reached = False
    target = None

    #Visit tiles until a tile contains a coin
    while not coin_reached and counter < len(path_queue):
        current_position = path_queue[counter]
        
        #get the 2D coordinates
        x = current_position % s.COLS
        y = current_position // s.COLS

        #check if we reached a coin
        if field[x, y] == 2:
            coin_reached = True
            target = current_position
        
        else:
            #left from the current position. 
            if current_position % s.COLS != 0 and field[x-1,y]!= -1 and parent[current_position-1] == -1:
                path_queue = np.append(path_queue, current_position-1)
                parent[current_position-1] = current_position

            #right from the current position
            if current_position % s.COLS != s.COLS-1 and field[x+1,y]!= -1 and parent[current_position+1] == -1:
                path_queue = np.append(path_queue, current_position+1)
                parent[current_position+1] = current_position

            #up from the current position
            if current_position >= s.COLS and field[x,y-1]!= -1 and parent[current_position- s.COLS] == -1:
                path_queue = np.append(path_queue,current_position- s.COLS)
                parent[current_position- s.COLS] = current_position
 
            #down from the current position
            if y < s.ROWS-1 and field[x,y+1] != -1 and parent[current_position+ s.COLS] == -1:
                path_queue = np.append(path_queue,current_position+ s.COLS)
                parent[current_position+ s.COLS] = current_position
                
        #increase counter to get the next tile from the queue
        counter = counter + 1
    
    if target is not None:
        #get the path from the nearest coin to the player by accessing the parent list
        path = [target]
        tile = target
        
        while tile != start:
            tile = int(parent[tile])
            path.append(tile)

        path = np.flip(path)

        path_length = len(path)

        #get the second tile of the path which indicates the direction to the nearest coin
        next_position = path[1]
        next_position_x = path[1] % s.COLS
        next_position_y = path[1] // s.COLS

        direction = [int(next_position_x < starting_point[0]), int(next_position_x > starting_point[0]), int(next_position_y < starting_point[1]), int(next_position_y > starting_point[1])] 

        return direction
    else:
        return np.array([0,0,0,0])
