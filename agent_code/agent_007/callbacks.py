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
    self.D = 21
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
    
    #if game_state['round'] <= 10:
    #    random_prob = 0.4

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
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    features = np.zeros(self.D)

    field = np.array(game_state['field'])

    coins = np.array(game_state['coins'])
    bombs = game_state['bombs']
    explosion_map = np.array(game_state['explosion_map'])
    _, _, bomb, (x, y) = game_state['self']

    #print(field)

    #We will normalize all features to [0,1]
    
    #the next feature is gonna be the direction we need to move to get to the next coin the fastest
    if len(coins) != 0:
        features[0:4] = breath_first_search(field.copy(), np.array([x,y]), coins)
    else:
       features[0:4] = np.array([0,0,0,0])


    #the next feature is gonna be the direction we need to move to get to the next crate the fastest
    crate_indices = np.array(np.where(field == 1)).T

    if crate_indices.size != 0:
        features[4:8] = breath_first_search(field.copy(), np.array([x,y]), crate_indices)
    else:
        features[4:8] = np.array([0,0,0,0])

    #print(features[4:8])

    #the next feature is gonna be if there is a crate around the player
    features[8:12] = np.array([ int(field[x-1,y] == 1) , int(field[x+1,y] == 1), int(field[x,y-1] == 1), int(field[x,y+1] == 1) ])
    #print(features[8:12])
    #the next feature is gonna be if there is a wall around the player
    #features[12:16] = np.array([ int(field[x-1,y] == -1) , int(field[x+1,y] == -1), int(field[x,y-1] == -1), int(field[x,y+1] == -1) ])
    
    
    #features, which could be used
    
    #the next feature is gonna be the timestep we are currently in
    #features[4] = game_state['step'] / s.MAX_STEPS

    #the next feature is gonna be the amount of bombs that are in a certain radius of the player
    # weighted by their distance and the time till they explode
    features[12:16] = search_for_bomb_in_radius(field.copy(), np.array([x,y]), bombs, 5, 0.5)
    #print(features[12:16])

    #the next feature is gonna be if there is an explosion anywhere around us and how long its gonna stay
    features[16:20] = (np.array([explosion_map[x+1,y], explosion_map[x-1,y], explosion_map[x,y-1], explosion_map[x,y+1]])) / s.EXPLOSION_TIMER


    #the next feature is gonna show if a bomb action is possible
    features[20] = int(bomb)

    #the next feature is if we are standing on a bomb
    features[21] = int(len(np.array((np.where(bomb[:,0] == (x,y))))) == 0)


    return features


    ''' For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
    '''

#perform a breadth first sreach to get the direction to nearest coin
def breath_first_search(field: np.array, starting_point: np.array, targets:np.array)->np.array: 
    """
    :param: field: Describes the current game board. 0 stays for free tile, 1 for stone walls and -1 for crates
            starting_point: The current position of the player. The position is a 1 dimensional value and the flattend version of the 2D field
    :return: np.array
    """

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


#perform a breadth first sreach that searches for bombs in a given radius
def search_for_bomb_in_radius(field: np.array, starting_point: np.array, bombs: list, radius: int, eps: float) -> np.array:
    """Searches for bombs in a certain radius around the player and
       weights them by how long they need till they explode and how far away they are


    Args:
        field (np.array): The whole game field
        starting_point (np.array): current player position
        bombs (np.array): location of the bombs
        radius (int): radius to search for bombs
        eps (float): the factor that is responsible for amount of influence of the distance of the bomb

    Returns:
        np.array: The positions of the bomb weighted by distance and time they need to explode
    """

    #update the field so that the coins are included
    for bomb in bombs:
        field[bomb[0][0],bomb[0][1]] = bomb[1] + 10
        
    #Use this list to get backtrace the path from the nearest coin to the players position to get the direction
    parent = np.ones(s.COLS*s.ROWS) * -1

    #distance from start position to current position
    distance = np.zeros(s.COLS*s.ROWS-1)

    #Queue for visiting the tiles. 
    start = starting_point[1] * s.COLS + starting_point[0]
    
    parent[start] = start

    path_queue = np.array([start])
    counter = 0
    
    distance_overceeded = False

    #save the positions of the bombs which are in the given radius
    bomb_position = []
    bomb_countdowns = []


    #Visit tiles until a tile contains a coin
    while not distance_overceeded and counter < len(path_queue):
        current_position = path_queue[counter]

        if distance[current_position] > radius:
            distance_overceeded = True
            break

        x = current_position % s.COLS
        y = current_position // s.ROWS
        
        #check if we reached a coin
        if field[x, y] >= 10 and current_position != start:
            bomb_position.append(current_position)
            bomb_countdowns.append(field[x,y])
        
        else:
            #left from the current position. 
            if current_position % s.COLS != 0 and field[x-1,y] != 1 and field[x-1,y]!= -1 and parent[current_position-1] == -1:
                path_queue = np.append(path_queue, current_position-1)
                parent[current_position-1] = current_position
                distance[current_position-1] = distance[current_position] +1

            #right from the current position
            if current_position % s.COLS != s.COLS-1 and field[x+1,y]!=1 and field[x+1,y]!= -1 and parent[current_position+1] == -1:
                path_queue = np.append(path_queue, current_position+1)
                parent[current_position+1] = current_position
                distance[current_position+1] = distance[current_position] +1

            #up from the current position
            if current_position >= s.COLS and field[x, y-1]!= 1 and field[x,y-1]!= -1 and parent[current_position-s.COLS] == -1:
                path_queue = np.append(path_queue,current_position-s.COLS)
                parent[current_position-s.COLS] = current_position
                distance[current_position-s.COLS] = distance[current_position] +1
 
            #down from the current position
            if y < s.ROWS-1 and field[x, y+1]!= 1 and field[x,y+1] != -1 and parent[current_position+s.COLS] == -1:
                path_queue = np.append(path_queue,current_position+s.COLS)
                parent[current_position+s.COLS] = current_position
                distance[current_position+s.COLS] = distance[current_position] +1

        counter = counter + 1

    if len(bomb_position) == 0:
      return np.array([0,0,0,0])

    directions = np.zeros( (len(bomb_position), 4) ) 

    for i,pos in enumerate(bomb_position):
      #get the path from the nearest bomb to the player
      path = [pos]
      tile = pos

      while tile != start:
        tile = int(parent[tile])
        path.append(tile)

      path = np.flip(path)

      next_position = path[1]
      next_position_x = path[1] % s.COLS
      next_position_y = path[1] // s.COLS

      value = 1 - ( distance[pos] * (bomb_countdowns[i]-10) ) / (radius * s.BOMB_TIMER + eps)
      
      direction = np.array([int(next_position_x < starting_point[0]), int(next_position_x > starting_point[0]), int(next_position_y < starting_point[1]), int(next_position_y > starting_point[1])]) * value
      directions[i] = direction

    direction = np.max(directions, axis=0)

    return direction




