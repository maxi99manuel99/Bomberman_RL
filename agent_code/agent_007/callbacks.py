import os
import pickle
import queue
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from collections import namedtuple, deque
from typing import Tuple

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
    self.D = 28
    self.previous_move = np.array([0,0,0,0])

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #self.weights = np.zeros((len(ACTIONS), self.D))
        self.regression_forests = [RandomForestRegressor() for i in range(len(ACTIONS))]

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.regression_forests = pickle.load(file)


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

    if self.train and (random.random() < random_prob or game_state['round'] < 500):
        self.logger.debug("Choosing action purely at random.")
        #80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    
    
    #approximate Q by linear regression
    #Q = np.matmul(state_to_features(self,game_state), self.weights.T)

    #approximate Q by regression forest
    Q =  [self.regression_forests[action_idx_to_test].predict([state_to_features(self, game_state)]) for action_idx_to_test in range(len(ACTIONS))]

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
    #return "WAIT"

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
    ## This is the dict before the game begins and after it ends
    #if game_state is None:
    #    return None
    
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

    #the next feature is gonna be if there is a crate in the near and we can find an escape route, so that it is sensible to drop a bomb
    explosion_indices = np.array(np.where(explosion_map > 0)).T
    crate_in_the_near, number_near_crates = check_for_crates(self, np.array([x,y]), field.copy())
    escape_possible, possible_escape_directions = check_escape_route(self, field.copy(), np.array([x,y]), explosion_indices, bombs)

    if crate_in_the_near:
        #print("crate is in the near")

        if escape_possible:
            #print("escape is possible")
            features[8] = 1
    
    #directions in which we should escape if we could plant a bomb here
    features[9:13] = possible_escape_directions

    features[9:13] = np.array([0,0,0,0])

    #the next feature is gonna be the amount of bombs that are in a certain radius of the player
    # weighted by their distance and the time till they explode
    #features[13:18] = search_for_bomb_in_radius(field.copy(), np.array([x,y]), bombs, 5)

    features[13:18] = danger(self, bombs, field, explosion_indices, x, y)

    #the next feature is gonna be if there is an explosion anywhere around us and how long its gonna stay
    features[18:22] = (np.array([explosion_map[x-1,y], explosion_map[x+1,y], explosion_map[x,y-1], explosion_map[x,y+1]])) / s.EXPLOSION_TIMER

    
    features[22:26] = np.array([ int(field[x-1,y] == 0) , int(field[x+1,y] == 0), int(field[x,y-1] == 0), int(field[x,y+1] == 0) ])

    #the next feature is gonna show if a bomb action is possible
    features[26] = int(bomb)

    #the next feature is gonna be how many crates are in our bomb radius right now
    features[27] = number_near_crates

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
    :param: field: Describes the current copied game board. 0 stays for free tile, 1 for stone walls and -1 for crates
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
def search_for_bomb_in_radius(field: np.array, starting_point: np.array, bombs: list, radius: int) -> np.array:
    """Searches for bombs in a certain radius around the player and
       weights them by how long they need till they explode and how far away they are


    Args:
        field (np.array): The whole copied game field
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
        if field[x, y] >= 10:
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
      return np.array([0,0,0,0,0])

    directions = np.zeros( (len(bomb_position), 5) ) 

    for i,pos in enumerate(bomb_position):

      if pos == start:
          directions[i] = np.array([0,0,0,0,1])
          continue

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

      value = 1 - ( distance[pos] * (bomb_countdowns[i]-10) ) / (radius * s.BOMB_TIMER)
      #print(distance[pos])
      #print(bomb_countdowns[i]-10)
      #print(value)
      #print(s.BOMB_TIMER)
      
      direction = np.array([int(next_position_x < starting_point[0]), int(next_position_x > starting_point[0]), int(next_position_y < starting_point[1]), int(next_position_y > starting_point[1]), 0]) * value
      directions[i] = direction

    direction = np.max(directions, axis=0)

    return direction

def check_for_crates(self, agent_position: np.array , field: np.array) -> bool:
    x, y = agent_position
    check_left, check_right, check_up, check_down =  np.array([field[x-1,y] != -1 , field[x+1,y] != -1, field[x,y-1] != -1, field[x,y+1] != -1])
    any_chest_found = False
    chest_counter = 0

    if check_right:
        for i in range(1,4):
            if x+i >= s.COLS:
                break 

            if field[x+i,y] == 1:
                any_chest_found = True
                chest_counter = chest_counter + 1

    if check_left:
        for i in range(1,4):
            if x-i < 0:
                break

            if field[x-i,y] == 1:
                any_chest_found = True
                chest_counter = chest_counter + 1

    if check_down:
        for i in range(1,4):
            if y+i >= s.ROWS:
                break

            if field[x,y+i] == 1:
                any_chest_found = True
                chest_counter = chest_counter +1

    if check_up:
        for i in range(1,4):
            if y-i < 0:
                break 
            
            if field[x,y-i] == 1:
                any_chest_found = True
                chest_counter = chest_counter + 1
    
    chest_counter = chest_counter / 9 #since 9 is the maximum number of chests you can destroy with one bomb
                                      #(appart from sitting in the middle of 12 bombs which is not a sensible case of the game)

    return any_chest_found, chest_counter

def check_escape_route(self, field: np.array, starting_point: np.array, explosion_indices: np.array, bombs: list) -> Tuple[bool, np.array]:
    """Tries to find an escape route after setting a bomb in the near of a crate. If there is an escape route
    then the feature flag will be set to 1 otherwise to 0 


    Args:
        field (np.array): The whole copied game field
        starting_point (np.array): current player position
        explosion_indices (np.array): location of explosions
        bombs (list): location of bombs

    Returns:
        bool: True if there was an escape route found
        np.array: All directions the agent can move in to follow one of the found escape paths
    """

    #update the field so that the explosions are included, treated like walls so you cant escape that way
    for explosion_index in explosion_indices:
        field[explosion_index[0], explosion_index[1]] = -1

    #update field so bombs are included, treated like walls so you cant escape that way
    for bomb in bombs:
        field[bomb[0][0],bomb[0][1]] = -1

        
    #distance from start position to current position
    distance = np.zeros(s.COLS*s.ROWS-1)

    #Use this list to get backtrace the path from the nearest coin to the players position to get the direction
    parent = np.ones(s.COLS*s.ROWS) * -1

    #Queue for visiting the tiles. 
    start = starting_point[1] * s.COLS + starting_point[0]
    
    parent[start] = start

    #queue to iterate over the tiles
    path_queue = np.array([start])
    counter = 0

    directions_to_escape = np.zeros(4)
    escape_found = False

    while counter < len(path_queue):
        current_position = path_queue[counter]
        #print(current_position)
        
        #get the 2D coordinates
        x = current_position % s.COLS
        y = current_position // s.ROWS

        #if our distance to the bomb is far enough or we would get away from the explosion, then we instantly return true
        if distance[current_position] > 3 or (starting_point[0] != x and starting_point[1] != y):
            #print("escape found")
            pos = int(current_position)
            while True:
                if parent[pos] == start:
                    next_position_x = pos % s.COLS
                    next_position_y = pos // s.COLS
                    direction = np.array([int(next_position_x < starting_point[0]), int(next_position_x > starting_point[0]), int(next_position_y < starting_point[1]), int(next_position_y > starting_point[1])])
                    directions_to_escape = np.max(np.vstack((direction, directions_to_escape)), axis=0)

                    break

                pos = int(parent[pos])
                

                
            escape_found = True
        
        else:
            #left from the current position. 
            if current_position % s.COLS != 0 and field[x-1,y] != -1 and field[x-1,y] != 1 and field[x-1,y]!= -1 and parent[current_position-1] == -1:
                path_queue = np.append(path_queue, current_position-1)
                parent[current_position-1] = current_position
                distance[current_position-1] = distance[current_position] +1

            #right from the current position
            if current_position % s.COLS != s.COLS-1 and field[x+1,y] != -1 and field[x+1,y]!=1 and field[x+1,y]!= -1 and parent[current_position+1] == -1:
                path_queue = np.append(path_queue, current_position+1)
                parent[current_position+1] = current_position
                distance[current_position+1] = distance[current_position] +1

            #up from the current position
            if current_position >= s.COLS and field[x,y-1] != -1 and field[x, y-1]!= 1 and field[x,y-1]!= -1 and parent[current_position-s.COLS] == -1:
                path_queue = np.append(path_queue,current_position-s.COLS)
                parent[current_position-s.COLS] = current_position
                distance[current_position-s.COLS] = distance[current_position] +1
 
            #down from the current position
            if y < s.ROWS-1 and field[x,y+1] != -1 and field[x, y+1]!= 1 and field[x,y+1] != -1 and parent[current_position+s.COLS] == -1:
                path_queue = np.append(path_queue,current_position+s.COLS)
                parent[current_position+s.COLS] = current_position
                distance[current_position+s.COLS] = distance[current_position] +1

        counter = counter + 1

    return escape_found, directions_to_escape


def checking(self, field: np.array, starting_point: np.array, explosion_indices: np.array, bombs: list, distance, bomb_pos, countdown) -> Tuple[bool, np.array]:
    
    #print("Printing distances")
    #print(np.reshape(distance, (s.ROWS, s.COLS)))
    #update the field so that the explosions are included, treated like walls so you cant escape that way
    for explosion_index in explosion_indices:
        field[explosion_index[0], explosion_index[1]] = -1

    #update field so bombs are included, treated like walls so you cant escape that way
    for bomb in bombs:
        field[bomb[0][0],bomb[0][1]] = -1

    #Use this list to get backtrace the path from the nearest coin to the players position to get the direction
    parent = np.ones(s.COLS*s.ROWS) * -1

    countdowns = np.ones(s.COLS * s.ROWS) * countdown

    #Queue for visiting the tiles. 
    start = starting_point[1] * s.COLS + starting_point[0]
    
    parent[start] = start

    #queue to iterate over the tiles
    path_queue = np.array([start])
    counter = 0

    directions_to_escape = np.zeros(4)
    escape_found = False

    while counter < len(path_queue):
        current_position = path_queue[counter]

        #get the 2D coordinates
        x = current_position % s.COLS
        y = current_position // s.ROWS

        #print("current position")
        #print(f" {x} {y}")
        #print("distance of current position")
        #print(distance[current_position])
        #print("Countdown")
        #print(countdowns[current_position])

        #if our distance to the bomb is far enough or we would get away from the explosion, then we instantly return true
        if distance[current_position] > 3 or (bomb_pos[0] != x and y != bomb_pos[1]):
            #print("YUHU, ESCAPE FOUND")
            pos = int(current_position)
            while True:
                if parent[pos] == start:
                    next_position_x = pos % s.COLS
                    next_position_y = pos // s.COLS
                    direction = np.array([int(next_position_x < starting_point[0]), int(next_position_x > starting_point[0]), int(next_position_y < starting_point[1]), int(next_position_y > starting_point[1])])
                    directions_to_escape = np.max(np.vstack((direction, directions_to_escape)), axis=0)

                    break

                pos = int(parent[pos])
                

                
            escape_found = True
        
        else:
            if countdowns[current_position] == 0:
                counter = counter + 1
                #print("Countdown already reached")
                continue

            #left from the current position. 
            if current_position % s.COLS != 0 and field[x-1,y] != -1 and field[x-1,y] != 1 and field[x-1,y]!= -1 and parent[current_position-1] == -1:
                path_queue = np.append(path_queue, current_position-1)
                parent[current_position-1] = current_position
                if distance[current_position-1] == 0:
                    distance[current_position-1] = distance[current_position] +1
                countdowns[current_position-1] = countdowns[current_position] -1

            #right from the current position
            if current_position % s.COLS != s.COLS-1 and field[x+1,y] != -1 and field[x+1,y]!=1 and field[x+1,y]!= -1 and parent[current_position+1] == -1:
                path_queue = np.append(path_queue, current_position+1)
                parent[current_position+1] = current_position
                if distance[current_position+1] == 0:
                    distance[current_position+1] = distance[current_position] +1
                countdowns[current_position+1] = countdowns[current_position] - 1

            #up from the current position
            if current_position >= s.COLS and field[x,y-1] != -1 and field[x, y-1]!= 1 and field[x,y-1]!= -1 and parent[current_position-s.COLS] == -1:
                path_queue = np.append(path_queue,current_position-s.COLS)
                parent[current_position-s.COLS] = current_position
                if distance[current_position-s.COLS] == 0:
                    distance[current_position-s.COLS] = distance[current_position] +1
                countdowns[current_position-s.COLS] = countdowns[current_position] - 1
 
            #down from the current position
            if y < s.ROWS-1 and field[x,y+1] != -1 and field[x, y+1]!= 1 and field[x,y+1] != -1 and parent[current_position+s.COLS] == -1:
                path_queue = np.append(path_queue,current_position+s.COLS)
                parent[current_position+s.COLS] = current_position
                if distance[current_position+s.COLS]== 0:
                    distance[current_position+s.COLS] = distance[current_position] +1
                countdowns[current_position+s.COLS] = countdowns[current_position] - 1

        counter = counter + 1

    return escape_found, directions_to_escape

                       
def danger(self, bombs, field, explosion_indices, x, y):
    test = np.array([0,0,0,0,0])
    
    """
    for bomb in bombs:
        field[bomb[0][0],bomb[0][1]] = -1

    #first check if steps in the different directions are possible
    #or if there are blocking objects
    #free fields have label 0
    free_fields = np.array([ int(field[x-1,y] == 0) , int(field[x+1,y] == 0), int(field[x,y-1] == 0), int(field[x,y+1] == 0), 1])
    test = np.min(np.vstack((test, free_fields)), axis=0)
    """

    #first check if there are bombs
    if len(bombs) != 0:
        #print("There are bombs in the field")
        #collect the number of positions and bombs
        bomb_positions= np.zeros((len(bombs),2))
        bomb_countdowns = np.zeros(len(bombs))

        for i, bomb in enumerate(bombs):
            bomb_positions[i] = np.array(bomb[0]) 
            bomb_countdowns[i] = bomb[1]
        
        #get all bombs, which are in the near
        x_near =np.where( np.logical_and( np.abs(bomb_positions[:, 0] - x) <=1 , np.abs(bomb_positions[:, 1] - y) <=4 ) )
        y_near =np.where( np.logical_and( np.abs(bomb_positions[:, 1] - y) <=1 , np.abs(bomb_positions[:, 0] - x) <=4 ) )

        #print("count")
        #print(bomb_countdowns)
        bomb_x = bomb_positions[x_near]
        bomb_x_count_down = bomb_countdowns[x_near]

        bomb_y = bomb_positions[y_near]
        bomb_y_count_down = bomb_countdowns[y_near]

        near = np.vstack([bomb_x, bomb_y])
        near = np.unique(near, axis = 0)

        #check if there are bombs in the near
        if near.shape[0] > 0:
            #print("There are bombs in the near")
            #print(bomb_x)
            #print(bomb_y)

            #print("There are bombs in the near")
            for i,pos in enumerate(bomb_x) :

                f = field.copy()
                #print(f.T)

                index = np.where(field == 0)
                f[index] = 10
 
                distance = np.zeros(s.COLS*s.ROWS)

                for j in range(int(pos[1])-1, max(int(pos[1]) - 4, 0), -1):
                    if f[int(pos[0]),j] == -1:
                        break
                    else: 
                        f[int(pos[0]), j] = bomb_x_count_down[i]

                        #update the distance
                        flatten =  j * s.COLS + int(pos[0])
                        distance[flatten] = int(pos[1]) - j
                        


                for j in range(int(pos[1])+1, min(int(pos[1])+4, s.ROWS-1)):
                    if f[int(pos[0]),j] == -1:
                        break
                    else: 
                        f[int(pos[0]), j] = bomb_x_count_down[i]- 1 

                        #update the distance
                        flatten =  j * s.COLS + int(pos[0])
                        distance[flatten] = j - int(pos[1])
            
                #print(np.reshape(distance, (s.ROWS, s.COLS)))

                #check for escape
                if (f[x, y] != 10) and test[4] != 1:
                    #print("Check if you can wait")
                    escape, _ = checking(self, field, np.array([x, y]), explosion_indices, bombs, distance , pos, bomb_x_count_down[i])
                    if(not escape):
                        #print("You can't wait")
                        test[4] = 1
                    else:
                        pass
                        #print("You can wait")

                if (f[x, y+1] != 10 and field[x, y+1] != 1 and field[x, y+1] != -1) and test[3] != 1:
                    #print("Check if you can move down")
                    escape, _ = checking(self, field, np.array([x, y+1]), explosion_indices, bombs, distance , pos, bomb_x_count_down[i])
                    if(not escape):
                        #print("You can't move down")
                        test[3] = 1
                    else:
                        pass
                        #print("You can move down")

                if (f[x, y-1] != 10 and field[x, y-1] != 1 and field[x, y-1] != -1) and test[2] != 1:
                    #print("Check if you can move up")
                    escape, _ = checking(self, field, np.array([x, y-1]), explosion_indices, bombs, distance , pos, bomb_x_count_down[i])
                    if(not escape):
                        #print("You can't move up")
                        test[2] = 1
                    else:
                        pass
                        #print("You can move up")

                if (f[x+1, y] != 10 and field[x+1, y] != 1 and field[x+1, y] != -1) and test[1] != 1:
                    #print("Check if you can move right")
                    escape, _ = checking(self, field, np.array([x+1, y]), explosion_indices, bombs, distance , pos, bomb_x_count_down[i])
                    if(not escape):
                        #print("You can't move right")
                        test[1] = 1
                    else:
                        pass
                        #print("You can move right")

                if (f[x-1, y] != 10 and field[x-1, y] != 1 and field[x-1, y] != -1) and test[0] != 1:
                    #print("Check if you can move left")
                    escape, _ = checking(self, field, np.array([x-1, y]), explosion_indices, bombs, distance , pos, bomb_x_count_down[i])
                    if(not escape):
                        #print("You can't move left")
                        test[0]= 1
                    else:
                        pass
                        #print("You can move left")
            
            #print("hallo-------------------------------------------------------")
            for i,pos in enumerate(bomb_y) :

                f = field.copy()
                #print(f.T)

                index = np.where(field == 0)
                f[index] = 10
 
                distance = np.zeros(s.COLS*s.ROWS)

                for j in range(int(pos[0])-1, max(int(pos[0]) - 4, 0), -1):
                    if f[j, int(pos[1])] == -1:
                        break
                    else: 
                        f[j,int(pos[1])] = bomb_y_count_down[i]

                        #update the distance
                        flatten =  int(pos[1]) * s.COLS + j
                        distance[flatten] = int(pos[0]) - j
                        

                for j in range(int(pos[0])+1, min(int(pos[0])+4, s.COLS-1)):
                    #print(int(pos[0]))
                    #print(s.COLS-1)
                    #print("sdjsijdisjdisdjis")
                    #print(j)
                    if f[j, int(pos[1])] == -1:
                        break
                    else: 
                        f[j, int(pos[1])] = bomb_y_count_down[i]

                        #update the distance
                        flatten =  int(pos[1]) * s.COLS + j
                        distance[flatten] = j - int(pos[0])
            
                #print(np.reshape(distance, (s.ROWS, s.COLS)))
                #print(f.T)

                #check for escape
                if (f[x, y] != 10) and test[4] != 1:
                    #print("Check if you can wait")
                    escape, _ = checking(self, field, np.array([x, y]), explosion_indices, bombs, distance , pos, bomb_y_count_down[i])
                    if(not escape):
                        #print("You can't wait")
                        test[4] = 1
                    else:
                        pass
                        #print("You can wait")

                if (f[x, y+1] != 10 and field[x, y+1] != 1 and field[x, y+1] != -1) and test[3] != 1:
                    #print("Check if you can move down")
                    escape, _ = checking(self, field, np.array([x, y+1]), explosion_indices, bombs, distance , pos, bomb_y_count_down[i])
                    if(not escape):
                        #print("You can't move down")
                        test[3] = 1
                    else:
                        pass
                        #print("You can move down")

                if (f[x, y-1] != 10 and field[x, y-1] != 1 and field[x, y-1] != -1) and test[2] != 1:
                    #print("Check if you can move up")
                    escape, _ = checking(self, field, np.array([x, y-1]), explosion_indices, bombs, distance , pos, bomb_y_count_down[i])
                    if(not escape):
                        #print("You can't move up")
                        test[2] = 1
                    else:
                        pass
                        #print("You can move up")

                if (f[x+1, y] != 10 and field[x+1, y] != 1 and field[x+1, y] != -1) and test[1] != 1:
                    #print("Check if you can move right")
                    escape, _ = checking(self, field, np.array([x+1, y]), explosion_indices, bombs, distance , pos, bomb_y_count_down[i])
                    if(not escape):
                        #print("You can't move right")
                        test[1] = 1
                    else:
                        pass
                        #print("You can move right")

                if (f[x-1, y] != 10 and field[x-1, y] != 1 and field[x-1, y] != -1) and test[0] != 1:
                    #print("Check if you can move left")
                    escape, _ = checking(self, field, np.array([x-1, y]), explosion_indices, bombs, distance , pos, bomb_y_count_down[i])
                    if(not escape):
                        #print("You can't move left")
                        test[0] = 1
                    else:
                        pass
                        #print("You can move left")
    
    return test
   