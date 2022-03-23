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

    #feature dimension
    self.D = 19

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.regression_forests = [RandomForestRegressor(n_estimators=5) for i in range(len(ACTIONS))]

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

    coin_distance = np.inf
    crate_distance = np.inf
    opponent_distance = np.inf

    #the next feature is gonna be the direction we need to move to get to the next coin the fastest
    if len(coins) != 0:
        coin_directions, coin_distance = breath_first_search(field.copy(), np.array([x,y]), coins, bombs , explosion_indices, others_position)
    else:
       coin_directions = np.array([0,0,0,0])

    #the next feature is gonna be the direction we need to move to get to the next crate the fastest
    crate_indices = np.array(np.where(field == 1)).T

    if crate_indices.size != 0:
       crate_directions, crate_distance = breath_first_search(field.copy(), np.array([x,y]), crate_indices, bombs, explosion_indices, others_position)
    else:
        crate_directions = np.array([0,0,0,0])

    #the next feature is gonna be the direction we need to move to get to the next oponnent the fastest
    if len(others_position) != 0:
        opponent_directions, opponent_distance = breath_first_search(field.copy(), np.array([x,y]), others_position, bombs, explosion_indices, others_position)
    else:
        opponent_directions = np.array([0,0,0,0])

    #the next feature is gonna be if there is a crate or enemy in the near and we can find an escape route, so that it is sensible to drop a bomb
    if check_escape_route(self, field.copy(), np.array([x,y]), explosion_indices, bombs, others_position)[0]:
        if check_for_crates(self, np.array([x,y]), field.copy()) or check_for_opponents(self, np.array([x,y]), field.copy(), others_position):
            features[4] = 1
    
    if field[x-1,y] != -1 and field[x-1,y] != 1 :
        features[5] = danger(self, np.array([x-1,y]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)
    
    if field[x+1,y] != -1 and field[x+1,y] != 1 :
        features[6] = danger(self, np.array([x+1,y]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)

    if field[x,y-1] != -1 and field[x,y-1] != 1 :
        features[7] = danger(self, np.array([x,y-1]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)

    if field[x,y+1] != -1 and field[x,y+1] != 1 :
        features[8] = danger(self, np.array([x,y+1]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)

    features[9] = danger(self, np.array([x,y]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)

    #print(features[5:10])

    #the next feature is gonna be if there is an explosion anywhere around us and how long its gonna stay
    features[10:14] = np.array([ int(explosion_map[x-1,y]!= 0), int(explosion_map[x+1,y]!= 0), int(explosion_map[x,y-1]!=0), int(explosion_map[x,y+1] != 0)  ]) 

    #valid moves
    features[14:18] = np.array([ int(field[x-1,y] == 0), int(field[x+1,y] == 0), int(field[x,y-1] == 0), int(field[x,y+1] == 0) ])

    #the next feature is gonna show if a bomb action is possible
    features[18] = int(bomb)

    coin_weight = coin_distance * 1.5
    crate_weight = crate_distance * 4
    opponent_weight = opponent_distance 
    
    if opponent_weight <= coin_weight and opponent_weight <= crate_weight:
        features[0:4] = opponent_directions
    elif coin_weight <= crate_weight:
        features[0:4] = coin_directions
    else:
        features[0:4] = crate_directions

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
        return np.array([0,0,0,0]),0
    
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

        return direction, path_length -1
    else:
        return np.array([0,0,0,0]), np.inf

def check_for_crates(self, agent_position: np.array , field: np.array) -> bool:
    x, y = agent_position
    check_left, check_right, check_up, check_down =  np.array([field[x-1,y] != -1 , field[x+1,y] != -1, field[x,y-1] != -1, field[x,y+1] != -1])

    if check_right:
        for i in range(1,4):
            if x+i >= s.COLS:
                break 

            if field[x+i,y] == 1:
                return True

    if check_left:
        for i in range(1,4):
            if x-i < 0:
                break

            if field[x-i,y] == 1:
                return True 

    if check_down:
        for i in range(1,4):
            if y+i >= s.ROWS:
                break

            if field[x,y+i] == 1:
                return True 

    if check_up:
        for i in range(1,4):
            if y-i < 0:
                break 
            
            if field[x,y-i] == 1:
                return True 

    return False

def check_for_opponents(self, agent_position: np.array , field: np.array, opponents) -> bool:
    x, y = agent_position
    check_left, check_right, check_up, check_down =  np.array([field[x-1,y] != -1 , field[x+1,y] != -1, field[x,y-1] != -1, field[x,y+1] != -1])

    for opponent in opponents:
        field[opponent[0], opponent[1]] = 2

    if check_right:
        for i in range(1,4):
            if x+i >= s.COLS:
                break 

            if field[x+i,y] == 2:
                return True

    if check_left:
        for i in range(1,4):
            if x-i < 0:
                break

            if field[x-i,y] == 2:
                return True 

    if check_down:
        for i in range(1,4):
            if y+i >= s.ROWS:
                break

            if field[x,y+i] == 2:
                return True 

    if check_up:
        for i in range(1,4):
            if y-i < 0:
                break 
            
            if field[x,y-i] == 2:
                return True 

    return False


def check_escape_route(self, field: np.array, starting_point: np.array, explosion_indices: np.array, bombs: list, opponents: np.array) -> Tuple[bool, np.array]:
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
    
    #update the field so that opponents are included
    for opponent in opponents:
        field[opponent[0],opponent[1]] = -1

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


def check_near_bombs(self, agent_position: np.array , field: np.array, bombs, steps_passed):
    x, y = agent_position
    #print(f"x: {x} y: {y}")
    #print("field")
    #print(field.T)

    for bomb in bombs:
        field[bomb[0][0], bomb[0][1]] = 10 + bomb[1] - steps_passed

    check_left, check_right, check_up, check_down =  np.array([field[x-1,y] != -1 , field[x+1,y] != -1, field[x,y-1] != -1, field[x,y+1] != -1])

    #print(f"check left {check_left} check right {check_right} check up {check_up} check down {check_down} ")
    bomb_found = False
    min_cooldown = 100

    if field[x,y] >= 10:
        bomb_found = True
        min_cooldown = min(min_cooldown, field[x,y] - 10)

    if check_right:
        for i in range(1,4):
            if x+i >= s.COLS:
                break 

            if field[x+i,y] >= 10:
                #print("1")
                bomb_found = True
                min_cooldown = min(min_cooldown, field[x+i,y] - 10)
                

    if check_left and min_cooldown != 0:
        for i in range(1,4):
            if x-i < 0:
                break

            if field[x-i,y] >= 10:
                #print("2")
                bomb_found = True
                min_cooldown = min(min_cooldown, field[x-i,y] - 10)

    if check_down and min_cooldown != 0:
        for i in range(1,4):
            if y+i >= s.ROWS:
                break

            if field[x,y+i] >= 10:
                #print("3")
                bomb_found = True
                min_cooldown = min(min_cooldown, field[x,y+i] - 10)

    if check_up and min_cooldown != 0:
        for i in range(1,4):
            if y-i < 0:
                break 
            
            if field[x,y-i] >= 10:
                bomb_found = True
                min_cooldown = min(min_cooldown, field[x,y-i] - 10) 

    return bomb_found, min_cooldown


def danger(self, starting_point, bombs, field, explosion_map, opponents):
    #update the field so that the bombs are included
    updated_field = field.copy()

    #update the field so that opponents are included
    for opponent in opponents:
        updated_field[opponent[0],opponent[1]] = -1
        updated_field[opponent[0]-1, opponent[1]] = -1
        updated_field[opponent[0]+1, opponent[1]] = -1
        updated_field[opponent[0], opponent[1]+1] = -1
        updated_field[opponent[0], opponent[1]-1] = -1
    
    #update the field so that bombs are included
    for bomb in bombs:
        updated_field[bomb[0][0],bomb[0][1]] = -1

    #print("Updated field: ")
    #print(updated_field.T)

    parent = np.ones(s.COLS*s.ROWS) * -1
    start = starting_point[1] * s.COLS + starting_point[0]
    parent[start] = start
    
    path_queue = np.array([start])
    counter = 0

    #distance from start position to current position
    distance = np.zeros(s.COLS*s.ROWS)

    while counter < len(path_queue):
        current_position = path_queue[counter]
        dist = distance[current_position]
        #print(f"Current position: {current_position}")

        #print(f"Distance: {dist}")

        #update field to include explosions since explosions might have passed and new explosions might have spawned
        #new explosions
        for bomb in bombs:
            #print(f"bomb timer: {bomb[1]}")
            if bomb[1] - distance[current_position] == -1:
                #print("update field 1")
                for i in range(-3,4):
                    if bomb[0][0]+i >= 0 and bomb[0][0]+i < s.COLS:
                        updated_field[bomb[0][0]+i,bomb[0][1]] = -1

                    if bomb[0][1]+i >= 0 and bomb[0][1]+i < s.ROWS:
                        updated_field[bomb[0][0],bomb[0][1]+i] = -1

            if bomb[1] - distance[current_position] < -1:
                #print("update field 2")
                for i in range(-3,4):
                    if bomb[0][0]+i >= 0 and bomb[0][0]+i < s.COLS and field[bomb[0][0]+i,bomb[0][1]] != -1:
                        updated_field[bomb[0][0]+i,bomb[0][1]] = 0

                    if bomb[0][1]+i >= 0 and bomb[0][1]+i < s.ROWS and field[bomb[0][0],bomb[0][1]+i] != -1:
                        updated_field[bomb[0][0],bomb[0][1]+i] = 0

        #get the 2D coordinates
        x = current_position % s.COLS
        y = current_position // s.ROWS

        #print(f"x: {x} y: {y}")

        bombs_found, min_cooldown = check_near_bombs(self, [x,y], updated_field.copy(), bombs, distance[current_position])

        #this move is 100% save
        if not bombs_found:
            #print("no bombs found")
            return False
        
        #print("bomb found!")

        #this path is 100% not save, try a new path
        if min_cooldown == 0:
            #print("cooldown is zero")
            counter = counter + 1
            continue
    
        #left from the current position. 
        if current_position % s.COLS != 0 and updated_field[x-1,y] != -1 and updated_field[x-1,y] != 1 and updated_field[x-1,y]!= -1 and parent[current_position-1] == -1 and explosion_map[x-1,y] - distance[current_position] <= 0:
            #print("go left")
            path_queue = np.append(path_queue, current_position-1)
            parent[current_position-1] = current_position
            distance[current_position-1] = distance[current_position] +1
        

        #right from the current position
        if current_position % s.COLS != s.COLS-1 and updated_field[x+1,y] != -1 and updated_field[x+1,y]!=1 and updated_field[x+1,y]!= -1 and parent[current_position+1] == -1 and explosion_map[x+1,y] - distance[current_position] <= 0:
            #print("go right")
            path_queue = np.append(path_queue, current_position+1)
            parent[current_position+1] = current_position
            distance[current_position+1] = distance[current_position] +1
        
        #up from the current position
        if current_position >= s.COLS and updated_field[x,y-1] != -1 and updated_field[x, y-1]!= 1 and updated_field[x,y-1]!= -1 and parent[current_position-s.COLS] == -1 and explosion_map[x,y-1] - distance[current_position] <= 0:
            #print("go up")
            path_queue = np.append(path_queue,current_position-s.COLS)
            parent[current_position-s.COLS] = current_position
            distance[current_position-s.COLS] = distance[current_position] +1
        
 
        #down from the current position
        if y < s.ROWS-1 and updated_field[x,y+1] != -1 and updated_field[x, y+1]!= 1 and updated_field[x,y+1] != -1 and parent[current_position+s.COLS] == -1 and explosion_map[x,y+1] - distance[current_position] <= 0:
            #print("go down")
            path_queue = np.append(path_queue,current_position+s.COLS)
            parent[current_position+s.COLS] = current_position
            distance[current_position+s.COLS] = distance[current_position] +1

        counter = counter + 1

        #print("---------")

    #print("DANGER")
    return True
       
      