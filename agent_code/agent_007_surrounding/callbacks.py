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
    self.D = 126

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #self.regression_forests = [RandomForestRegressor(n_estimators=5) for i in range(len(ACTIONS))]
        with open("my-saved-model.pt", "rb") as file:
            self.regression_forests = pickle.load(file)
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
    random_prob = .06
    
    if self.train and random.random() < random_prob:  #or game_state['round'] < 100):
        self.logger.debug("Choosing action purely at random.")
        #80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    
    #approximate Q by regression forest
    Q =  [self.regression_forests[action_idx_to_test].predict([state_to_features(self, game_state)]) for action_idx_to_test in range(len(ACTIONS))]
    #print("Q-values:")
    #print(Q)

    #our policy is the argmax of the approximated Q-function
    action_idx = np.argmax(Q)
    #self.logger.debug("Querying model for action.")
    
    #print(f"Action: {ACTIONS[action_idx]}")
    
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
    _, _, own_bomb, (x, y) = game_state['self']
    others = game_state['others']
    others_position = np.zeros( (len(others), 2), dtype=int)

    for i,opponent in enumerate(others):
        others_position[i] = opponent[3]

    explosion_indices = np.array(np.where(explosion_map > 0)).T

    #Build a map that contains the distance to the nearest coin for each field
    coin_distance_map = build_coin_distance_map(self, field, coins)
    coin_distance_curr_pos = coin_distance_map[x][y]
    #directions to move to the closest coin
    possible_directions_towards_coin = np.array([int(coin_distance_map[x-1][y] < coin_distance_curr_pos), int(coin_distance_map[x+1][y] < coin_distance_curr_pos), int(coin_distance_map[x][y-1] < coin_distance_curr_pos), int(coin_distance_map[x][y+1] < coin_distance_curr_pos)])
    coin_distance = min(coin_distance_map[x-1][y], coin_distance_map[x+1][y],coin_distance_map[x][y-1], coin_distance_map[x][y+1])
    
    #update field with bombs and opponents to check valid moves
    updated_field = field.copy()
    for bomb in bombs:
        updated_field[bomb[0][0]][bomb[0][1]] = -1
    
    for opponent_pos in others_position:
        updated_field[opponent_pos[0]][opponent_pos[1]] = -1

    #valid directions that the agent can walk to right now
    valid_moves = np.array([ int(updated_field[x-1,y] == 0), int(updated_field[x+1,y] == 0), int(updated_field[x,y-1] == 0), int(updated_field[x,y+1] == 0) ])
    
    #directions that lead to a dead end 
    dead_end_directions = check_dead_end_directions(self, (x,y), field.copy(), bombs, others_position)

    #radius of bomb explosions, including the field of the bomb
    radius = s.BOMB_POWER + 1
    #diameter is used for features that check the local area around the agent
    #these features will contain a diameter*diameter map around the agent
    #(e.g bombs around agent will be a diameter*diameter map around the agent containing all bomb timers)
    diameter = 7

    bomb_timers_around_agent = np.ones((diameter, diameter)) * 100
    opponents_around_agent = np.zeros((diameter,diameter))

    #These for loops are used to build all the maps used for the features that 
    #contain the diameter*diameter fields around the agent
    for i in range(diameter):
        for j in range(diameter):
            current_x = x-3+i
            current_y = y-3+j
            if current_x >=0 and current_x < s.COLS and current_y >=0 and current_y < s.ROWS:
                
                #bombs around agent
                for bomb in bombs:
                    bomb_detonation_point_x = bomb[0][0]
                    bomb_detonation_point_y = bomb[0][1]
                    if np.all(bomb[0] == (current_x, current_y)):
                        bomb_timers_around_agent[i][j] = bomb[1]

                    for k in range(1,radius):
                        if field[bomb_detonation_point_x+k][bomb_detonation_point_y] == -1:
                                break
                        
                        if np.all((bomb_detonation_point_x + k, bomb_detonation_point_y) == (current_x, current_y)):
                            bomb_timers_around_agent[i][j] = bomb[1]

                    for k in range(1,radius):
                        if field[bomb_detonation_point_x-k][bomb_detonation_point_y] == -1:
                                break
                        
                        if np.all((bomb_detonation_point_x - k, bomb_detonation_point_y) == (current_x, current_y)):
                            bomb_timers_around_agent[i][j] = bomb[1]

                    for k in range(1,radius):
                        if field[bomb_detonation_point_x][bomb_detonation_point_y+k] == -1:
                                break
                        
                        if np.all((bomb_detonation_point_x, bomb_detonation_point_y+k) == (current_x, current_y)):
                            bomb_timers_around_agent[i][j] = bomb[1]

                    for k in range(1,radius):
                        if field[bomb_detonation_point_x][bomb_detonation_point_y-k] == -1:
                                break
                        
                        if np.all((bomb_detonation_point_x, bomb_detonation_point_y-k) == (current_x, current_y)):
                            bomb_timers_around_agent[i][j] = bomb[1]


                #opponents around player
                if [current_x, current_y] in others_position.tolist():
                    opponents_around_agent[i][j] = 1
                 
    #the first feature is which different directions will lead to the closest coins
    features[0:4] = possible_directions_towards_coin

    #the next features are gonna be the valid directions we can move to right now
    features[4:8] = valid_moves
    
    #the next feature are gonna be the directions that lead to dead ends right now
    features[8:12] = dead_end_directions
    
    #the next feature are gonna be explosions that are located in fields directly around the agent
    features[12:16] = np.array([ int(explosion_map[x-1,y]!= 0), int(explosion_map[x+1,y]!= 0), int(explosion_map[x,y-1]!=0), int(explosion_map[x,y+1] != 0)  ])
    
    #the next feature is gonna be the diameter*diameter map of opponents around the agent
    features[16:65] = opponents_around_agent.flatten()
    
    #the next feature is gonna be the diameter*diameter map of bomb timers around the agent
    features[65:114] = bomb_timers_around_agent.flatten()
    
    #the next feature is gonna be if there is a bomb action currently possible
    features[114] = int(own_bomb)

    #the next feature is gonna be the direction we need to move to get to the next crate the fastest
    crate_indices = np.array(np.where(field == 1)).T

    if crate_indices.size != 0:
       features[115:119], crate_distance = breadth_first_search(field.copy(), np.array([x,y]), crate_indices, bombs, explosion_indices, others_position)
    else:
       features[115:119] = np.array([0,0,0,0])
       crate_distance = 100

    #the next feature is gonna be the direction we need to move to get to the next oponnent the fastest
    if len(others_position) != 0:
        features[119:123], opponent_distance = breadth_first_search(field.copy(), np.array([x,y]), others_position, bombs, explosion_indices, others_position)
    else:
        features[119:123] = np.array([0,0,0,0])
        opponent_distance = 100

    #the last three features are gonna indicate how far away the closest coin, crate and opponent are
    features[123] = coin_distance
    features[124] = crate_distance
    features[125] = opponent_distance

    return features


def check_dead_end_directions(self, starting_point, field, bombs, opponents):
    """Checks, which directions lead to dead ends

    Args:
        starting_point: The point to look for dead_ends from
        field: The current field including walls etc.
        bombs: List of positions where bombs are located, these will block the agents path
        opponents: List of positions where opponents are located, these will block the agents path

    Returns:
        Array indicating, which directions will lead to a dead end and which will not
    """

    #update field with bombs, these will be treated like walls
    for bomb in bombs:
        field[bomb[0][0],bomb[0][1]] = -1
    
    #update field with opponents, these will be treated like walls
    for opponent in opponents:
        field[opponent[0]][opponent[1]] = -1

    x, y = starting_point
    dead_end_directions = [0,0,0,0]

    #check directions for dead end
    #if we can move less than 4 fields straight and there is no possibility to change
    #the axis the agent is walking on while walking on this straigth, the direction is considered a dead end
    #check for dead end left of the agent position
    for i in range(1,5):
        if field[x-i][y] != 0:
            dead_end_directions[0] = 1
            break
        if field[x-i][y-1] == 0 or field[x-i][y+1] == 0:
            break
    
    #check for dead end right of the agent position
    for i in range(1,5):
        if field[x+i][y] != 0:
            dead_end_directions[1] = 1
            break
        if field[x+i][y-1] == 0 or field[x+i][y+1] == 0:
            break
    
    #check for dead end above the agent
    for i in range(1,5):
        if field[x][y-i] != 0:
            dead_end_directions[2] = 1
            break
        if field[x-1][y-i] == 0 or field[x+1][y-i] == 0:
            break
    
    #check for dead end below the agent
    for i in range(1,5):
        if field[x][y+i] != 0:
            dead_end_directions[3] = 1
            break
        if field[x-1][y+i] == 0 or field[x+1][y+i] == 0:
            break

    return np.array(dead_end_directions)
            
    

def build_coin_distance_map(self, field, coin_positions):
    """Given a field and positions where coins are located this function will
    build a map that contains the distance to the closest coin for every position on the field

    Args:
        field: The current field including walls etc.
        coin_positions: List of positions where coins are currently located

    Returns:
        A map that contains the distance to the closest coin for every position on the field
    """
    coin_distance_map = np.empty((s.COLS, s.ROWS))
    #distance is defaulted to 100 meaning if there are no coins on the field
    #all distances will be 100
    coin_distance_map.fill(100)

    #for every coin update distance map
    for starting_point in coin_positions:
        #array for visited points
        visited = np.zeros((s.COLS, s.ROWS))

        #Queue for visiting the tiles that contains their position and their distance to the coin
        path_queue = deque(maxlen=s.COLS*s.ROWS)
        path_queue.append((starting_point, 0))

        visited[starting_point[0]][starting_point[1]] = 1

        #Visit tiles until a tile contains a coin
        while path_queue:
            current_position, distance = path_queue.popleft()
            if distance > 7:
                break
            x = current_position[0]
            y = current_position[1]

            #update coin distance map
            coin_distance_map[x][y] = min(coin_distance_map[x][y], distance)
            
            #now visit all neighbours if there is no wall and we did not visit them yet
            #since there is a wall around the whole field we do not need to check if we go
            #out of range of the collums and rows
            #left from the current position. 
            next_pos_x = x-1
            if field[next_pos_x][y]!= -1 and visited[next_pos_x][y] == 0:
                path_queue.append(([next_pos_x, y], distance+1))
                visited[next_pos_x][y] = 1

            #right from the current position
            next_pos_x = x+1
            if field[next_pos_x][y] != -1 and visited[next_pos_x][y] == 0:
                path_queue.append(([next_pos_x,y], distance+1))
                visited[next_pos_x][y] = 1

            #up from the current position
            next_pos_y = y-1
            if field[x][next_pos_y] != -1 and visited[x][next_pos_y] == 0:
                path_queue.append(([x, next_pos_y], distance+1))
                visited[x][next_pos_y] = 1

            #down from the current position
            next_pos_y = y+1
            if field[x][next_pos_y] != -1 and visited[x][next_pos_y] == 0:
                path_queue.append(([x, next_pos_y], distance+1))
                visited[x][next_pos_y] = 1
    return coin_distance_map

def breadth_first_search(field, starting_point, targets, bombs , explosion_indices, opponents): 
    """performs a breadth first search  on the field from a given starting point to find the fastest 
    way to a target point (when multiple target points are given it will find the fastest way to the closest)

    Args:
        field : The field to perform the breadth first search on including walls etc.
        starting_point: The position to start the breadth first search from (x,y)
        targets: List of targets to find the fastest way too [(x,y),(x,y),(x,y)]
        bombs: List of positions where bombs are located (need them to update the field, because they block path)
        explosion_indices: List of positions where explosions are located (need them to update the field, because they block path)
        opponents: List of positions where opponents are located (need them to update the field, because they block path)

    Returns:
         The next move that should be performed to walk towards the closest target as well as the distance to that target
    """
   

    #update the field so that the bombs are included, treated like walls 
    for bomb in bombs:
        field[bomb[0][0],bomb[0][1]] = -1
    
    #update the field so that opponents are included, treated like walls
    for opponent in opponents:
        field[opponent[0],opponent[1]] = -1

    #update the field so that the explosions are included, treated like walls
    for explosion_index in explosion_indices:
        field[explosion_index[0], explosion_index[1]] = -1

    #crates are not treated like walls because you can destroy them to get to your target

    #update the field so that the targets are included, marked by a 2 on the field
    for target in targets:
        field[target[0],target[1]] = 2
    
    #Use this list to get backtrace the path from the nearest target to the players position to get the direction
    parent = np.ones(s.COLS*s.ROWS) * -1

    #Queue for visiting the tiles. 
    start = starting_point[1] * s.COLS + starting_point[0]
    parent[start] = start

    #check if the player is already on a target field
    if field[starting_point[0],starting_point[1]] == 2:
        return np.array([0,0,0,0]),0
    
    path_queue = np.array([start])
    counter = 0
    
    target_reached = False
    target = None

    #Visit tiles until a tile contains a target
    while not target_reached and counter < len(path_queue):
        current_position = path_queue[counter]
        
        #get the 2D coordinates
        x = current_position % s.COLS
        y = current_position // s.COLS

        #check if we reached a target
        if field[x, y] == 2:
            target_reached = True
            target = current_position
        
        else:
            #append neighbours if path in the directions is not blocked and if we have not visited them yet
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
    
    #if we found a viable path (not blocked by walls etc.)
    #we traverse this path to get the direction we need to move towards next
    if target is not None:
        #get the path from the nearest target to the player by accessing the parent list
        path = [target]
        tile = target
        
        while tile != start:
            tile = int(parent[tile])
            path.append(tile)

        path = np.flip(path)

        path_length = len(path)

        #get the second tile of the path which indicates the direction to the nearest target
        next_position_x = path[1] % s.COLS
        next_position_y = path[1] // s.COLS

        direction = [int(next_position_x < starting_point[0]), int(next_position_x > starting_point[0]), int(next_position_y < starting_point[1]), int(next_position_y > starting_point[1])] 

        return direction, path_length -1
    #if there was no path found we return  distance 100 and no possible direction
    else:
        return np.array([0,0,0,0]), 100