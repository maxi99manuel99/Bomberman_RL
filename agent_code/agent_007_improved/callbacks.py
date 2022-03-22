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
    self.D = 348

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
    random_prob = .06
    
    if self.train and (random.random() < random_prob or game_state['round'] < 100):
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


    coin_distance_map = build_coin_distance_map(self, field, coins)
    coin_distance_curr_pos = coin_distance_map[x][y]
    #directions to move to the closest coin
    possible_directions_towards_coin = np.array([int(coin_distance_map[x-1][y] < coin_distance_curr_pos), int(coin_distance_map[x+1][y] < coin_distance_curr_pos), int(coin_distance_map[x][y-1] < coin_distance_curr_pos), int(coin_distance_map[x][y+1] < coin_distance_curr_pos)])
    coin_distance = min(coin_distance_map[x-1][y], coin_distance_map[x+1][y],coin_distance_map[x][y-1], coin_distance_map[x][y+1])
    #valid moves
    valid_moves = np.array([ int(field[x-1,y] == 0), int(field[x+1,y] == 0), int(field[x,y-1] == 0), int(field[x,y+1] == 0) ])
    #directions that lead to a dead end
    dead_end_directions = check_dead_end_directions(self, (x,y), field.copy(), bombs)

    radius = 4
    diameter = 9

    crates_around_agent = np.zeros((diameter,diameter))
    bomb_timers_around_agent = np.ones((diameter, diameter)) * 100
    explosions_duration_around_agent = np.zeros((diameter,diameter))
    opponents_around_agent = np.zeros((diameter,diameter))

    for i in range(diameter):
        for j in range(diameter):
            current_x = x-radius+i
            current_y = y-radius+j
            if current_x >=0 and current_x < s.COLS and current_y >=0 and current_y < s.ROWS:

                #crate around player
                #(number of crates that are in explosion range of the field we are looking at)
                if field[current_x][current_y] != -1:
                    for k in range(1,radius):
                        if field[current_x+k][current_y] == -1:
                            break
                        elif field[current_x+k][current_y] == 1:
                            crates_around_agent[i][j] = crates_around_agent[i][j] + 1
                    for k in range(1,radius):
                        if field[current_x-k][current_y] == -1:
                            break
                        elif field[current_x-k][current_y] == 1:
                            crates_around_agent[i][j] = crates_around_agent[i][j] + 1
                    for k in range(1,radius):
                        if field[current_x][current_y+k] == -1:
                            break
                        elif field[current_x][current_y+k] == 1:
                            crates_around_agent[i][j] = crates_around_agent[i][j] + 1
                    for k in range(1,radius):
                        if field[current_x][current_y-k] == -1:
                            break
                        elif field[current_x][current_y-k] == 1:
                            crates_around_agent[i][j] = crates_around_agent[i][j] + 1
                
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

                
                #explostions around player
                explosions_duration_around_agent[i][j] = explosion_map[current_x][current_y]

                #opponents around player
                if [current_x, current_y] in others_position.tolist():
                    opponents_around_agent[i][j] = 1
                 

    features[0:4] = possible_directions_towards_coin
    #print(features[0:4])
    #print("possible coin direction")
    #print(features[0:4])

    features[4:85] = crates_around_agent.flatten()
    #print("crates around agent")
    #print(features[4:85].reshape((9,9)).T)
    
    features[85:89] = valid_moves
    #print("valid moves")
    #print(features[85:89])
    
    features[89:93] = dead_end_directions
    #print("dead end directions")
    #print(features[89:93])
    
    features[93:174] = explosions_duration_around_agent.flatten()
    #print("explosion duration around agent ")
    #print(features[93:174].reshape((9,9)).T)
    
    features[174:255] = opponents_around_agent.flatten()
    #print("opp")
    #print(opponents_around_agent.reshape((9,9)).T)
    
    features[255:336] = bomb_timers_around_agent.flatten()
    #print("bomb")
    #print(bomb_timers_around_agent.reshape((9,9)).T)
  
    features[336] = int(own_bomb)

    #the next feature is gonna be the direction we need to move to get to the next crate the fastest
    crate_indices = np.array(np.where(field == 1)).T

    if crate_indices.size != 0:
       features[337:341], crate_distance = breath_first_search(field.copy(), np.array([x,y]), crate_indices, bombs, explosion_indices, others_position)
    else:
       features[337:341] = np.array([0,0,0,0])
       crate_distance = 100

    #the next feature is gonna be the direction we need to move to get to the next oponnent the fastest
    if len(others_position) != 0:
        features[341:345], opponent_distance = breath_first_search(field.copy(), np.array([x,y]), others_position, bombs, explosion_indices, others_position)
    else:
        features[341:345] = np.array([0,0,0,0])
        opponent_distance = 100

    features[345] = coin_distance
    features[346] = crate_distance
    features[347] = opponent_distance

    return features


def check_dead_end_directions(self, starting_point, field, bombs):
    #print("field")
    #print(field.T)
    for bomb in bombs:
        field[bomb[0][0],bomb[0][1]] = -1

    x, y = starting_point
    dead_end_directions = [0,0,0,0]

    for i in range(1,5):
        if field[x-i][y] != 0:
            dead_end_directions[0] = 1
            break
        if field[x-i][y-1] == 0 or field[x-i][y+1] == 0:
            break
    
    for i in range(1,5):
        if field[x+i][y] != 0:
            dead_end_directions[1] = 1
            break
        if field[x+i][y-1] == 0 or field[x+i][y+1] == 0:
            break
    
    for i in range(1,5):
        if field[x][y-i] != 0:
            dead_end_directions[2] = 1
            break
        if field[x-1][y-i] == 0 or field[x+1][y-i] == 0:
            break
    
    for i in range(1,5):
        if field[x][y+i] != 0:
            dead_end_directions[3] = 1
            break
        if field[x-1][y+i] == 0 or field[x+1][y+i] == 0:
            break

    return np.array(dead_end_directions)
            
    

def build_coin_distance_map(self, field: np.array, coin_positions: np.array) -> np.array:
    coin_distance_map = np.empty((s.COLS, s.ROWS))
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

        return direction, path_length -1
    else:
        return np.array([0,0,0,0]), 100