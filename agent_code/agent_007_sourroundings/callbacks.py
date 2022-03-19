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
    self.D = 493

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
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
    random_prob = .02
    
    if self.train and (random.random() < random_prob or game_state['round'] < 500):
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
    _, _, own_bomb, (x, y) = game_state['self']
    others = game_state['others']
    others_position = np.zeros( (len(others), 2), dtype=int)

    for i,opponent in enumerate(others):
        others_position[i] = opponent[3]

    explosion_indices = np.array(np.where(explosion_map > 0)).T

    coin_distances_around_agent = np.ones((9, 9)) * 100
    crates_around_agent = np.zeros((9,9))
    bomb_timers_around_agent = np.ones((9, 9)) * 100
    explosions_duration_around_agent = np.zeros((9,9))
    opponents_around_agent = np.zeros((9,9))
    field_around_agent = np.ones((9,9)) * -1

    coin_distance_map = build_coin_distance_map(self, field, coins)

    for i in range(9):
        for j in range(9):
            current_x = x-4+i
            current_y = y-4+j
            if current_x >=0 and current_x < s.COLS and current_y >=0 and current_y < s.ROWS:
                #coins around agent
                coin_distances_around_agent[i,j] = coin_distance_map[current_x][current_y]

                #crate around player
                #(number of crates that are in explosion range of the field we are looking at)
                if field[current_x][current_y] != -1:
                    for k in range(1,4):
                        if field[current_x+k][current_y] == -1:
                            break
                        elif field[current_x+k][current_y] == 1:
                            crates_around_agent[i][j] = crates_around_agent[i][j] + 1
                    for k in range(1,4):
                        if field[current_x-k][current_y] == -1:
                            break
                        elif field[current_x-k][current_y] == 1:
                            crates_around_agent[i][j] = crates_around_agent[i][j] + 1
                    for k in range(1,4):
                        if field[current_x][current_y+k] == -1:
                            break
                        elif field[current_x][current_y+k] == 1:
                            crates_around_agent[i][j] = crates_around_agent[i][j] + 1
                    for k in range(1,4):
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

                    for k in range(1,4):
                        if field[bomb_detonation_point_x+k][bomb_detonation_point_y] == -1:
                                break
                        
                        if np.all((bomb_detonation_point_x + k, bomb_detonation_point_y) == (current_x, current_y)):
                            bomb_timers_around_agent[i][j] = bomb[1]

                    for k in range(1,4):
                        if field[bomb_detonation_point_x-k][bomb_detonation_point_y] == -1:
                                break
                        
                        if np.all((bomb_detonation_point_x - k, bomb_detonation_point_y) == (current_x, current_y)):
                            bomb_timers_around_agent[i][j] = bomb[1]

                    for k in range(1,4):
                        if field[bomb_detonation_point_x][bomb_detonation_point_y+k] == -1:
                                break
                        
                        if np.all((bomb_detonation_point_x, bomb_detonation_point_y+k) == (current_x, current_y)):
                            bomb_timers_around_agent[i][j] = bomb[1]

                    for k in range(1,4):
                        if field[bomb_detonation_point_x][bomb_detonation_point_y-k] == -1:
                                break
                        
                        if np.all((bomb_detonation_point_x, bomb_detonation_point_y-k) == (current_x, current_y)):
                            bomb_timers_around_agent[i][j] = bomb[1]

                
                #explostions around player
                explosions_duration_around_agent[i][j] = explosion_map[current_x][current_y]

                #opponents around player
                if [current_x, current_y] in others_position.tolist():
                    opponents_around_agent[i][j] = 1
                 
                #the general field around the player so he finds something like dead ends
                field_around_agent[i][j] = field[current_x][current_y]


    features[0:81] = coin_distances_around_agent.flatten()
    #print("Coin Distances")
    #print(coin_distances_around_agent.T)
    features[81:162] = bomb_timers_around_agent.flatten()
    #print("Bomb Timers")
    #print(bomb_timers_around_agent.T)
    features[162:243] = explosions_duration_around_agent.flatten()
    #print("Explosion Duration")
    #(explosions_duration_around_agent.T)
    features[243:324] = opponents_around_agent.flatten()
    #print("Opponents Around")
    #print(opponents_around_agent.T)
    features[324:405] = crates_around_agent.flatten()
    #print("Crates Around")
    #print(crates_around_agent.T)
    #the next feature is if we can drop a bomb right now
    features[405] = int(own_bomb)
    #print("Bomb Possible")
    #print(features[245])
    #the last feature is gonna be the field around the player for stuff like valid moves and dead_ends
    features[406:487] = field_around_agent.flatten()
    #print("Valid Moves")
    #print(features[246:250])
    escape_possible, _ = check_escape_route(self, field, (x,y), explosion_indices, bombs, others_position)
    features[487] = int(escape_possible)
    
    features[488:492] = 0
    if field[x-1,y] != -1 and field[x-1,y] != 1 :
        features[488] = danger(self, np.array([x-1,y]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)
    
    if field[x+1,y] != -1 and field[x+1,y] != 1 :
        features[489] = danger(self, np.array([x+1,y]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)

    if field[x,y-1] != -1 and field[x,y-1] != 1 :
        features[490] = danger(self, np.array([x,y-1]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)

    if field[x,y+1] != -1 and field[x,y+1] != 1 :
        features[491] = danger(self, np.array([x,y+1]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)

    features[492] = danger(self, np.array([x,y]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)

    '''bomb_timers = bomb_timers_around_agent.copy()
    if bomb_timers[4][4] == 1:
        if bomb_timers[3][4] != 1:
            print("LEFT LIFE_SAVING_MOVE")
        if bomb_timers[5][4] != 1: 
            print("RIGHT LIFE_SAVING_MOVE")
        if bomb_timers[4][3] != 1:
            print("UP LIFE_SAVING_MOVE")
        if bomb_timers[4][5] != 1:
            print("DOWN LIFE_SAVING_MOVE")
        else:
            print("DEADLY_MOVE")

    crate_counts = crates_around_agent.copy()
    opponents = opponents_around_agent.copy()
    if not escape_possible:
        print("DEADLY_BOMB")
    elif crate_counts[4][4] == 1:
        print("OK_BOMB")
    elif crate_counts[4][4] == 2:
        print("GOOD_BOMB")
    elif crate_counts[4][4] >= 3:
        print("VERY_GOOD_BOMB")
    elif np.all(opponents == 0):
        print("SENSELESS_BOMB")'''

    return features

    

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

def danger(self, starting_point, bombs, field, explosion_map, opponents):
    #update the field so that the bombs are included
    updated_field = field.copy()

    #update the field so that opponents are included
    for opponent in opponents:
        updated_field[opponent[0],opponent[1]] = -1
    
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
