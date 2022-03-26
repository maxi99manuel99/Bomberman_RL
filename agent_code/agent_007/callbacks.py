import os
import pickle
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
    #Exploration vs exploitation
    random_prob = .1

    if self.train and (random.random() < random_prob or game_state['round'] < 100):
        self.logger.debug("Choosing action purely at random.")
        #80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    
    #approximate Q by regression forest
    Q =  [self.regression_forests[action_idx_to_test].predict([state_to_features(self, game_state)]) for action_idx_to_test in range(len(ACTIONS))]

    #our policy is the argmax of the approximated Q-function
    action_idx = np.argmax(Q)
    self.logger.debug("Querying model for action.")

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

    coin_distance = np.inf
    crate_distance = np.inf
    opponent_distance = np.inf

    #direction and distance, to the next coin
    if len(coins) != 0:
        coin_directions, coin_distance = breadth_first_search(field.copy(), np.array([x,y]), coins, bombs , explosion_indices, others_position)
    else:
       coin_directions = np.array([0,0,0,0])

    #direction and distance to the next crate
    crate_indices = np.array(np.where(field == 1)).T

    if crate_indices.size != 0:
       crate_directions, crate_distance = breadth_first_search(field.copy(), np.array([x,y]), crate_indices, bombs, explosion_indices, others_position)
    else:
        crate_directions = np.array([0,0,0,0])

    #direction and distance to the next opponent
    if len(others_position) != 0:
        opponent_directions, opponent_distance = breadth_first_search(field.copy(), np.array([x,y]), others_position, bombs, explosion_indices, others_position)
    else:
        opponent_directions = np.array([0,0,0,0])

    #the next feature is gonna be if there is a crate or enemy in the near and we can find an escape route, so that it is sensible to drop a bomb
    #if danger returns 1 there will be no escape route when planting a bomb right now
    if check_for_crates(self, np.array([x,y]), field.copy()) or check_for_opponents(self, np.array([x,y]), field.copy(), others_position):
            updated_bombs = bombs.copy()
            updated_bombs.append(((x,y), 4))
            if not danger(self, np.array([x,y]), updated_bombs.copy(), field.copy(), explosion_map.copy(), others_position.copy()):
                features[4] = 1

    #the next feature is gonna be which moves are very dangerous (deadly)
    if field[x-1,y] != -1 and field[x-1,y] != 1 :
        features[5] = danger(self, np.array([x-1,y]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)
        
    if field[x+1,y] != -1 and field[x+1,y] != 1 :
        features[6] = danger(self, np.array([x+1,y]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)
        
    if field[x,y-1] != -1 and field[x,y-1] != 1 :
        features[7] = danger(self, np.array([x,y-1]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)

    if field[x,y+1] != -1 and field[x,y+1] != 1 :
        features[8] = danger(self, np.array([x,y+1]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)

    features[9] = danger(self, np.array([x,y]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)

    #the next feature is gonna be if there is an explosion anywhere around us and how long its gonna stay
    features[10:14] = np.array([ int(explosion_map[x-1,y]!= 0), int(explosion_map[x+1,y]!= 0), int(explosion_map[x,y-1]!=0), int(explosion_map[x,y+1] != 0)  ]) 

    #update field with bombs and opponents to check valid moves
    updated_field = field.copy()
    for bomb in bombs:
        updated_field[bomb[0][0]][bomb[0][1]] = -1
    
    for opponent_pos in others_position:
        updated_field[opponent_pos[0]][opponent_pos[1]] = -1

    #the next feature are gonna be which moves are valid right now
    features[14:18] = np.array([ int(updated_field[x-1,y] == 0), int(updated_field[x+1,y] == 0), int(updated_field[x,y-1] == 0), int(updated_field[x,y+1] == 0) ])
    #print("Valid Moves:", features[14:18])

    #the next feature is gonna show if a bomb action is possible
    features[18] = int(own_bomb)

    #weighting coin, crate and opponent-distance and combining them into one feature
    #giving the direction to the next target
    coin_weight = coin_distance
    crate_weight = crate_distance * 4
    opponent_weight = opponent_distance 
    
    if opponent_weight <= coin_weight and opponent_weight <= crate_weight:
        features[0:4] = opponent_directions
    elif coin_weight <= crate_weight:
        features[0:4] = coin_directions
    else:
        features[0:4] = crate_directions

    return features


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
    #if there was no path found we return infinite distance and no possible direction
    else:
        return np.array([0,0,0,0]), np.inf

def check_for_crates(self, agent_position, field):
    """Given a field and an agent_position, this function checks if any crates would be destroyed if 
    a bomb was to be planted at that agent_position

    Args:
        agent_position: The position of the agent, where we want to check if a bomb would hit any crates
        field: The current field including walls etc.

    Returns:
        Bool whether planting a bomb at the given position would destroy any crates or not
    """
    x, y = agent_position
    
    #field == 1 indicates a crate, -1 indicates a wall
    #we can break when we hit a wall, since this will block the further explosion
    #check if the bomb would hit a crate on the right side of the agent
    for i in range(1,4):
        if field[x+i][y] == -1:
            break 

        if field[x+i,y] == 1:
            return True

    #check if the bomb would hit a crate on the left side of the agent
    for i in range(1,4):
        if field[x-i][y] == -1:
            break

        if field[x-i,y] == 1:
            return True 

    #check if the bomb would hit a crate below the agent
    for i in range(1,4):
        if field[x][y+i] == -1:
            break

        if field[x,y+i] == 1:
            return True 

    #check if the bomb would hit a crate above the agent
    for i in range(1,4):
        if field[x][y-i] == -1:
            break 
        
        if field[x,y-i] == 1:
            return True 

    #no crates hit
    return False

def check_for_opponents(self, agent_position, field, opponents):
    """Given a field, an agent_position and positions where opponents are located this function checks if any opponents would be hit if 
    a bomb was to be planted at that agent_position

    Args:
        agent_position: The position of the agent, where we want to check if a bomb would hit any opponents
        field: The current field including walls etc.
        opponents: Positions where opponents are located

    Returns:
        Bool whether planting a bomb at the given position would hit any opponents or not
    """
    x, y = agent_position

    #update field to include the opponents that we want to check for, indicated by a 2
    for opponent in opponents:
        field[opponent[0], opponent[1]] = 2

    #field == 2 indicates a opponent, -1 indicates a wall
    #we can break when we hit a wall, since this will block the further explosion
    #check if the bomb would hit any opponent on the right side of the agent
    for i in range(1,4):
        if field[x+i][y] == -1:
            break 

        if field[x+i,y] == 2:
            return True

    #check if the bomb would hit any opponent on the left side of the agent
    for i in range(1,4):
        if field[x-i][y] == -1:
            break

        if field[x-i,y] == 2:
            return True 

    #check if the bomb would hit any opponent below the agent
    for i in range(1,4):
        if field[x][y+i] == -1:
            break

        if field[x,y+i] == 2:
            return True 

    #check if the bomb would hit any opponent above the agents
    for i in range(1,4):
        if field[x][y-i] == -1:
            break 
        
        if field[x,y-i] == 2:
            return True 

    return False

def check_near_bombs(self, agent_position, field, bombs, steps_passed):
    """Given a field, an agent_position and positions where bombs are located
    as well as their cooldown (and how many steps we look into the future, which will decrease their colldown)
    this function checks if the agent positions is within
    the explosion radius of one of the bombs and if so how long this bomb will take
    to explode

    This function is used as a helping function for the danger function

    Args:
        agent_position: The position of the agent, where we want to check if a bomb would hit any opponents
        field: The current field including walls etc.
        bombs: Positions where bombs are located
        steps_passed: The amount of steps we are looking into the future, which will decrease bomb_cooldown

    Returns:
        Bool indicating wheter the agent_position is in range of a possible bomb explosion
        and how long this bomb will take to explode
    """

    x, y = agent_position

    #update the field so that the bomb positions are included (indicated by values over 10)
    #the field will contain the cooldown of the bomb + 10
    for bomb in bombs:
        field[bomb[0][0], bomb[0][1]] = 10 + bomb[1] - steps_passed

    #we want to find the bomb, that would hit the agent with the smallest cooldown
    #since this is the bomb that put him in danger the soonest
    bomb_found = False
    min_cooldown = 100

    #check if a bomb is on the field of the agent
    if field[x,y] >= 10:
        bomb_found = True
        min_cooldown = min(min_cooldown, field[x,y] - 10)

    #check if a bomb is to the right of the agent, that will hit him if he stays at that position
    #when min_cooldown is 0 this means we have already found a bomb with the minimum possible cooldown
    #so it makes no sense to continue searching
    if min_cooldown != 0:
        for i in range(1,4):
            if field[x+i][y] == -1:
                break 

            if field[x+i,y] >= 10:
                bomb_found = True
                min_cooldown = min(min_cooldown, field[x+i,y] - 10)
                
    #check if a bomb is to the left of the agent, that will hit him if he stays at that position
    if min_cooldown != 0:
        for i in range(1,4):
            if field[x-i][y] == -1:
                break

            if field[x-i,y] >= 10:
                bomb_found = True
                min_cooldown = min(min_cooldown, field[x-i,y] - 10)

    #check if a bomb is below the agent, that will hit him if he stays at that position
    if min_cooldown != 0:
        for i in range(1,4):
            if field[x][y+i] == -1:
                break

            if field[x,y+i] >= 10:
                bomb_found = True
                min_cooldown = min(min_cooldown, field[x,y+i] - 10)

    #check if a bomb is above of the agent, that will hit him if he stays at that position
    if min_cooldown != 0:
        for i in range(1,4):
            if field[x][y-i] == -1:
                break 
            
            if field[x,y-i] >= 10:
                bomb_found = True
                min_cooldown = min(min_cooldown, field[x,y-i] - 10) 

    return bomb_found, min_cooldown


def danger(self, starting_point, bombs, field, explosion_map, opponents, starting_distance=0):
    """Given a field and a starting point as well as a list of bomb positions, opponents
    and an explosion map, this function checks if being on the starting point in the given situation is deadly
    or if there is a path to escape from possible bombs, explosions etc at the current time.

    Args:
        starting_point: The point to check if it is deadly in the current situation
        bombs: List of positions of bombs on the field
        field: The current field including walls etc.
        explosion_map: A map that includes all current ongoing explosions and how long they will stay
        opponents: List of positions of opponents
        starting_distance (optional): You can use this parameter if the agent is distant from a 
        position you want to check, to check if the position will be dangerous when the agent arrives.
        Defaults to 0. 

    Returns:
        Bool indicating if the position is deadly in the current situation or not
    """
    #update the field so that the bombs are included
    updated_field = field.copy()

    #update the field so that opponents are included, treated like walls
    for opponent in opponents:
        updated_field[opponent[0],opponent[1]] = -1
        updated_field[opponent[0]-1, opponent[1]] = -1
        updated_field[opponent[0]+1, opponent[1]] = -1
        updated_field[opponent[0], opponent[1]+1] = -1
        updated_field[opponent[0], opponent[1]-1] = -1
    
    #update the field so that bombs are included, treated like walls
    for bomb in bombs:
        updated_field[bomb[0][0],bomb[0][1]] = -1


    parent = np.ones(s.COLS*s.ROWS) * -1
    start = starting_point[1] * s.COLS + starting_point[0]
    parent[start] = start
    
    path_queue = np.array([start])
    counter = 0

    #distance from start position to current position
    distance = np.ones(s.COLS*s.ROWS) * starting_distance

    while counter < len(path_queue):
        current_position = path_queue[counter]
        dist = distance[current_position]

        #update field to include explosions since explosions might have passed and new explosions might have spawned
        for bomb in bombs:
            if bomb[1] - distance[current_position] == -1 or bomb[1] - distance[current_position] == 0:
                for i in range(-3,4):
                    if bomb[0][0]+i >= 0 and bomb[0][0]+i < s.COLS:
                        updated_field[bomb[0][0]+i,bomb[0][1]] = -1

                    if bomb[0][1]+i >= 0 and bomb[0][1]+i < s.ROWS:
                        updated_field[bomb[0][0],bomb[0][1]+i] = -1

            elif bomb[1] - distance[current_position] < -1:
                for i in range(-3,4):
                    if bomb[0][0]+i >= 0 and bomb[0][0]+i < s.COLS and field[bomb[0][0]+i,bomb[0][1]] != -1:
                        updated_field[bomb[0][0]+i,bomb[0][1]] = 0

                    if bomb[0][1]+i >= 0 and bomb[0][1]+i < s.ROWS and field[bomb[0][0],bomb[0][1]+i] != -1:
                        updated_field[bomb[0][0],bomb[0][1]+i] = 0

        #get the 2D coordinates
        x = current_position % s.COLS
        y = current_position // s.ROWS
        
        #check if there are any dangerous bombs around this position
        #and how long they will take to explode
        bombs_found, min_cooldown = check_near_bombs(self, [x,y], field.copy(), bombs, distance[current_position])

        #the given position is 100% save in the current situation
        if not bombs_found:
            return False
        
        #this path is 100% not save, try a new path
        if min_cooldown == 0:
            counter = counter + 1
            continue
    
        #left from the current position. 
        if current_position % s.COLS != 0 and updated_field[x-1,y] != -1 and updated_field[x-1,y] != 1 and updated_field[x-1,y]!= -1 and parent[current_position-1] == -1 and explosion_map[x-1,y] - distance[current_position] <= 0:
            path_queue = np.append(path_queue, current_position-1)
            parent[current_position-1] = current_position
            distance[current_position-1] = distance[current_position] +1

        #right from the current position
        if current_position % s.COLS != s.COLS-1 and updated_field[x+1,y] != -1 and updated_field[x+1,y]!=1 and updated_field[x+1,y]!= -1 and parent[current_position+1] == -1 and explosion_map[x+1,y] - distance[current_position] <= 0:
            path_queue = np.append(path_queue, current_position+1)
            parent[current_position+1] = current_position
            distance[current_position+1] = distance[current_position] +1
        
        #up from the current position
        if current_position >= s.COLS and updated_field[x,y-1] != -1 and updated_field[x, y-1]!= 1 and updated_field[x,y-1]!= -1 and parent[current_position-s.COLS] == -1 and explosion_map[x,y-1] - distance[current_position] <= 0:
            path_queue = np.append(path_queue,current_position-s.COLS)
            parent[current_position-s.COLS] = current_position
            distance[current_position-s.COLS] = distance[current_position] +1
        
        #down from the current position
        if y < s.ROWS-1 and updated_field[x,y+1] != -1 and updated_field[x, y+1]!= 1 and updated_field[x,y+1] != -1 and parent[current_position+s.COLS] == -1 and explosion_map[x,y+1] - distance[current_position] <= 0:
            path_queue = np.append(path_queue,current_position+s.COLS)
            parent[current_position+s.COLS] = current_position
            distance[current_position+s.COLS] = distance[current_position] +1

        counter = counter + 1

    #no safe path found
    return True
       
      