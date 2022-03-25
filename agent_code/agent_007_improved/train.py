from collections import namedtuple, deque
from hashlib import new
from pyexpat import features
from attr import field
import numpy as np

import pickle
from typing import List

import events as e
from .callbacks import state_to_features
import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TO_INT = {'UP':0, 'RIGHT' : 1 , 'DOWN': 2, 'LEFT': 3, 'WAIT':4, 'BOMB': 5}

FEATURES_TO_INT = {"DIRECTION_TO_COIN": [0,1,2,3],
                   "VALID_MOVES": [4,5,6,7],
                   "DEAD_END_DIRECTIONS": [8,9,10,11],
                   "EXPLOSIONS_AROUND_AGENT": [12,13,14,15],
                   "OPPONENTS_AROUND_AGENT": np.arange(16,65),
                   "BOMB_TIMERS_AROUND_AGENT": np.arange(65,114),
                   "BOMB_ACTIVE": 114,
                   "DIRECTION_TO_CRATE": [115,116,117,118],
                   "DIRECTION_TO_OPPONENT": [119,120,121,122],
                   "COIN_DISTANCE": 123,
                   "CRATE_DISTANCE": 124,
                   "OPPONENT_DISTANCE": 125}


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'episode_rewards', 'timestep', 'episode_next_states'))

EpisodeTransition = namedtuple('EpisodeTransition',
                        ('state', 'action', 'next_state', 'reward', 'timestep'))                    

# Hyper parameters -- DO modify
GAMMA = 0.1 # discount factor
N = 2 # number of timesteps to look into the future for n_step td

TRANSITION_HISTORY_SIZE = 50000  # keep only ... last transitions
BATCH_SIZE = 30000 # subset of the transitions used for gradient update
BATCH_PRIORITY_SIZE = 100 #sample the batch with the biggest squared loss

RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    #save total rewards for each game
    self.total_rewards = 0

    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.episode_transitions = deque(maxlen=s.MAX_STEPS)
    self.episode_reward_vector = deque(maxlen=s.MAX_STEPS)
    self.episode_next_states = deque(maxlen=s.MAX_STEPS)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    #In the first game state our agent can not achieve anything so we ingore it
    if old_game_state == None:
        return
        
    #Append custom events for rewards
    total_events = append_custom_events(self, old_game_state, new_game_state, events)

    #calculate total reward for this timestep
    step_reward = reward_from_events(self, total_events)
    self.total_rewards = self.total_rewards + step_reward

    # state_to_features is defined in callbacks.py
    # fill experience buffer with transitions
    self.episode_transitions.append(EpisodeTransition(state_to_features(self, old_game_state),self_action, state_to_features(self, new_game_state), step_reward, old_game_state['step']))
    self.episode_reward_vector.append(step_reward)
    self.episode_next_states.append(state_to_features(self, new_game_state))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    #calculate total reward for this timestep
    step_reward = reward_from_events(self, events)
    self.total_rewards = self.total_rewards + step_reward

    #append last transition (does not work for temporal difference since we needa next step)
    self.episode_transitions.append(EpisodeTransition(state_to_features(self,last_game_state), last_action, None, step_reward, last_game_state['step']))
    self.episode_reward_vector.append(step_reward)
    self.episode_next_states.append(None)

    #append episode transitions to total transitions
    for i, episode_t in enumerate(self.episode_transitions):
        self.transitions.append(Transition(episode_t[0], episode_t[1], episode_t[2],self.episode_reward_vector.copy() ,episode_t[4], self.episode_next_states.copy()))

    #reset total rewards per game back to 0
    #clear episode_rewards and episode_transitions
    print("total rewards: ",self.total_rewards)

    self.total_rewards = 0
    self.episode_reward_vector.clear()
    self.episode_next_states.clear()
    self.episode_transitions.clear()

    if last_game_state['round'] < 800:
        return
    """
    SAMPLE A BATCH FROM THE EXPERIENCE BUFFER AND USE IT TO IMPROVIZE THE WEIGHT VECTORS
    """

    #only start fitting after we have tried enough random actions
    #if last_game_state['round'] < 99:
     #   return

    #random subset of experience buffer
    indices = np.random.choice(np.arange(len(self.transitions), dtype=int), min(len(self.transitions), BATCH_SIZE), replace=False)
    batch = np.array(self.transitions, dtype=Transition)[indices]
   
    #create subbatch for every action and improve weight vector by gradient update
    for action in ACTIONS:
        subbatch_indices = np.where(batch[:,1] == action)[0]
        subbatch = batch[subbatch_indices]
        
        subbatch_old_states = np.zeros((len(subbatch_indices), self.D))
        for i in range(len(subbatch_indices)):
            subbatch_old_states[i] = batch[subbatch_indices[i]][0]

        #If an action is not present in our current batch we can not update it. Also, we send the number of rounds to the function to decrease 
        #the learning rate.
        if len(subbatch) != 0:
            #calculate the responses
            #if last_game_state['round'] == 99 :
             #   response = np.array([monte_carlo(self, transition[3], transition[4]) for transition in subbatch])
            if False:
                pass
            else:   
                #continue with monte carlo
                response = np.array([monte_carlo(self, transition[3], transition[4]) for transition in subbatch])

                #temporal difference 
                #response = np.array([temporal_difference(self, transition[3][transition[4]-1], transition[2]) for transition in subbatch])

                #sarsa 
                #response = np.array([sarsa(self,transition[3][transition[4]-1], transition[2], transition[1]) for transition in subbatch], dtype = object)

            #train the forest
            self.regression_forests[ACTION_TO_INT[action]].fit(subbatch_old_states, response)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.regression_forests, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """


    game_rewards = {
        e.INVALID_ACTION: -20,
        e.GOT_KILLED: -200,
        e.COIN_COLLECTED: 17,
        e.OPPONENT_ELIMINATED: 80,
        'LIFE_SAVING_MOVE': 30,
        'GOOD_BOMB_PLACEMENT': 17,
        'BAD_BOMB_PLACEMENT': -100,
        'DEADLY_MOVE': -100,
        'MOVES_TOWARD_TARGET': 3,
        'WAITING_ONLY_OPTION': 20,
        'BAD_MOVE': -3,
    }
    #print("All rewards for this action")
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            #print(f"Event: {event}")
            #print(f"Reward: {game_rewards[event]}")
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    #print("-----------------")

    return reward_sum

def append_custom_events(self,old_game_state: dict, new_game_state: dict, events: List[str]) -> List[str]:

    """
    Appends all our custom events to the events list
    so we can calculate the total rewards out of these

    Args:
        events (List[str]): The non custom events, that happened between two game states
        old_game_state: dict: The game state before the events happened
        new_game_state: dict: The game state after the events happened

    Returns:
        List[str]: Full event list with custom events appended
    """

    if e.INVALID_ACTION in events:
        return events
    
    features = state_to_features(self,old_game_state)
    _, _, _, (x,y) = old_game_state['self']
    field = np.array(old_game_state['field'])
    coins = np.array(old_game_state['coins'])
    bombs = old_game_state['bombs']
    explosion_map = np.array(old_game_state['explosion_map'])
    others = old_game_state['others']
    others_position = np.zeros( (len(others), 2), dtype=int)
    explosion_indices = np.array(np.where(explosion_map > 0)).T
    explosion_around_agent = np.array([ int(explosion_map[x-1,y]!= 0), int(explosion_map[x+1,y]!= 0), int(explosion_map[x,y-1]!=0), int(explosion_map[x,y+1] != 0)  ])

    for i,opponent in enumerate(others):
        others_position[i] = opponent[3]

    danger_left, danger_right, danger_up, danger_down, danger_wait = [0,0,0,0,0]

    if field[x-1,y] != -1 and field[x-1,y] != 1 :
        danger_left = danger(self, np.array([x-1,y]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)
    
    if field[x+1,y] != -1 and field[x+1,y] != 1 :
        danger_right = danger(self, np.array([x+1,y]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)

    if field[x,y-1] != -1 and field[x,y-1] != 1 :
        danger_up = danger(self, np.array([x,y-1]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)

    if field[x,y+1] != -1 and field[x,y+1] != 1 :
        danger_down = danger(self, np.array([x,y+1]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)
    
    dangerous_movement = [danger_left, danger_right, danger_up, danger_down]
    danger_wait = danger(self, np.array([x,y]), bombs.copy(), field.copy(), explosion_map.copy(), others_position)

    #check, if waiting is dangerous we need to move 
    if danger_wait == 1: 
        #check if did a life saving move
        if danger_left == 0 and e.MOVED_LEFT in events:
            events.append("LIFE_SAVING_MOVE")
        elif danger_right == 0 and e.MOVED_RIGHT in events:
            events.append("LIFE_SAVING_MOVE")
        elif danger_up == 0 and e.MOVED_UP in events:
            events.append("LIFE_SAVING_MOVE")
        elif danger_down == 0 and e.MOVED_DOWN in events:
            events.append("LIFE_SAVING_MOVE")
        else: 
            events.append("DEADLY_MOVE")

    elif e.BOMB_DROPPED in events:
        #check if dropped the bomb correctly
        if check_escape_route(self, field.copy(), np.array([x,y]), explosion_indices, bombs, others_position)[0]:
            if check_for_crates(self, np.array([x,y]), field.copy()) or np.any(features[FEATURES_TO_INT['OPPONENTS_AROUND_AGENT']] == 1):
                events.append("GOOD_BOMB_PLACEMENT")
            else:
                events.append("BAD_BOMB_PLACEMENT")
        else:
            events.append("DEADLY_MOVE")
    else:
        valid_list = features[FEATURES_TO_INT['VALID_MOVES']].copy()
        valid_list[ np.where( np.logical_or(dangerous_movement == 1, explosion_around_agent == 1) ) ] = 0

        explosion_left , explosion_right, explosion_up, explosion_down = explosion_around_agent
        coin_left , coin_right, coin_up, coin_down = features[FEATURES_TO_INT['DIRECTION_TO_COIN']]
        crate_left , crate_right, crate_up, crate_down = features[FEATURES_TO_INT['DIRECTION_TO_CRATE']]
        opponent_left, opponent_right, opponent_up, opponent_down = features[FEATURES_TO_INT['DIRECTION_TO_OPPONENT']]
        coin_distance, crate_distance, opponent_distance = [features[FEATURES_TO_INT['COIN_DISTANCE']], features[FEATURES_TO_INT['CRATE_DISTANCE']], features[FEATURES_TO_INT['OPPONENT_DISTANCE']]]
        
        coin_weight = coin_distance * 1.5
        crate_weight = crate_distance * 2.3
        opponent_weight = opponent_distance 

        if np.all(valid_list == 0) and e.WAITED in events:
            events.append("WAITING_ONLY_OPTION")
        
        #check if performed a deadly move 
        #->bomb
        elif (danger_left == 1 and e.MOVED_LEFT in events) or (danger_right == 1 and e.MOVED_RIGHT in events) or (danger_up == 1 and e.MOVED_UP in events) or (danger_down == 1 and e.MOVED_DOWN in events) or (danger_wait == 1 and e.WAITED in events):
            events.append("DEADLY_MOVE")
        #->eyplosion
        elif (explosion_left == 1 and e.MOVED_LEFT in events) or (explosion_right == 1 and e.MOVED_RIGHT in events) or (explosion_up== 1 and e.MOVED_UP in events) or (explosion_down== 1 and e.MOVED_DOWN in events):
            events.append("DEADLY_MOVE")

        #check if we moved towards the target witht the best weight right now (coin, crate, enemy)
        elif opponent_weight <= coin_weight and opponent_weight <= crate_weight:
            if (e.MOVED_LEFT in events and opponent_left == 1) or (e.MOVED_RIGHT in events and opponent_right == 1) or (e.MOVED_UP in events and opponent_up == 1) or (e.MOVED_DOWN in events and opponent_down == 1):
                events.append("MOVES_TOWARD_TARGET")
            else:
                events.append("BAD_MOVE")       
        elif coin_weight <= crate_weight:
            if (e.MOVED_LEFT in events and coin_left == 1) or (e.MOVED_RIGHT in events and coin_right == 1) or (e.MOVED_UP in events and coin_up == 1) or (e.MOVED_DOWN in events and coin_down == 1):
                events.append("MOVES_TOWARD_TARGET")
            else:
                events.append("BAD_MOVE")   
        else:
            if (e.MOVED_LEFT in events and crate_left == 1) or (e.MOVED_RIGHT in events and crate_right == 1) or (e.MOVED_UP in events and crate_up == 1) or (e.MOVED_DOWN in events and crate_down == 1):
                events.append("MOVES_TOWARD_TARGET")
            else:
                events.append("BAD_MOVE")   
       
    

    return events

#Methods for the current guess of the Q-function

#TD
def temporal_difference(self, reward, next_state):
    #print("d")
    y = reward
    if next_state is not None:
        #y = y + GAMMA * np.matmul(next_state, self.weights.T).max()
        Q =  [self.regression_forests[action_idx_to_test].predict([next_state]) for action_idx_to_test in range(len(ACTIONS))]
        y = y + GAMMA * np.max(Q)

    return y

def n_step_td(self, episode_rewards, timestep, episode_next_steps):
    y = 0
     
    for i, t in enumerate(range(timestep-1, timestep-1+N)):
        if episode_next_steps[t] is None:
            break

        y = y + np.power(GAMMA, i) * episode_rewards[t] + np.power(GAMMA, N) * np.matmul(episode_next_steps[t], self.weights.T).max()

    return y

def monte_carlo(self, episode_rewards, timestep):
    eps = np.array(episode_rewards.copy())

    y = 0
    #print("monte carlo episode rewards", episode_rewards)
    #for i, t in enumerate(range(timestep-1, len(episode_rewards))):
    #    y = y + np.power(GAMMA, i) * episode_rewards[t]
    
    a = np.arange(len(eps) - (timestep-1) )
    b = eps[timestep-1:]
    c = np.tile(GAMMA, len(eps) - (timestep-1) )

    c = np.power(c, a)

    d = c * b 

    e = np.sum(d)
    """
    print("-------")
    print(e)
    print(y)
    print("--------")
    """

    return e


def sarsa(self, reward, next_state, action):
    #print("--------")
    #print(action)
    #print(reward)
    y = reward
    if next_state is not None:
        y = y + GAMMA * self.regression_forests[ACTION_TO_INT[action]].predict([next_state])

    return y

def check_escape_route(self, field: np.array, starting_point: np.array, explosion_indices: np.array, bombs: list, opponents: np.array):
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
            if bomb[1] - distance[current_position] == -1 or bomb[1] - distance[current_position] == 0:
                #print("update field 1")
                for i in range(-3,4):
                    if bomb[0][0]+i >= 0 and bomb[0][0]+i < s.COLS:
                        updated_field[bomb[0][0]+i,bomb[0][1]] = -1

                    if bomb[0][1]+i >= 0 and bomb[0][1]+i < s.ROWS:
                        updated_field[bomb[0][0],bomb[0][1]+i] = -1

            elif bomb[1] - distance[current_position] < -1:
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

        bombs_found, min_cooldown = check_near_bombs(self, [x,y], field.copy(), bombs, distance[current_position])

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
            if field[x+i][y] == -1:
                break 

            if field[x+i,y] >= 10:
                #print("1")
                bomb_found = True
                min_cooldown = min(min_cooldown, field[x+i,y] - 10)
                

    if check_left and min_cooldown != 0:
        for i in range(1,4):
            if field[x-i][y] == -1:
                break

            if field[x-i,y] >= 10:
                #print("2")
                bomb_found = True
                min_cooldown = min(min_cooldown, field[x-i,y] - 10)

    if check_down and min_cooldown != 0:
        for i in range(1,4):
            if field[x][y+i] == -1:
                break

            if field[x,y+i] >= 10:
                #print("3")
                bomb_found = True
                min_cooldown = min(min_cooldown, field[x,y+i] - 10)

    if check_up and min_cooldown != 0:
        for i in range(1,4):
            if field[x][y-i] == -1:
                break 
            
            if field[x,y-i] >= 10:
                bomb_found = True
                min_cooldown = min(min_cooldown, field[x,y-i] - 10) 

    return bomb_found, min_cooldown


def check_for_crates(self, agent_position: np.array , field: np.array) -> bool:
    x, y = agent_position
    check_left, check_right, check_up, check_down =  np.array([field[x-1,y] != -1 , field[x+1,y] != -1, field[x,y-1] != -1, field[x,y+1] != -1])

    if check_right:
        for i in range(1,4):
            if field[x+i][y] == -1:
                break 

            if field[x+i,y] == 1:
                return True

    if check_left:
        for i in range(1,4):
            if field[x-i][y] == -1:
                break

            if field[x-i,y] == 1:
                return True 

    if check_down:
        for i in range(1,4):
            if field[x][y+i] == -1:
                break

            if field[x,y+i] == 1:
                return True 

    if check_up:
        for i in range(1,4):
            if field[x][y-i] == -1:
                break 
            
            if field[x,y-i] == 1:
                return True 

    return False