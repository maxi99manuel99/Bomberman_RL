from collections import namedtuple, deque
from pyexpat import features
from attr import field
import numpy as np
import matplotlib.pyplot as plt

import pickle
from typing import List

from sklearn.linear_model import lasso_path

import events as e
from .callbacks import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TO_INT = {'UP':0, 'RIGHT' : 1 , 'DOWN': 2, 'LEFT': 3, 'WAIT':4, 'BOMB': 5}

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
GAMMA = 0.75 # discount factor

TRANSITION_HISTORY_SIZE = 10000  # keep only ... last transitions
BATCH_SIZE = 500 # subset of the transitions used for gradient update
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
    step_rewards = reward_from_events(self, total_events)
    self.total_rewards = self.total_rewards + step_rewards

    # state_to_features is defined in callbacks.py
    # fill experience buffer with transitions
    self.transitions.append(Transition(state_to_features(self,old_game_state), self_action, state_to_features(self,new_game_state), step_rewards))

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
    step_rewards = reward_from_events(self, events)

    #append last transition (does not work for temporal difference since we needa next step)
    self.transitions.append(Transition(state_to_features(self,last_game_state), last_action, None, step_rewards))

    #reset total rewards per game back to 0
    print(self.total_rewards)
    self.total_rewards = 0

    """
    SAMPLE A BATCH FROM THE EXPERIENCE BUFFER AND USE IT TO IMPROVIZE THE WEIGHT VECTORS
    """

    #random subset of experience buffer
    indices = np.random.choice(np.arange(len(self.transitions), dtype=int), min(len(self.transitions), BATCH_SIZE), replace=False)
    batch = np.array(self.transitions, dtype=Transition)[indices]

    #save a list of the squared losses
    squared_loss = np.zeros(min(len(self.transitions), BATCH_SIZE))
    
    priority_size = min(len(self.transitions), BATCH_PRIORITY_SIZE)

    #calculate the squared loss for each training instance
    for i,transition in enumerate(batch):
        #temporal difference
        y = temporal_difference(self, transition[3], transition[2])

        #other methods
        ...

        squared_loss[i] = np.square( y - transition[0].dot(self.weights[ACTION_TO_INT[transition[1]]]) )
        
    #sort the squarred loss list 
    best_indices = np.argpartition(squared_loss, -priority_size)[-priority_size:]

    #now get the prioritized batch
    batch_priority = batch[best_indices]

    #create subbatch for every action and improve weight vector by gradient update
    for action in ACTIONS:
        subbatch = batch_priority[np.where(batch_priority[:,1] == action)]

        #If an action is not present in our current batch we can not update it. Also, we send the number of rounds to the function to decrease 
        #the learning rate.
        if len(subbatch) != 0:
            gradient_update(self, subbatch, ACTION_TO_INT[action], last_game_state["round"])

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.weights, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    game_rewards = {
        #e.INVALID_ACTION: -300,
        e.KILLED_SELF: -1000,
        #e.COIN_COLLECTED: 30,
        #e.COIN_FOUND: 10,
        #e.CRATE_DESTROYED: 200,
        #'BOMB_HIT_NOTHING': -10,
        #'TOWARDS_COIN': 2, 
        #'NO_COIN': -1,
        'BOMB_NEXT_TO_CRATE': 500,
        'NEXT_TO_CRATE': 10,
        'BOMB_DROPPED_FALSE': -150,
        #e.WAITED: -1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def append_custom_events(self,old_game_state: dict, new_game_state: dict, events: List[str]) -> List[str]:
    features = state_to_features(self,old_game_state)
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
    _, _, _, old_pos = old_game_state['self']
    _, _, _, new_pos =  new_game_state['self']
    
    if len(new_game_state['coins']) != 0 and len(old_game_state['coins']) != 0:
        best_dist_old = np.sum(np.abs(np.subtract(old_game_state['coins'], old_pos)), axis=1).min()
        best_dist_new = np.sum(np.abs(np.subtract(new_game_state['coins'], new_pos)), axis=1).min()

        #check if we got closer to a coin
        if(best_dist_new < best_dist_old):
            events.append("TOWARDS_COIN")

        #every time step we do not collect a coin
        if e.COIN_COLLECTED not in events:
            events.append("NO_COIN")

    #if e.BOMB_EXPLODED in events and e.CRATE_DESTROYED not in events and e.KILLED_OPPONENT not in events:
    #    events.append("BOMB_HIT_NOTHING")

    field = np.array(old_game_state['field'])

    new_next_to_crate = field[new_pos[0]-1,new_pos[1]] ==1 or field[new_pos[0]+1,new_pos[1]] ==1 or field[new_pos[0],new_pos[1]-1] == 1 or field[new_pos[0],new_pos[1]+1] == 1
    old_next_to_crate = field[old_pos[0]-1,old_pos[1]] ==1 or field[old_pos[0]+1,old_pos[1]] ==1 or field[old_pos[0],old_pos[1]-1] == 1 or field[old_pos[0],old_pos[1]+1] == 1
    
    if e.BOMB_DROPPED in events and old_next_to_crate:
        events.append("BOMB_NEXT_TO_CRATE")

    if e.BOMB_DROPPED in events and not old_next_to_crate:
        events.append("BOMB_DROPPED_FALSE")

    if new_next_to_crate:
        events.append("NEXT_TO_CRATE")

    return events
    
#Gradient update to improvize the weight vectors
def gradient_update(self, subbatch: List[Transition], action_index: int, round: int):
    """
    improve weight vector

    Args:
        subbatch (List[Transition]): the subbatch for the current action
        action_index: int: The index of the action used to update the correct
        position in the weight vector
    """

    #gradient update hyperparameter
    alpha = 0.8

    #sum in the gradient update formula
    sum = 0
    for transition in subbatch:
        #temporal difference
        y = temporal_difference(self, transition[3], transition[2])

        #other Methods
        #...
        
        sum = sum + transition[0] * (y - transition[0].dot(self.weights[action_index]))
    
    self.weights[action_index] = self.weights[action_index] + (alpha / (len(subbatch) * round)) * sum


#Methods for the current guess of the Q-function

#TD
def temporal_difference(self, reward, next_state):
    y = reward
    if next_state is not None:
        y = y + GAMMA * np.matmul(next_state, self.weights.T).max()

    return y
