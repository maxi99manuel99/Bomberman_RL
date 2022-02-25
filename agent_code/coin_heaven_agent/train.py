from collections import namedtuple, deque
import numpy as np

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 20  # keep only ... last transitions
GAMMA = 0.6 # discount factor
BATCH_SIZE = 10 # subset of the transitions used for gradient update
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

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

    #Append custom events for rewards
    total_events = append_custom_events(old_game_state, new_game_state, events)

    #calculate total reward for this timestep
    step_rewards = reward_from_events(self, total_events)

    # state_to_features is defined in callbacks.py
    # fill experience buffer with transitions
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), step_rewards))


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

    #append last transition
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, step_rewards))

    #random subbatch of TS for gradient update
    batch = np.random.choice(self.transitions, BATCH_SIZE, replace=False)

    #create subbatch for every action
    #and improve weight vector by gradient update
    for i, action in enumerate(ACTIONS):
        subbatch = batch[np.where(batch[:,'action'] == action)]
        gradient_update(self, subbatch, i)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.INVALID_ACTION: -10,
        e.COIN_COLLECTED: 5,
        'TOWARDS_COIN': 1,
        'NO_COIN': -2  
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def gradient_update(self, subbatch: List[Transition], action_index: int):
    """
    improve weight vector

    Args:
        subbatch (List[Transition]): the subbatch for the current action
        action_index: int: The index of the action used to update the correct
        position in the weight vector
    """

    #gradient update hyperparameter
    alpha = 0.4

    #sum in the gradient update formula
    sum = 0
    for transition in subbatch:
        #temporal difference
        y = transition['reward'] + GAMMA * np.matmul(transition['next_state'], self.weights).max()
        sum = sum + transition['state'] * (y - transition['state'].dot(self.weights[action_index]))
    
    self.weights[action_index] = self.weights[action_index] + (alpha / len(subbatch)) * sum

def append_custom_events(old_game_state: dict, new_game_state: dict, events: List[str]) -> List[str]:
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
    best_dist_old = best_dist = np.sum(np.abs(np.subtract(old_game_state['coins'], old_pos)), axis=1).min()
    best_dist_new = np.sum(np.abs(np.subtract(new_game_state['coins'], new_pos)), axis=1).min()

    #check if we got closer to a coin
    if(best_dist_new < best_dist_old):
        events.append("TOWARDS_COIN")

    #every time step we do not collect a coin
    if e.COIN_COLLEDCTED not in events:
        events.append("NO_COIN")

    return events
    

