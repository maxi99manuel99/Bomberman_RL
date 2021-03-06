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

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'episode_rewards', 'timestep', 'episode_next_states'))

EpisodeTransition = namedtuple('EpisodeTransition',
                        ('state', 'action', 'next_state', 'reward', 'timestep'))                    

# Hyper parameters -- DO modify
GAMMA = 0.4 # discount factor
N = 2 # number of timesteps to look into the future for n_step td

TRANSITION_HISTORY_SIZE = 100000  # keep only ... last transitions
BATCH_SIZE = 20000 # subset of the transitions used for gradient update
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
    features_new = state_to_features(self, new_game_state)
    self.episode_transitions.append(EpisodeTransition(state_to_features(self, old_game_state),self_action, features_new, step_reward, old_game_state['step']))
    self.episode_reward_vector.append(step_reward)
    self.episode_next_states.append(features_new)


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


    """
    SAMPLE A BATCH FROM THE EXPERIENCE BUFFER AND USE IT TO IMPROVIZE THE WEIGHT VECTORS
    """

    #only start fitting after we have tried enough random actions
    if last_game_state['round'] < 99:
        return

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
            if last_game_state['round'] == 99 :
                response = np.array([monte_carlo(self, transition[3], transition[4]) for transition in subbatch])
            else:
                #continue with monte carlo
                response = np.array([monte_carlo(self, transition[3], transition[4]) for transition in subbatch])

                #temporal difference 
                #response = np.array([temporal_difference(self, transition[3][transition[4]-1], transition[2]) for transition in subbatch])

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
        e.COIN_COLLECTED: 50,
        #'STEP': -0.2,
        #e.INVALID_ACTION: -10

        
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            #print(events)
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    #print(reward_sum)
    return reward_sum

def append_custom_events(self,old_game_state: dict, new_game_state: dict, events: List[str]) -> List[str]:
    old_features = state_to_features(self,old_game_state)
    new_features = state_to_features(self,new_game_state)
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

    #a_1 = old_features.reshape((3,3))
    #a_2 = new_features.reshape((3,3))

    #print(a_1)
    #if e.INVALID_ACTION in events:
    #    return events

    #events.append("STEP")

    return events

#Methods for the current guess of the Q-function

#TD
def temporal_difference(self, reward, next_state):
    print("d")
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
    y = 0
    #print("monte carlo episode rewards", episode_rewards)
    for i, t in enumerate(range(timestep-1, len(episode_rewards))):
        y = y + np.power(GAMMA, i) * episode_rewards[t]

    return y

