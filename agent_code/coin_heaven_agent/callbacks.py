import os
import pickle
import random
import numpy as np

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
    self.D = 7
    self.num_actions = 6

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.weights = np.zeros((self.num_actions, self.D))

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.weights = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .2
    
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        #80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    
    #approximate Q by linear regression
    Q = np.matmul(state_to_features(game_state), self.weights.T)
    action_idx = np.argmax(Q)

    #self.logger.debug("Querying model for action.")
    return ACTIONS[action_idx]


def state_to_features(game_state: dict) -> np.array:
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
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    D = 7
    features = np.zeros(D)

    
    field = np.array(game_state['field'])
    coins = np.array(game_state['coins'])
    _, _, _, (x, y) = game_state['self']

    #we will normalize all features to [0,1]
    
    #our first 4 features are gonna be, what field type is around us
    features[0:4] = (np.array([field[x+1, y], field[x-1, y], field[x, y+1], field[x, y-1]]) +1) / 2

    #as a next feature we take our own position
    features[4] = (y * s.WIDTH + x) / (s.WIDTH * s.HEIGHT)

    #the next feature is gonna be the timestep we are currently in
    features[5] = game_state['step'] / s.MAX_STEPS

    #our last feature is gonna be the direction to the nearest coin (connection vec)
    connection_vec = (coins - np.array([x, y]))
    shortest_vec_indx = np.argmin(np.linalg.norm(connection_vec, axis=1))

    features[6] = (connection_vec[shortest_vec_indx][1] * (s.WIDTH-1) + connection_vec[shortest_vec_indx][0]) / ((s.WIDTH-1) * (s.HEIGHT-1))

    return features

    '''# For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)'''





