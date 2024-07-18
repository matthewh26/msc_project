import pyspiel
import numpy as np
from quoridor_agents import NeuralNetwork


class PdcoeaBot():
    def __init__(self,state_dict):
        self.state_dict = state_dict
        self.network = NeuralNetwork()
        self.network.load_state_dict(self.state_dict)

    def step(self,state):

        legal_actions = state.legal_actions(state.current_player())
        out = self.network(state.observation_tensor())
        while True:
            action = np.argmax(out)
            if action in legal_actions:
                break
            else:
                out[action] = np.min(out)
        return action 


