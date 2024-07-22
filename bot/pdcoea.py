import torch
import numpy as np
from quoridor_agents import NeuralNetwork


class PdcoeaBot():
    def __init__(self,state_dict):
        self.state_dict = state_dict
        self.network = NeuralNetwork(np.zeros(222369))
        self.network.load_state_dict(torch.load(self.state_dict))

    def step(self,state):

        legal_actions = state.legal_actions(state.current_player())
        input = torch.Tensor(state.observation_tensor())
        out = self.network(input)
        out = out.detach().numpy()
        while True:
            action = np.argmax(out)
            if action in legal_actions:
                break
            else:
                out[action] = np.min(out)
        return action 



