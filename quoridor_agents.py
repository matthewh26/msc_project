import numpy as np

import torch
import torch.nn as nn

from absl import app
from absl import flags

import pyspiel
import pdcoea

FLAGS = flags.FLAGS


class NeuralNetwork(nn.Module):
    def __init__(self, layer_params):
        super().__init__()
        self.lin1 = nn.Linear(1445, 128)
        self.lin2 = nn.Linear(128, 289)
        self.log_softmax = nn.LogSoftmax()
        self.layer_params = layer_params
        self.init_layer_params()

    def init_layer_params(self):
        l1_weights = torch.from_numpy(np.array(self.layer_params[:184960]).reshape((128,1445))).to(torch.float32)
        l1_bias = torch.from_numpy(np.array(self.layer_params[184960:185088])).to(torch.float32)
        l2_weights = torch.from_numpy(np.array(self.layer_params[185088:222080]).reshape((289,128))).to(torch.float32)
        l2_bias = torch.from_numpy(np.array(self.layer_params[222080:])).to(torch.float32)
        self.lin1.weight.data = l1_weights
        self.lin1.bias.data = l1_bias
        self.lin2.weight.data = l2_weights
        self.lin2.bias.data = l2_bias        
    
    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = self.log_softmax(x)
        return x



def dominance_function(x, y, game_string):

    players = [NeuralNetwork(x),NeuralNetwork(y)]
    
    game = pyspiel.load_game(game_string)
    state = game.new_initial_state()

    turn_player = 0
    for i in range(100):
        input = torch.Tensor(state.observation_tensor())
        out = players[turn_player](input)
        out = out.detach().numpy()
        while True:
            action = np.argmax(out)
            if action in state.legal_actions(state.current_player()):
                break
            else:
                out[action] = np.min(out)
        state.apply_action(action)


        turn_player = abs(turn_player-1)
        if state.is_terminal():
            print(state.returns()[0])
            return state.returns()[0]
    print("0")
    return 0


def main(_):
    flags.DEFINE_string("game_string", "quoridor", "Game string")
    #print(dominance_function(np.random.normal(0,1,222369),np.random.normal(1,1,222369),FLAGS.game_string))
    
    a, b = pdcoea.pdcoea(pop_size = 10,
                  chromosome_len = 222369,
                  epochs = 10000,
                  g = dominance_function,
                  chi = 0.1,
                  real_valued = True,
                  mean = 0,
                  stdv = 5,
                  game_string = FLAGS.game_string
    )
    print('complete!')

    torch.save(NeuralNetwork(a[0]).state_dict(), 'model_3_state_dict.pth')


if __name__ == "__main__":

    app.run(main)

    

    
