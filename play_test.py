import numpy 

import pyspiel
from open_spiel.python.bots import human
import bot.pdcoea as pdbot


bots = [pdbot.PdcoeaBot('model_2_state_dict.pth'),human.HumanBot()]


def play_game(bots):
    game = pyspiel.load_game("quoridor")
    state = game.new_initial_state()
    print("Initial state:\n{}".format(state))

    while not state.is_terminal():
        current_player = state.current_player()
        bot = bots[current_player]
        action = bot.step(state)
        state.apply_action(action)
        action_str = state.action_to_string(current_player, action)
        print("Player {} sampled action: {}".format(current_player,
                                                        action_str))
        print(state)

    returns = state.returns()
    print(returns)
    print("game complete!")


play_game(bots)  