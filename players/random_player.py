# players/random_player.py
import numpy as np
import random
from .base_player import BasePlayer

class RandomPlayer(BasePlayer):
    def get_action(self, game):
        valid = np.where(game.get_valid_moves() == 1)[0]
        return random.choice(valid)