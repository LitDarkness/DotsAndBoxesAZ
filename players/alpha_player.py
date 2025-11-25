
# players/alpha_player.py
import numpy as np
from .base_player import BasePlayer
from config import Config

class AlphaPlayer(BasePlayer):
    def __init__(self, nnet, mcts, temp=0, name="AlphaZero"):
        """
        :param nnet: 神经网络实例
        :param mcts: MCTS 实例
        :param temp: 温度参数 (0=贪婪/竞技, 1=探索/训练)
        """
        super().__init__(name)
        self.nnet = nnet
        self.mcts = mcts
        self.temp = temp

    def get_action(self, game):
        # 使用 MCTS 计算动作概率
        # 注意: 这里一定要传 game 的副本或者让 MCTS 内部处理
        # 我们的 mcts.get_action_prob 已经处理了 clone，所以直接传 game 即可
        probs = self.mcts.get_action_prob(game, temp=self.temp)
        
        if self.temp == 0:
            # 贪婪模式：直接选概率最大的
            return np.argmax(probs)
        else:
            # 探索模式：按概率采样
            action = np.random.choice(len(probs), p=probs)
            return action
            
    def update_temp(self, temp):
        self.temp = temp