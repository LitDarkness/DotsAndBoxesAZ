from abc import ABC, abstractmethod

class BasePlayer(ABC):
    def __init__(self, name="Unknown"):
        self.name = name

    @abstractmethod
    def get_action(self, game):
        """
        输入: 当前游戏实例 (DotsAndBoxes)
        输出: 动作索引 (int)
        """
        pass