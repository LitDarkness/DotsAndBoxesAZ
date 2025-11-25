# evaluations/arena.py
import numpy as np
from tqdm import tqdm

class Arena:
    def __init__(self, player1, player2, game, display=None):
        """
        :param player1: 玩家1实例 (e.g. New Model)
        :param player2: 玩家2实例 (e.g. Old Model)
        :param game: 游戏环境类 (DotsAndBoxes)
        :param display: 可视化函数 (可选)
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose=False):
        """
        执行一局游戏
        返回: 赢家编号 (1 或 -1), 或者是平局的微小分值
        """
        players = [self.player2, None, self.player1]
        cur_player = 1
        game = self.game.clone() # 确保不污染外部环境
        game.reset()
        
        it = 0
        while not game.done:
            it += 1
            if verbose:
                print(f"Turn {it}, Player {cur_player}")
                if self.display:
                    self.display(game)
            
            # 获取动作
            action = players[cur_player + 1].get_action(game)
            
            # 验证动作合法性 (防守性编程)
            valids = game.get_valid_moves()
            if valids[action] == 0:
                print(f"Error: Player {cur_player} 选了非法动作 {action}")
                return -cur_player # 判负
            
            # 执行
            _, _, _, info = game.step(action)
            
            # 连走处理
            if info['captured'] == 0:
                cur_player *= -1
                
        if verbose:
            print("Game Over: ", game.get_game_result())
            
        # 判定胜负
        # 这里的 reward 是相对于 Player 1 的
        scores = game.get_game_result() # [p1, p2]
        return scores[0] - scores[1] # 返回净胜分

    def play_games(self, num, verbose=False):
        """
        执行多局对战，为了公平，双方轮流执先手。
        """
        p1_wins = 0
        p2_wins = 0
        draws = 0
        
        # 前半程: P1 先手
        print("Arena Testing: P1 (New) vs P2 (Old)")
        for _ in tqdm(range(num // 2), desc="P1 Starts"):
            result = self.play_game(verbose=verbose)
            if result > 0: p1_wins += 1
            elif result < 0: p2_wins += 1
            else: draws += 1
            
        # 后半程: 交换角色，P2 (其实是原来的 player2) 此时作为逻辑上的 P1 先手
        # 我们通过交换 self.player1 和 self.player2 来实现
        self.player1, self.player2 = self.player2, self.player1
        
        print("Arena Testing: P2 (Old) vs P1 (New)")
        for _ in tqdm(range(num // 2), desc="P2 Starts"):
            # 注意: 这里 play_game 返回的是当前 self.player1 (也就是 Old) 的净胜分
            result = self.play_game(verbose=verbose)
            if result > 0: p2_wins += 1    # Old 赢了
            elif result < 0: p1_wins += 1  # New 赢了 (Old 输了)
            else: draws += 1
            
        # 换回来
        self.player1, self.player2 = self.player2, self.player1
        
        return p1_wins, p2_wins, draws