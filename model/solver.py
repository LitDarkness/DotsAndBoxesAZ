# model/solver.py
import numpy as np

class MinimaxSolver:
    """
    针对残局的精确求解器。
    使用 Alpha-Beta 剪枝的 Minimax 算法。
    """
    
    @staticmethod
    def solve(game):
        """
        计算当前局面的精确价值 (归一化分差)
        返回: 当前玩家预期的最终收益 [-1, 1]
        """
        # 获取总格子数，用于归一化
        total_boxes = game.rows * game.cols
        
        # 启动递归搜索
        # alpha, beta 初始化
        score_diff = MinimaxSolver._minimax(game, -999, 999)
        
        # 归一化返回
        return score_diff / total_boxes

    @staticmethod
    def _minimax(game, alpha, beta):
        """
        递归核心
        返回: 最终 P1得分 - P2得分 (绝对分差)
        """
        if game.done:
            scores = game.get_game_result()
            return scores[0] - scores[1] # P1 - P2

        valid_moves = np.where(game.get_valid_moves() == 1)[0]
        
        # 启发式排序：这里简单乱序，如果有 heuristic 会更快
        # 在点格棋中，通常不需要太复杂的排序，因为残局分支不多
        
        # 区分当前是 Max 节点 (P1) 还是 Min 节点 (P2)
        if game.current_player == 1:
            max_eval = -999
            for action in valid_moves:
                # 必须 clone，否则会破坏上一层的状态
                # 注意：频繁 clone 是 Python 的性能瓶颈，
                # 但在剩余步数 < 10 时，计算量大约是几千次，是可以接受的
                next_game = game.clone()
                next_game.step(action)
                
                eval_val = MinimaxSolver._minimax(next_game, alpha, beta)
                max_eval = max(max_eval, eval_val)
                
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break # Beta 剪枝
            return max_eval
        
        else: # current_player == -1 (P2)
            min_eval = 999
            for action in valid_moves:
                next_game = game.clone()
                next_game.step(action)
                
                eval_val = MinimaxSolver._minimax(next_game, alpha, beta)
                min_eval = min(min_eval, eval_val)
                
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break # Alpha 剪枝
            return min_eval