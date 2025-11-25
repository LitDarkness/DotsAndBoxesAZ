# players/heuristic_player.py

import numpy as np
import random
from .base_player import BasePlayer  # 相对导入，只有在作为包被调用时才有效
from config import Config

class HeuristicPlayer(BasePlayer):
    def __init__(self, name="HeuristicBot"):
        super().__init__(name)

    def get_action(self, game):
        valid_moves = np.where(game.get_valid_moves() == 1)[0]
        
        # 1. 尝试寻找得分机会 (Greedy Strategy)
        for action in valid_moves:
            if self._creates_box(game, action):
                return action
        
        # 2. 如果没有得分机会，寻找"安全"的移动 (Safe Strategy)
        # 安全移动定义：走了这一步后，不会让任何格子变成 3 条边 (给对手送分)
        safe_moves = []
        for action in valid_moves:
            if not self._gives_away_box(game, action):
                safe_moves.append(action)
                
        if len(safe_moves) > 0:
            return random.choice(safe_moves)
        
        # 3. 如果必须送分（全是坏棋），随机选一个
        return random.choice(valid_moves)

    def _action_to_coord(self, game, action):
        # 复用 game 的逻辑，这里手动算一下避免访问私有方法，或者直接调用 game._action_to_coord 如果允许
        # 这里为了稳健，简单重写一次解析逻辑
        num_h = Config.NUM_HORIZONTAL_EDGES
        if action < num_h:
            row = action // game.cols
            col = action % game.cols
            return 'h', row, col
        else:
            idx = action - num_h
            row = idx // (game.cols + 1)
            col = idx % (game.cols + 1)
            return 'v', row, col

    def _count_box_edges(self, game, r, c):
        """计算坐标 (r,c) 的格子当前有几条边"""
        count = 0
        if game.horz_edges[r, c] == 1: count += 1
        if game.horz_edges[r+1, c] == 1: count += 1
        if game.vert_edges[r, c] == 1: count += 1
        if game.vert_edges[r, c+1] == 1: count += 1
        return count

    def _creates_box(self, game, action):
        """预判：如果走这步，会不会构成格子"""
        etype, r, c = self._action_to_coord(game, action)
        
        # 模拟走这步，看相关格子的边数是否变成 4
        # 注意：这里我们只通过逻辑判断，不真正修改 game state
        
        if etype == 'h':
            # 检查上方 (r-1, c) 和 下方 (r, c)
            if r > 0 and self._count_box_edges(game, r-1, c) == 3: return True
            if r < game.rows and self._count_box_edges(game, r, c) == 3: return True
        else:
            # 检查左方 (r, c-1) 和 右方 (r, c)
            if c > 0 and self._count_box_edges(game, r, c-1) == 3: return True
            if c < game.cols and self._count_box_edges(game, r, c) == 3: return True
            
        return False

    def _gives_away_box(self, game, action):
        """预判：如果走这步，会不会让某个格子的边数变成 3 (给对手送分)"""
        etype, r, c = self._action_to_coord(game, action)
        
        # 逻辑：如果相关格子的当前边数是 2，我走了一步变成 3，那就是送分
        if etype == 'h':
            if r > 0 and self._count_box_edges(game, r-1, c) == 2: return True
            if r < game.rows and self._count_box_edges(game, r, c) == 2: return True
        else:
            if c > 0 and self._count_box_edges(game, r, c-1) == 2: return True
            if c < game.cols and self._count_box_edges(game, r, c) == 2: return True
            
        return False

