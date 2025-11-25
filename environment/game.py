
# environment/game.py
import numpy as np
from config import Config

class DotsAndBoxes:
    def __init__(self):
        self.rows = Config.BOARD_ROWS
        self.cols = Config.BOARD_COLS
        self.action_size = Config.ACTION_SIZE
        
        self.reset()

    def reset(self):
        """重置游戏状态"""
        # 0: 未占, 1: 已占
        # 形状: (rows+1, cols)
        self.horz_edges = np.zeros((self.rows + 1, self.cols), dtype=np.int8)
        # 形状: (rows, cols+1)
        self.vert_edges = np.zeros((self.rows, self.cols + 1), dtype=np.int8)
        
        # 0: 空, 1: 玩家1, -1: 玩家2
        self.boxes = np.zeros((self.rows, self.cols), dtype=np.int8)
        
        # 1 表示玩家1先手, -1 表示玩家2
        self.current_player = 1
        self.last_action = None
        self.done = False
        
        return self.get_state()
    
    def clone(self):
        """
        【性能优化核心】
        手动克隆游戏状态，比 copy.deepcopy 快 10-20 倍。
        我们只复制必要的 numpy 数组和标量。
        """
        # 创建一个新的空实例 (避开 __init__ 的开销)
        new_game = DotsAndBoxes.__new__(DotsAndBoxes)
        
        # 复制基础标量
        new_game.rows = self.rows
        new_game.cols = self.cols
        new_game.action_size = self.action_size
        new_game.current_player = self.current_player
        new_game.done = self.done
        
        # 复制 Numpy 数组 (使用 C 语言级别的 copy)
        new_game.horz_edges = self.horz_edges.copy()
        new_game.vert_edges = self.vert_edges.copy()
        new_game.boxes = self.boxes.copy()
        
        # 如果有 last_action 记录，也复制一下
        new_game.last_action = self.last_action
        
        return new_game
    
    def get_state(self):
        """
        返回原始的游戏状态数据，用于内部逻辑
        """
        return {
            'horz_edges': self.horz_edges.copy(),
            'vert_edges': self.vert_edges.copy(),
            'boxes': self.boxes.copy(),
            'current_player': self.current_player
        }

    def get_canonical_state(self):
        """
        【关键】为神经网络准备的输入状态
        返回形状: (3, 2*rows+1, 2*cols+1)
        通道0: 当前玩家拥有的格子 (1=有, 0=无)
        通道1: 对手拥有的格子
        通道2: 所有的边状态 (1=有边, 0=无边)
        
        注：为了保持空间关系，我们将边和格子映射到一个更高分辨率的网格上，或者简单堆叠。
        这里采用最适合 CNN 的 Feature Plane 堆叠方式，形状为 (3, rows, cols) 略显不够，
        因为边是在格子之间的。
        
        为了简化模型训练，我们采用 Feature Planes 堆叠：
        Channel 0: 玩家自己的格子 (rows, cols)
        Channel 1: 对手的格子 (rows, cols)
        Channel 2: 剩下的合法动作提示 (可选，这里暂时只用边信息)
        
        *改进方案*: 为了让CNN最好地理解，我们构建 (3, rows+1, cols+1) 的矩阵不够直观。
        这里使用最通用的 AlphaZero 编码方式：
        Input: (3, rows, cols) -> 信息量丢失边。
        Input: (3, 2*rows+1, 2*cols+1) -> 稀疏矩阵，但保留空间几何关系。
        """
        H, W = 2 * self.rows + 1, 2 * self.cols + 1
        state = np.zeros((3, H, W), dtype=np.float32)
        
        # 填充边信息 (Channel 2)
        # 偶数行奇数列是横边，奇数行偶数列是竖边
        for r in range(self.rows + 1):
            for c in range(self.cols):
                if self.horz_edges[r, c] == 1:
                    state[2, 2*r, 2*c+1] = 1.0
                    
        for r in range(self.rows):
            for c in range(self.cols + 1):
                if self.vert_edges[r, c] == 1:
                    state[2, 2*r+1, 2*c] = 1.0

        # 填充格子信息 (Channel 0: Self, Channel 1: Opponent)
        # 格子位于 (2r+1, 2c+1)
        for r in range(self.rows):
            for c in range(self.cols):
                owner = self.boxes[r, c]
                if owner == self.current_player:
                    state[0, 2*r+1, 2*c+1] = 1.0
                elif owner == -self.current_player:
                    state[1, 2*r+1, 2*c+1] = 1.0
                    
        return state

    def get_valid_moves(self):
        """返回一个长度为 ACTION_SIZE 的二进制向量，1表示合法，0表示非法"""
        valid = np.zeros(self.action_size, dtype=np.int8)
        
        # 检查横边
        h_flat = self.horz_edges.flatten()
        valid[:len(h_flat)] = (h_flat == 0)
        
        # 检查竖边
        v_flat = self.vert_edges.flatten()
        valid[len(h_flat):] = (v_flat == 0)
        
        return valid

    def _action_to_coord(self, action):
        """将整数动作转换为 (type, row, col)"""
        num_h = Config.NUM_HORIZONTAL_EDGES
        if action < num_h:
            # 横边
            row = action // self.cols
            col = action % self.cols
            return 'h', row, col
        else:
            # 竖边
            idx = action - num_h
            row = idx // (self.cols + 1)
            col = idx % (self.cols + 1)
            return 'v', row, col

    def step(self, action):
        """
        执行一步动作
        返回: next_state, reward, done, info
        """
        if self.done:
            raise ValueError("Game is already over")
            
        # 1. 解析动作
        etype, r, c = self._action_to_coord(action)
        
        # 2. 合法性检查
        if etype == 'h':
            if self.horz_edges[r, c] == 1:
                raise ValueError(f"Invalid move: Horizontal {r},{c} already taken")
            self.horz_edges[r, c] = 1
        else:
            if self.vert_edges[r, c] == 1:
                raise ValueError(f"Invalid move: Vertical {r},{c} already taken")
            self.vert_edges[r, c] = 1
            
        # 3. 检查是否构成了格子 (获得分数)
        captured_count = 0
        
        # 检查该边周围可能形成的格子
        if etype == 'h':
            # 检查上方格子 (r-1, c)
            if r > 0 and self._is_box_complete(r-1, c):
                self.boxes[r-1, c] = self.current_player
                captured_count += 1
            # 检查下方格子 (r, c)
            if r < self.rows and self._is_box_complete(r, c):
                self.boxes[r, c] = self.current_player
                captured_count += 1
        else: # etype == 'v'
            # 检查左方格子 (r, c-1)
            if c > 0 and self._is_box_complete(r, c-1):
                self.boxes[r, c-1] = self.current_player
                captured_count += 1
            # 检查右方格子 (r, c)
            if c < self.cols and self._is_box_complete(r, c):
                self.boxes[r, c] = self.current_player
                captured_count += 1
        
        # 4. 状态流转
        info = {'captured': captured_count}
        
        # 检查游戏是否结束
        if np.all(self.boxes != 0):
            self.done = True
        
        # 5. 处理 Extra Move 机制
        # 如果这一步得分了，current_player 不变 (奖励机制将在 MCTS 层面处理)
        # 如果没得分，切换玩家
        if captured_count == 0:
            self.current_player *= -1
            
        # 计算最终奖励 (仅当游戏结束时返回)
        reward = 0
        if self.done:
            p1_score = np.sum(self.boxes == 1)
            p2_score = np.sum(self.boxes == -1)
            # 如果当前视角是 P1
            if p1_score > p2_score:
                reward = 1
            elif p2_score > p1_score:
                reward = -1
            else:
                reward = 0 # 平局
            
            # 调整奖励视角：如果最后一步走完是 P2，但计算的是 P1 的分，
            # 实际上 MCTS 使用的是 value head，不需要在这里做复杂的相对奖励转换，
            # 而是依赖 get_canonical_state 的视角。
            
        return self.get_state(), reward, self.done, info

    def _is_box_complete(self, r, c):
        """检查坐标 (r,c) 的格子是否四边都已连接"""
        # 上: horz(r, c), 下: horz(r+1, c)
        # 左: vert(r, c), 右: vert(r, c+1)
        return (self.horz_edges[r, c] == 1 and
                self.horz_edges[r+1, c] == 1 and
                self.vert_edges[r, c] == 1 and
                self.vert_edges[r, c+1] == 1)

    def get_game_result(self):
        """返回游戏结果列表 [p1_score, p2_score]"""
        return [np.sum(self.boxes == 1), np.sum(self.boxes == -1)]

# 简单测试代码
if __name__ == "__main__":
    game = DotsAndBoxes()
    print(f"初始化 3x3 游戏。动作空间: {game.action_size}")
    
    # 模拟走一步
    # 假设 action 0 是第一行第一个横边
    state, reward, done, info = game.step(0)
    print(f"Player {game.current_player} 走了 Action 0. 状态: {info}")
    
    valid_moves = game.get_valid_moves()
    print(f"剩余合法动作数: {np.sum(valid_moves)}")