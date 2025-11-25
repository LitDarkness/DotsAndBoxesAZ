# model/mcts.py
import sys
import os
import math
import numpy as np
import torch
from model.solver import MinimaxSolver

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config import Config

class MCTS:
    def __init__(self, nnet):
        self.nnet = nnet
        self.args = Config
        
        self.Qsa = {}       # Q(s,a)
        self.Nsa = {}       # N(s,a)
        self.Ns = {}        # N(s)
        self.Ps = {}        # P(s)
        self.Es = {}        # E(s)
        self.Vs = {}        # Valid moves

    def get_action_prob(self, game, temp=1):
        """
        获得动作概率。
        """
        # 1. 模拟搜索
        for i in range(self.args.NUM_MCTS_SIMS):
            self.search(game.clone())

        s = self.get_state_string(game)
        
        # 兜底：防止根节点未被扩展
        if s not in self.Ns:
             self.search(game.clone())

        counts = [self.Nsa.get((s, a), 0) for a in range(self.args.ACTION_SIZE)]

        # 【修复 Crash 的核心兜底】
        # 如果 Solver 导致没有产生任何访问计数，或者搜索次数太少
        # 我们必须返回一个"合法的"均匀分布，绝对不能包含非法动作
        if sum(counts) == 0:
            valid_moves = game.get_valid_moves()
            # 防止除以0 (极罕见情况：无路可走)
            if np.sum(valid_moves) > 0:
                probs = valid_moves / np.sum(valid_moves)
            else:
                probs = valid_moves # 全0
            return probs

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        
        if counts_sum == 0:
            # 再次兜底
            valid_moves = game.get_valid_moves()
            return valid_moves / np.sum(valid_moves)
            
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, game):
        s = self.get_state_string(game)

        # ---------------------------------------------------------
        # 1. 终止状态检查 (真正的 Game Over)
        # ---------------------------------------------------------
        if s not in self.Es:
            if game.done:
                scores = game.get_game_result()
                diff = scores[0] - scores[1]
                total = game.rows * game.cols
                
                # 使用归一化分差
                normalized_v = diff / total
                
                if game.current_player == 1:
                    self.Es[s] = normalized_v
                else:
                    self.Es[s] = -normalized_v
            else:
                self.Es[s] = None

        if self.Es[s] is not None:
            return self.Es[s]

        # ---------------------------------------------------------
        # 2. 扩展节点 (Expansion)
        # ---------------------------------------------------------
        if s not in self.Ns:
            # A. 获取合法动作
            valid_moves = game.get_valid_moves()
            self.Vs[s] = valid_moves
            
            # B. 混合求解器 (Hybrid Solver) 接管
            # -----------------------------------------------
            remaining_moves = np.sum(valid_moves)
            # 3x3 棋盘建议设为 8-10。
            # 如果设得太高，Python 递归会很慢；设得太低，覆盖面不够。
            SOLVER_THRESHOLD = -1 
            
            if remaining_moves <= SOLVER_THRESHOLD:
                # 调用上帝视角
                exact_diff = MinimaxSolver.solve(game)
                
                # 转换视角
                if game.current_player == 1:
                    v = exact_diff
                else:
                    v = -exact_diff
                
                # 【关键修复】
                # 即使是 Solver 算出结果，也必须初始化 Ps 和 Ns
                # 否则 MCTS 树在这里断掉了，get_action_prob 就会报错
                # 我们给所有合法动作均匀概率，让 MCTS 框架能继续选子节点
                self.Ps[s] = valid_moves / np.sum(valid_moves)
                self.Ns[s] = 0
                
                # 可以选择是否缓存 v 到 Es，这里暂不缓存以允许重复走这个逻辑(虽然有点浪费)
                # 或者：self.Es[s] = v (如果确定 Solver 是完美的)
                return v

            # C. 神经网络预测
            # -----------------------------------------------
            canonical_board = game.get_canonical_state()
            
            # 转 Tensor
            board_tensor = torch.tensor(np.ascontiguousarray(canonical_board), dtype=torch.float32).unsqueeze(0)
            if torch.cuda.is_available():
                board_tensor = board_tensor.cuda()
            
            self.nnet.eval()
            with torch.no_grad():
                pi, v = self.nnet(board_tensor)

            pi = torch.exp(pi).data.cpu().numpy()[0]
            pi = pi * valid_moves
            sum_pi = np.sum(pi)
            if sum_pi > 0:
                pi /= sum_pi
            else:
                pi = valid_moves / np.sum(valid_moves)

            self.Ps[s] = pi
            self.Ns[s] = 0
            return v.item()

        # 3. 选择 (Selection)
        valid_moves = self.Vs[s]
        best_uct = -float('inf')
        best_act = -1
        c_puct = self.args.C_PUCT
        sqrt_ns = math.sqrt(self.Ns[s])

        for a in range(self.args.ACTION_SIZE):
            if valid_moves[a]:
                if (s, a) in self.Qsa:
                    q = self.Qsa[(s, a)]
                    n = self.Nsa[(s, a)]
                else:
                    q = 0
                    n = 0
                
                u = c_puct * self.Ps[s][a] * sqrt_ns / (1 + n)
                
                if q + u > best_uct:
                    best_uct = q + u
                    best_act = a

        a = best_act
        
        # 4. 模拟与递归
        # 注意：这里需要处理连走逻辑导致的回溯 value 翻转问题
        # 记录 step 之前的玩家
        current_player_before_step = game.current_player
        
        game.step(a)
        v = self.search(game)

        # 5. 回溯 (Backprop)
        # 如果 step 之后玩家换人了，说明这一步没有连走，下一层的 v 是对手视角的，要取反
        # 如果没换人（连走），v 还是我方视角的，保持原样
        if game.current_player != current_player_before_step:
            v = -v

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v

    def get_state_string(self, game):
        return game.horz_edges.tobytes() + game.vert_edges.tobytes() + game.boxes.tobytes() + str(game.current_player).encode()