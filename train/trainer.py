# train/trainer.py
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from random import shuffle
from tqdm import tqdm
import time

# -----------------------------------------------------
# 包管理与路径修复
# -----------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config import Config
from model.mcts import MCTS
from players.alpha_player import AlphaPlayer
from evaluations.arena import Arena
from train.utils import CheckpointManager
from players.heuristic_player import HeuristicPlayer

class Trainer:
    def __init__(self, game, nnet):
        """
        :param game: 游戏环境实例
        :param nnet: 当前待训练的神经网络
        """
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__()  # 用于评估的旧网络 (Previous Net)
        self.args = Config
        self.mcts = MCTS(self.nnet)
        
        # 工具类
        self.cm = CheckpointManager()
        
        # 【优化1】使用双端队列实现滑动窗口
        # maxlen 限制历史数据量，防止内存爆炸，同时保证数据的新鲜度
        # 假设保留最近 20 次迭代的数据
        self.train_examples_history = deque(maxlen=self.args.TRAIN_WINDOW_SIZE) 
        
        # 标记是否是从断点恢复的
        self.loaded_from_checkpoint = False
    def generate_heuristic_batch(self, num_games):
        """
        生成脚本对战数据 (用于模仿学习)
        """
        data = []
        # 初始化两个脚本互殴
        h1 = HeuristicPlayer("Teacher1")
        h2 = HeuristicPlayer("Teacher2")
        
        print(f"正在生成 {num_games} 局脚本对战数据 (Imitation Data)...")
        
        # 纯CPU逻辑，跑得飞快
        for _ in tqdm(range(num_games), desc="Generating Script Data"):
            game = self.game.clone()
            game.reset()
            episode_data = []
            
            while not game.done:
                # 获取当前玩家视角
                current_p = h1 if game.current_player == 1 else h2
                
                # 1. 脚本直接给出动作
                action = current_p.get_action(game)
                
                # 2. 构造 One-Hot 策略向量 (模仿学习的核心)
                # 也就是说：神经网络，你要学会在这个局面下，100% 选这个动作
                pi = np.zeros(self.args.ACTION_SIZE)
                pi[action] = 1.0
                
                # 3. 收集数据
                sym = game.get_canonical_state()
                episode_data.append([sym, game.current_player, pi, None])
                
                # 4. 执行
                _, _, _, info = game.step(action)
                
                # 连走逻辑
                if info['captured'] == 0:
                    pass # 这里其实不需要切换 player 变量，因为 game 内部切了
            
            # 5. 游戏结束，回填胜负
            scores = game.get_game_result()
            diff = scores[0] - scores[1]
            
            for x in episode_data:
                board, player, pi, _ = x
                if player == 1:
                    v = 1 if diff > 0 else -1 if diff < 0 else 0
                else:
                    v = 1 if diff < 0 else -1 if diff > 0 else 0
                data.append((board, pi, v))
                
        return data

    def pretrain_with_script(self, num_games=1000, epochs=5):
        """
        【新增功能】预训练模式
        """
        print("\n>>> 启动模仿学习 (Imitation Learning) <<<")
        print("AI 将先观看脚本互博，快速掌握基础规则...")
        
        # 1. 生成数据
        examples = self.generate_heuristic_batch(num_games)
        print(f"生成了 {len(examples)} 个局面样本。")
        
        # 2. 训练神经网络
        # 临时调整一下参数，让它学快点
        original_epochs = self.args.EPOCHS
        self.args.EPOCHS = epochs
        
        self.train_network(examples)
        
        self.args.EPOCHS = original_epochs # 还原参数
        
        # 3. 保存为 best 模型，作为后续 RL 的起点
        print(">>> 预训练完成！已保存为 'best.pth.tar'")
        self.cm.save_checkpoint(self.nnet, "best.pth.tar", is_best=True)
        self.cm.save_checkpoint(self.nnet, "checkpoint_latest.pth.tar")
        
        # 也就是让 Self-Play 历史池里先装满这些高质量数据
        self.train_examples_history.append(examples)

    def try_load_latest(self):
        """
        尝试加载最新的检查点，实现断点续训
        """
        try:
            # 尝试加载 'checkpoint_latest.pth.tar'
            self.cm.load_checkpoint(self.nnet, 'checkpoint_latest.pth.tar')
            print(">>> 成功恢复上次训练进度！")
            self.loaded_from_checkpoint = True
            return True
        except FileNotFoundError:
            print(">>> 未发现中断的训练进度，将从头开始。")
            return False
        except Exception as e:
            print(f">>> 加载检查点出错: {e}，将从头开始。")
            return False

    def execute_episode(self):
        """
        执行一局自我博弈
        """
        train_examples = []
        game = self.game.clone()
        self.mcts = MCTS(self.nnet)
        episode_step = 0
        
        while True:
            episode_step += 1
            # 前期探索，后期利用
            temp = 1 if episode_step <= 10 else 0
            
            pi = self.mcts.get_action_prob(game, temp=temp)
            sym = game.get_canonical_state()
            train_examples.append([sym, game.current_player, pi, None])
            
            action = np.random.choice(len(pi), p=pi)
            _, _, _, info = game.step(action)
            
            if game.done:
                # ---------------------------------------------------------
                # 【修改点】: 引入分数差奖励 (Score Difference Reward)
                # ---------------------------------------------------------
                scores = game.get_game_result() # [p1_score, p2_score]
                
                # 计算净胜分差
                diff = scores[0] - scores[1]
                
                # 归一化到 [-1, 1] 之间
                # 3x3 总格数 = 9。
                # 9:0 -> diff=9 -> reward=1.0
                # 5:4 -> diff=1 -> reward=0.111
                total_boxes = self.game.rows * self.game.cols
                normalized_score = diff / total_boxes
                
                return_data = []
                for x in train_examples:
                    board, player, pi, _ = x
                    
                    # 这里的 player 是保存该局面时的行动者
                    # 如果当时是 P1，他的目标是最大化 (P1-P2)，所以 v = normalized_score
                    # 如果当时是 P2，他的目标是最小化 (P1-P2)，或者说最大化 (P2-P1)，所以 v = -normalized_score
                    if player == 1:
                        v = normalized_score
                    else:
                        v = -normalized_score
                        
                    return_data.append((board, pi, v))
                    
                return return_data

    def learn(self):
        """
        主训练循环
        """
        # 尝试恢复进度
        self.try_load_latest()
        
        print(f"开始训练... 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        for i in range(1, self.args.NUM_SELF_PLAY_ITERS + 1):
            print(f"\n------ Iteration {i} / {self.args.NUM_SELF_PLAY_ITERS} ------")
            start_time = time.time()
            
            # A. 自我博弈 (Self-Play)
            # ------------------------------------------------
            # 如果是刚加载完 checkpoint，可以跳过第一次 self-play (可选)，
            # 这里为了简单，我们选择总是执行。
            iteration_examples = []
            
            # 使用 tqdm 监控采集进度
            for _ in tqdm(range(self.args.NUM_EPISODES), desc="Self-Play"):
                iteration_examples += self.execute_episode()
            
            # 【优化3】将新数据推入滑动窗口
            self.train_examples_history.append(iteration_examples)
            print(f"当前历史池大小: {len(self.train_examples_history)} 组 (Window Size)")
            
            # B. 数据准备
            # ------------------------------------------------
            # 展平所有历史数据
            train_data = []
            for e in self.train_examples_history:
                train_data.extend(e)
            shuffle(train_data)
            
            print(f"训练样本总数: {len(train_data)}")
            
            # C. 备份当前网络用于评估
            # ------------------------------------------------
            self.cm.save_checkpoint(self.nnet, "temp.pth.tar")
            self.pnet.load_state_dict(self.nnet.state_dict()) 
            
            # 【关键修复】把 pnet 也搬到 GPU，否则 Arena 阶段会报错
            if torch.cuda.is_available():
                self.pnet.cuda()
            
            pmcts = MCTS(self.pnet)
            
            # D. 神经网络训练
            # ------------------------------------------------
            self.train_network(train_data)
            
            # E. 竞技场评估 (Pitting)
            # ------------------------------------------------
            print("评估中: New Net vs Old Net...")
            nmcts = MCTS(self.nnet)
            
            player_new = AlphaPlayer(self.nnet, nmcts, temp=0)
            player_old = AlphaPlayer(self.pnet, pmcts, temp=0)
            
            arena = Arena(player_new, player_old, self.game)
            
            # 减少对局数以加快迭代 (例如 14局: 各先手7次)
            pwins, nwins, draws = arena.play_games(self.args.ARENA_COMPARE_GAMES)
            
            print(f"结果 (New/Old/Draw): {pwins} / {nwins} / {draws}")
            
            # F. 模型决策与保存
            # ------------------------------------------------
            # 胜率阈值 0.55
            if pwins + nwins > 0 and float(pwins) / (pwins + nwins) < self.args.UPDATE_THRESHOLD:
                print(">>> 拒绝新模型 (REJECTED)")
                # 回滚权重
                self.nnet.load_state_dict(self.pnet.state_dict())
            else:
                print(">>> 接受新模型 (ACCEPTED)")
                # 保存为最佳模型
                self.cm.save_checkpoint(self.nnet, "best.pth.tar", is_best=True)
                
            # 【优化2】无论是否接受，都保存为 latest，用于断点续训
            self.cm.save_checkpoint(self.nnet, "checkpoint_latest.pth.tar")
            
            print(f"Iteration {i} 耗时: {time.time()-start_time:.1f}s")

    def train_network(self, examples):
        """
        神经网络训练函数
        优化点：更清晰的 Loss 监控，更高效的 Batch 处理
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.LEARNING_RATE)
        
        if torch.cuda.is_available():
            self.nnet.cuda()
        self.nnet.train()
        
        # 预先转换数据类型，减少循环内的开销 (如果内存允许)
        # 如果内存吃紧，保持在循环内转换
        
        for epoch in range(self.args.EPOCHS):
            # 随机 Batch 索引
            batch_count = int(len(examples) / self.args.BATCH_SIZE)
            
            # 使用 tqdm 的 set_postfix 实时显示 Loss
            t = tqdm(range(batch_count), desc=f"Train Epoch {epoch+1}")
            
            pi_losses = []
            v_losses = []
            
            for _ in t:
                # 手动 Batch 采样 (比 DataLoader 更轻量)
                sample_ids = np.random.randint(len(examples), size=self.args.BATCH_SIZE)
                batch = [examples[i] for i in sample_ids]
                
                boards, pis, vs = list(zip(*batch))
                
                # 转换为 Tensor 并移动到 GPU
                # astype(np.float32) 很重要，PyTorch 默认 float32
                boards = torch.tensor(np.array(boards), dtype=torch.float32)
                target_pis = torch.tensor(np.array(pis), dtype=torch.float32)
                target_vs = torch.tensor(np.array(vs), dtype=torch.float32)
                
                if torch.cuda.is_available():
                    boards = boards.cuda()
                    target_pis = target_pis.cuda()
                    target_vs = target_vs.cuda()

                # 前向传播
                out_pi, out_v = self.nnet(boards)
                
                # 计算损失
                # pi_loss: out_pi 是 log_softmax，target_pis 是概率
                # loss = - sum(target * log_pred)
                l_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
                
                # v_loss: MSE
                l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size(0)
                
                total_loss = l_pi + l_v
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # 记录
                pi_losses.append(l_pi.item())
                v_losses.append(l_v.item())
                
                # 更新进度条上的 Loss 显示
                t.set_postfix(L_pi=np.mean(pi_losses[-50:]), L_v=np.mean(v_losses[-50:]))