# DotsAndBoxesAZ: AlphaZero for Dots and Boxes

这是一个基于 **AlphaZero** 算法（MCTS + 深度神经网络）实现的 **Dots and Boxes (点格棋)** 强化学习 AI。

项目包含完整的训练流水线、混合 Minimax 求解器以及一个友好的控制台对战界面。

## 游戏规则 (Game Rules)

**Dots and Boxes** (点格棋) 是一个经典的二人博弈游戏。规则看似简单，实则包含了复杂的拓扑策略。

1.  **棋盘**: 游戏在一个点阵（例如 $3 \times 3$ 或 $4 \times 4$ 的格子）上进行。
2.  **行动**: 双方轮流连接两个相邻的点（画一条横线或竖线）。
3.  **得分与连走 (Critical Rule)**:
    *   当一条线围成了一个完整的 **1x1 方格**时，该玩家占领此格（得 1 分）。
    *   **关键机制**: 占领格子的玩家获得 **额外一回合 (Extra Move)**，必须继续画线，直到无法构成新方格为止。
4.  **胜利**: 当棋盘上所有线条都被画满时游戏结束，占领格子多的一方获胜。

> **策略提示**: 高手博弈的核心在于 **"Double Cross" (双重交叉)** 策略——即故意留给对手长链的最后两个格子不吃，迫使对手在下一轮不得不打开新的长链，从而让你获得更多格子。
## 特性 (Features)

*   **AlphaZero 核心**: 使用 PyTorch 实现 ResNet + MCTS 架构。
*   **混合求解器 (Hybrid Solver)**: 在残局阶段 (<=8 步) 自动切换为 Minimax 完美求解器，解决端局计算力不足问题。
*   **奖励工程**: 引入分数差奖励 (Score Difference Reward) 解决 AI 贪婪问题。
*   **训练加速**: 支持模仿学习 (Imitation Learning) 预训练，快速启动。
*   **交互式对战**: 提供 CLI 界面，支持人类 vs AI、AI vs AI、胜率实时评估。

## 安装 (Installation)

1. 克隆仓库:
   ```bash
   git clone https://github.com/LitDarkness/DotsAndBoxesAZ
   cd AlphaDots
   ```

2. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
   *(注意: 推荐使用支持 CUDA 的 PyTorch 以获得最佳训练速度)*

## 使用方法 (Usage)

### 1. 开始对战 (Play)
直接运行以下命令即可与训练好的 AI 对战：
```bash
python play.py
```
*   支持选择不同的模型权重。
*   支持显示实时胜率仪表盘。

### 2. 训练模型 (Train)
如果你想从头训练属于你的 AI：
```bash
python main.py
```
*   在菜单中选择 **"1. 模仿学习启动"** 可以快速获得一个强力 AI。
*   训练过程会自动保存模型到 `saved_models/` 目录。

## 文件结构 (Structure)

*   `main.py`: 训练入口。
*   `play.py`: 对战/评估入口。
*   `config.py`: 全局超参数配置。
*   `model/`: 神经网络 (Net) 与 蒙特卡洛树搜索 (MCTS) 实现。
*   `train/`: 训练循环 (Trainer) 与 检查点管理。
*   `environment/`: 游戏规则逻辑与可视化。
*   `players/`: 不同类型的玩家接口 (Human, Alpha, Heuristic)。

