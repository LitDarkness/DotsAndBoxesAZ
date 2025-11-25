# debug_bot.py
import sys
import os

# 1. 确保当前目录在 Python 搜索路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 2. 现在可以安全地导入模块了
from environment.game import DotsAndBoxes
from players.heuristic_player import HeuristicPlayer
from players.random_player import RandomPlayer

def run_debug_game():
    # 初始化游戏
    game = DotsAndBoxes()
    
    # 初始化玩家
    # P1: 规则机器人 (Heuristic)
    # P2: 随机机器人 (Random)
    bot1 = HeuristicPlayer("SmartBot")
    bot2 = RandomPlayer("RandomBot")
    
    print(f"--- 开始对局: {bot1.name} (P1) vs {bot2.name} (P2) ---")
    
    step = 0
    while not game.done:
        step += 1
        
        # 决定当前是谁的回合
        # game.current_player 是 1 或 -1
        current_bot = bot1 if game.current_player == 1 else bot2
        
        # 获取动作
        action = current_bot.get_action(game)
        
        # 为了调试打印好看，转换一下坐标
        # 注意：这里我们调用 game 内部的方法辅助打印，实际训练不需要
        try:
            etype, r, c = game._action_to_coord(action)
            coord_str = f"{etype}({r},{c})"
        except:
            coord_str = str(action)
            
        # 执行动作
        state, reward, done, info = game.step(action)
        
        captured = info['captured']
        symbol = "+" if captured > 0 else "-"
        print(f"Step {step:02d} | P{1 if game.current_player==1 else 2} 动作: {coord_str} | {symbol} 连走: {'是' if captured>0 else '否'}")

    # 游戏结束
    scores = game.get_game_result()
    print("\n==========================")
    print(f"游戏结束!")
    print(f"P1 ({bot1.name}): {scores[0]} 格")
    print(f"P2 ({bot2.name}): {scores[1]} 格")
    
    if scores[0] > scores[1]:
        print(">>> 胜利者: P1 <<<")
    elif scores[1] > scores[0]:
        print(">>> 胜利者: P2 <<<")
    else:
        print(">>> 平局 <<<")

if __name__ == "__main__":
    run_debug_game()