# debug_bot.py
import sys
import os
import time

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from environment.game import DotsAndBoxes
from environment.visualizer import Visualizer  # <--- 导入新写的可视化器
from players.heuristic_player import HeuristicPlayer
from players.random_player import RandomPlayer

def run_debug_game():
    game = DotsAndBoxes()
    vis = Visualizer() # <--- 实例化
    
    # 你可以修改这里，让两个 HeuristicPlayer 互殴，看看谁厉害
    bot1 = HeuristicPlayer("RedBot")
    bot2 = HeuristicPlayer("BlueBot") 
    
    print("准备开始...")
    time.sleep(1)
    
    step = 0
    while not game.done:
        step += 1
        
        # 1. 渲染当前帧
        vis.clear_screen()
        current_p_name = bot1.name if game.current_player == 1 else bot2.name
        info = f"Step {step} | Turn: {current_p_name} (P{1 if game.current_player==1 else 2})"
        vis.render(game, info)
        
        # 2. 思考 (加点延迟方便看清)
        time.sleep(0.5) 
        
        # 3. 决策与执行
        current_bot = bot1 if game.current_player == 1 else bot2
        action = current_bot.get_action(game)
        
        state, reward, done, info = game.step(action)
        
        # 如果连走，这里会循环回来，画面会刷新
        if info['captured'] > 0:
            print(f"*** {current_bot.name} 占领了格子! 连走! ***")
            time.sleep(0.5)

    # 游戏结束，最后再显示一次
    vis.clear_screen()
    vis.render(game, "GAME OVER")
    
    scores = game.get_game_result()
    winner = "Draw"
    if scores[0] > scores[1]: winner = bot1.name
    elif scores[1] > scores[0]: winner = bot2.name
    
    print(f"\n最终胜者: {winner}!")

if __name__ == "__main__":
    run_debug_game()