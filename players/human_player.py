# players/human_player.py
from .base_player import BasePlayer
import sys

class HumanPlayer(BasePlayer):
    def get_action(self, game):
        # 打印提示
        print("\n请输入动作 (格式: 行 列 方向)")
        print("例如: '0 0 h' (左上横边) 或 '1 2 v' (第1行第2列竖边)")
        print("输入 'q' 退出游戏")
        
        valid_moves = game.get_valid_moves()
        
        while True:
            user_input = input(f"P{1 if game.current_player==1 else 2} > ")
            
            if user_input.lower() == 'q':
                sys.exit()
                
            try:
                # 解析输入
                parts = user_input.strip().split()
                if len(parts) != 3:
                    print("格式错误。请使用: 行 列 方向 (e.g., 0 0 h)")
                    continue
                    
                r, c, d = int(parts[0]), int(parts[1]), parts[2].lower()
                
                # 转换坐标到 Action ID
                # 我们需要反向查找，或者使用 game 内部的逻辑
                # 这里手动算一下 Action ID
                if d == 'h': # 横边
                    # ID范围: 0 ~ (rows+1)*cols - 1
                    # index = r * cols + c
                    action_id = r * game.cols + c
                    if action_id >= game.action_size: # 简单越界检查
                         print("坐标越界")
                         continue
                elif d == 'v': # 竖边
                    # ID范围: num_h ~ total - 1
                    # 偏移量 = num_horizontal_edges
                    # index = offset + r * (cols+1) + c
                    num_h = (game.rows + 1) * game.cols
                    action_id = num_h + r * (game.cols + 1) + c
                else:
                    print("方向错误，必须是 h 或 v")
                    continue
                
                # 检查合法性
                if action_id >= game.action_size or valid_moves[action_id] == 0:
                    print("非法动作！该边已被占用或不存在。")
                    continue
                
                return action_id
                
            except ValueError:
                print("输入无效，请输入数字坐标。")
            except Exception as e:
                print(f"发生错误: {e}")