# environment/visualizer.py
import os
import platform

class Visualizer:
    def __init__(self):
        # 定义颜色代码，让显示更直观
        self.C_RESET  = "\033[0m"
        self.C_DOT    = "\033[97m" # 白色点
        self.C_EDGE   = "\033[33m" # 黄色边
        self.C_P1     = "\033[91m" # 红色 Player 1
        self.C_P2     = "\033[94m" # 蓝色 Player 2
        
    def clear_screen(self):
        """清屏，用于制作动画效果"""
        system = platform.system()
        if system == "Windows":
            os.system('cls')
        else:
            os.system('clear')

    def render(self, game, info_text=""):
        """打印棋盘当前状态"""
        rows, cols = game.rows, game.cols
        board_str = ""
        
        # 遍历每一行 (包括点行和中间的格子行)
        for r in range(rows + 1):
            # --- 1. 打印点和横边 ---
            line_top = "  " # 缩进
            for c in range(cols):
                # 点
                line_top += f"{self.C_DOT}+{self.C_RESET}"
                # 横边
                if game.horz_edges[r, c] == 1:
                    line_top += f"{self.C_EDGE}---{self.C_RESET}"
                else:
                    line_top += "   " # 空边
            # 最后一个点
            line_top += f"{self.C_DOT}+{self.C_RESET}\n"
            board_str += line_top
            
            # --- 2. 打印竖边和格子归属 (最后一行不需要) ---
            if r < rows:
                line_mid = "  "
                for c in range(cols + 1):
                    # 竖边
                    if game.vert_edges[r, c] == 1:
                        line_mid += f"{self.C_EDGE}|{self.C_RESET}"
                    else:
                        line_mid += " "
                    
                    # 格子拥有者 (位于两个竖边之间)
                    if c < cols:
                        owner = game.boxes[r, c]
                        if owner == 1:
                            line_mid += f" {self.C_P1}1{self.C_RESET} "
                        elif owner == -1:
                            line_mid += f" {self.C_P2}2{self.C_RESET} "
                        else:
                            line_mid += "   " # 空格子
                board_str += line_mid + "\n"

        print(self.C_RESET + "-" * 30)
        print(info_text)
        print(board_str)
        print(f"Scores -> {self.C_P1}P1: {game.get_game_result()[0]}{self.C_RESET} | {self.C_P2}P2: {game.get_game_result()[1]}{self.C_RESET}")
        print("-" * 30)