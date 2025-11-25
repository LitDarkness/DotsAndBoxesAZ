import sys
import os
import time
import torch
import re
import glob
import numpy as np

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from environment.game import DotsAndBoxes
from environment.visualizer import Visualizer
from model.net import DotsAndBoxesNet
from model.mcts import MCTS
from players.alpha_player import AlphaPlayer
from players.heuristic_player import HeuristicPlayer
from config import Config

# ==============================================================================
#  UI 与 辅助工具
# ==============================================================================

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_progress_bar(val, width=20):
    """
    绘制胜率条
    val: [-1, 1], -1 代表 P2(或对手) 赢, 1 代表 P1(或当前) 赢
    但是 AlphaZero 的 Value 是相对于"当前行动者"的。
    为了直观，我们将 val 转换为 "P1 的胜率"。
    """
    # 将 [-1, 1] 映射到 [0, 1]
    # win_prob 是当前玩家赢的概率
    win_prob = (val + 1) / 2
    
    filled = int(width * win_prob)
    bar = "█" * filled + "░" * (width - filled)
    
    color = Colors.GREEN if win_prob > 0.5 else Colors.RED
    return f"{color}[{bar}] {win_prob*100:.1f}%{Colors.RESET}"

def get_ai_evaluation(game, ai_player):
    """
    使用 AI 的神经网络"直觉"来评估当前局面
    不需要 MCTS，只看 Value Head，速度极快
    """
    if not hasattr(ai_player, 'nnet'):
        return None
        
    # 1. 获取 Canonical State (当前玩家视角)
    canonical = game.get_canonical_state()
    board_tensor = torch.tensor(np.ascontiguousarray(canonical), dtype=torch.float32).unsqueeze(0)
    
    if torch.cuda.is_available():
        board_tensor = board_tensor.cuda()
        
    ai_player.nnet.eval()
    with torch.no_grad():
        _, v = ai_player.nnet(board_tensor)
        
    # v 是 "当前行动玩家" 的预计收益 [-1, 1]
    return v.item()

def get_available_models(folder='saved_models'):
    if not os.path.exists(folder): return []
    files = glob.glob(os.path.join(folder, "*.pth.tar"))
    files.sort(key=os.path.getmtime)
    return [os.path.basename(f) for f in files]

def select_model_menu(prompt="请选择模型"):
    models = get_available_models()
    if not models:
        print(f"{Colors.RED}[错误] 没有找到模型文件！{Colors.RESET}")
        return None

    print(f"\n{Colors.CYAN}--- {prompt} ---{Colors.RESET}")
    # 倒序显示，最新的在最上面
    for i, f in enumerate(reversed(models)):
        print(f"{i+1}. {f}")
    
    while True:
        try:
            choice = input(f"请输入序号 (1-{len(models)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                # 因为是 reversed 显示，所以索引要反过来取
                return models[len(models) - 1 - idx]
            else:
                print("序号无效。")
        except ValueError:
            print("请输入数字。")

# ==============================================================================
#  玩家加载与逻辑
# ==============================================================================

class HumanPlayer:
    def __init__(self, name="Human"):
        self.name = name
    def get_action(self, game):
        # 逻辑上移到 UI 循环中处理
        pass

def load_ai_player(model_filename, player_name="AI"):
    print(f"正在加载 {model_filename} ...")
    nnet = DotsAndBoxesNet()
    
    filepath = os.path.join("saved_models", model_filename)
    if not os.path.exists(filepath):
        print(f"{Colors.RED}[Error]{Colors.RESET} 文件不存在，使用随机权重。")
    else:
        checkpoint = torch.load(filepath, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        if 'state_dict' in checkpoint:
            nnet.load_state_dict(checkpoint['state_dict'])
        else:
            nnet.load_state_dict(checkpoint)
            
    if torch.cuda.is_available(): nnet.cuda()
    nnet.eval()
    
    # 对战时增加搜索次数，展示最强实力
    Config.NUM_MCTS_SIMS = 800 
    mcts = MCTS(nnet)
    
    return AlphaPlayer(nnet, mcts, temp=0, name=f"{player_name}")

def robust_input(game):
    """鲁棒的输入处理"""
    valid_moves = game.get_valid_moves()
    while True:
        user_in = input(f"{Colors.YELLOW}你的操作 > {Colors.RESET}").strip().lower()
        if user_in in ['q', 'quit']: sys.exit()
            
        nums = re.findall(r'\d+', user_in)
        direction = re.search(r'[hv]', user_in)
        
        if len(nums) != 2 or not direction:
            print("格式: '行 列 方向' (例: 0 0 h)")
            continue
            
        r, c = int(nums[0]), int(nums[1])
        d = direction.group()
        
        try:
            if d == 'h': action_id = r * game.cols + c
            else: num_h = (game.rows + 1) * game.cols; action_id = num_h + r * (game.cols + 1) + c
                
            if action_id < 0 or action_id >= game.action_size:
                print("坐标越界。")
                continue
            if valid_moves[action_id] == 0:
                print("位置已占用。")
                continue
            return action_id
        except: print("输入解析错误。")

# ==============================================================================
#  游戏主循环 (The Game Loop)
# ==============================================================================

def play_match(p1, p2, game, visualizer):
    """执行一局完整的游戏"""
    game.reset()
    step = 0
    
    # 如果其中一方是 AI，我们用它来做局势分析师
    analyzer = None
    if hasattr(p1, 'nnet'): analyzer = p1
    elif hasattr(p2, 'nnet'): analyzer = p2
    
    while not game.done:
        clear_screen()
        current_p = p1 if game.current_player == 1 else p2
        
        # --- 1. 仪表盘区域 ---
        print(f"{Colors.BOLD}Dots & Boxes 竞技场{Colors.RESET}".center(50))
        print("-" * 50)
        
        # 胜率估算 (AI 直觉)
        eval_msg = "N/A"
        if analyzer:
            # 获取当前玩家视角的价值
            v = get_ai_evaluation(game, analyzer)
            # 如果 current_player 是 1，v 就是 P1 的优势
            # 如果 current_player 是 -1，v 是 P2 的优势
            # 我们统一转换成 P1 的胜率显示
            p1_advantage = v if game.current_player == 1 else -v
            eval_msg = draw_progress_bar(p1_advantage)
            
        print(f" Round: {step:02d} | 局势评估(P1胜率): {eval_msg}")
        print("-" * 50)
        print(f" 🔴 P1 (先手): {p1.name}")
        print(f" 🔵 P2 (后手): {p2.name}")
        print("-" * 50)
        
        # --- 2. 棋盘区域 ---
        last_info = ""
        if game.last_action is not None:
             last_info = f"上一步: {current_p.name} 走棋"
        visualizer.render(game, info_text=last_info)
        
        # --- 3. 状态栏 ---
        scores = game.get_game_result()
        print(f"\n当前比分: {Colors.RED}{scores[0]}{Colors.RESET} : {Colors.BLUE}{scores[1]}{Colors.RESET}")
        print(f"轮到: {Colors.BOLD}{current_p.name}{Colors.RESET}")
        
        # --- 4. 动作获取 ---
        action = -1
        if isinstance(current_p, HumanPlayer):
            action = robust_input(game)
        else:
            print("思考中...", end="", flush=True)
            time.sleep(0.5) # 稍微延迟让用户看清盘面
            action = current_p.get_action(game)
            
        # --- 5. 执行 ---
        state, reward, done, info = game.step(action)
        step += 1
        
        # 连走提示
        if info['captured'] > 0:
            print(f"\n{Colors.GREEN}>>> {current_p.name} 连得 {info['captured']} 分！继续行动！<<<{Colors.RESET}")
            time.sleep(1.0)

    # --- 结算 ---
    clear_screen()
    print(f"\n{Colors.YELLOW}" + "="*50)
    print("                GAME OVER".center(50))
    print("="*50 + f"{Colors.RESET}")
    
    visualizer.render(game)
    scores = game.get_game_result()
    
    print(f"\n最终比分: {Colors.RED}P1 {scores[0]}{Colors.RESET} - {Colors.BLUE}P2 {scores[1]}{Colors.RESET}")
    
    if scores[0] > scores[1]: print(f"🏆 获胜者: {p1.name}")
    elif scores[1] > scores[0]: print(f"🏆 获胜者: {p2.name}")
    else: print("🤝 平局！")
    
    input("\n按回车键返回...")

# ==============================================================================
#  菜单
# ==============================================================================

def main():
    game = DotsAndBoxes()
    vis = Visualizer()
    
    while True:
        clear_screen()
        print(f"{Colors.CYAN}=======================================")
        print("      AlphaDots 控制中心 v3.0         ")
        print("======================================={Colors.RESET}")
        print("1. 人类 vs AI")
        print("2. AI vs AI (内战/观战)")
        print("3. AI vs 脚本 (基准测试)")
        print("4. 人类 vs 人类")
        print("5. 退出")
        print("---------------------------------------")
        
        choice = input("请选择 (1-5): ")
        
        player_a, player_b = None, None
        
        # --- 配置玩家 ---
        if choice == '1':
            model = select_model_menu("选择 AI 模型")
            if not model: continue
            ai = load_ai_player(model, "AlphaZero")
            human = HumanPlayer("Human")
            
            # 选择先后手
            first = input("你想先手吗? (y/n): ").lower()
            if first == 'y':
                player_a, player_b = human, ai
            else:
                player_a, player_b = ai, human
                
        elif choice == '2':
            m1 = select_model_menu("选择 AI-1 (红方)")
            if not m1: continue
            m2 = select_model_menu("选择 AI-2 (蓝方)")
            if not m2: continue
            
            player_a = load_ai_player(m1, "AI-1")
            player_b = load_ai_player(m2, "AI-2")
            
        elif choice == '3':
            m1 = select_model_menu("选择 AI 模型")
            if not m1: continue
            
            ai = load_ai_player(m1, "AlphaZero")
            script = HeuristicPlayer("ScriptBot")
            
            first = input("AI 执先手? (y/n): ").lower()
            if first == 'y':
                player_a, player_b = ai, script
            else:
                player_a, player_b = script, ai
                
        elif choice == '4':
            player_a = HumanPlayer("Player 1")
            player_b = HumanPlayer("Player 2")
            
        elif choice == '5':
            sys.exit()
        else:
            continue
            
        # --- 开始对战 ---
        play_match(player_a, player_b, game, vis)

if __name__ == "__main__":
    main()