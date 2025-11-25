# main.py
import sys
import os
import torch
import time

from environment.game import DotsAndBoxes
from model.net import DotsAndBoxesNet
from train.trainer import Trainer
from config import Config
from train.utils import CheckpointManager

def main():
    cm = CheckpointManager()
    
    print("\n=======================================")
    print("      AlphaDots 训练控制台             ")
    print("=======================================")
    print(f"当前设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("---------------------------------------")
    print("1. [推荐] 模仿学习启动 (先跟脚本学，再自我博弈)")
    print("2. [传统] 从零开始训练 (完全随机探索)")
    print("3. [继续] 加载上次进度继续训练")
    print("4. [测试] 仅生成脚本数据测试性能")
    print("=======================================")
    
    choice = input("请选择方案 (1-4): ")

    # 初始化环境
    game = DotsAndBoxes()
    nnet = DotsAndBoxesNet()
    
    if torch.cuda.is_available():
        nnet.cuda()

    trainer = Trainer(game, nnet)

    if choice == '1':
        # 方案 A: 模仿学习 -> 自我进化
        print("\n--- 阶段一: 模仿脚本 (Imitation Phase) ---")
        # 生成脚本对战数据，训练 10 个 Epoch
        # 这比 MCTS 快几十倍，因为不需要搜索
        trainer.pretrain_with_script(num_games=10000, epochs=10)
        
        print("\n--- 阶段二: 自我进化 (Self-Play Phase) ---")
        print("AI 已掌握基础策略，现在开始超越老师...")
        # 重置 iteration 计数
        trainer.learn() 

    elif choice == '2':
        # 方案 B: 纯 AlphaZero
        print("\n--- 从零开始 (Tabula Rasa) ---")
        # 清除旧的 checkpoint，防止干扰
        if os.path.exists("saved_models/checkpoint_latest.pth.tar"):
            print("警告: 检测到旧存档，将在 3 秒后覆盖...")
            time.sleep(3)
            # 这里可以选择手动删除，或者让 trainer 覆盖
            # trainer.skip_load = True (需要在 Trainer 里支持，这里简单处理)
        
        trainer.learn()

    elif choice == '3':
        # 方案 C: 继续
        print("\n--- 继续训练 ---")
        trainer.learn() # Trainer 内部会自动 try_load_latest

    elif choice == '4':
        # 测试脚本生成速度
        start = time.time()
        data = trainer.generate_heuristic_batch(100)
        print(f"生成 100 局耗时: {time.time()-start:.4f}s")
        print(f"平均每局包含 {len(data)/100:.1f} 步")

    else:
        print("无效选择")

if __name__ == "__main__":
    main()