# train/utils.py
import os
import torch
import shutil
from datetime import datetime

class CheckpointManager:
    def __init__(self, folder='saved_models'):
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def save_checkpoint(self, model, filename, is_best=False):
        """
        保存模型，如果是最佳模型，额外备份一份
        """
        filepath = os.path.join(self.folder, filename)
        
        # 1. 保存模型权重
        torch.save({
            'state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat()
        }, filepath)
        
        # 2. 如果是最佳模型，且文件名不是 best.pth.tar，则复制一份
        if is_best:
            best_path = os.path.join(self.folder, 'best.pth.tar')
            # 【修复】只有当源文件名不等于 'best.pth.tar' 时才复制
            if filename != 'best.pth.tar':
                shutil.copyfile(filepath, best_path)
                print(f"   > Copied to {best_path}")
            else:
                print(f"   > Saved as {best_path}")

    def load_checkpoint(self, model, filename='best.pth.tar'):
        """
        安全加载模型
        """
        filepath = os.path.join(self.folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at {filepath}")
            
        checkpoint = torch.load(filepath, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 兼容只保存 state_dict 或保存了完整 dict 的情况
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print(f">>> 已加载模型: {filepath}")
        return model

    def list_checkpoints(self):
        """列出所有可用模型"""
        files = [f for f in os.listdir(self.folder) if f.endswith('.pth.tar')]
        return sorted(files)