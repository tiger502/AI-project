import random
import numpy as np
import torch
import os


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_submission_file(predictions, output_file):
    """创建提交文件"""
    with open(output_file, 'w') as f:
        f.write('guid,tag\n')
        for guid, label in predictions:
            f.write(f'{guid},{label}\n')
