import os
from pathlib import Path
import torch


class Config:
    # 路径配置
    ROOT_DIR = Path(__file__).parent.parent.absolute()
    DATA_DIR = os.path.join(ROOT_DIR, "data", "data")  # 注意这里，如果你的数据在data/data下
    TRAIN_FILE = os.path.join(ROOT_DIR, "data", "train.txt")
    TEST_FILE = os.path.join(ROOT_DIR, "data", "test_without_label.txt")

    # 模型配置
    BERT_MODEL = "bert-base-uncased"
    IMAGE_MODEL = "resnet50"
    HIDDEN_SIZE = 768
    NUM_CLASSES = 3

    # 训练配置
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    MAX_LENGTH = 128
    IMAGE_SIZE = 224

    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    @classmethod
    def validate_paths(cls):
        """验证所有必要的路径和文件"""
        required_paths = {
            'DATA_DIR': cls.DATA_DIR,
            'TRAIN_FILE': cls.TRAIN_FILE,
            'TEST_FILE': cls.TEST_FILE
        }

        for name, path in required_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at: {path}")