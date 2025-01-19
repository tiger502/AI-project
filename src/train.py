import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers import BertTokenizer
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import logging

from config import Config
from dataset import MultiModalDataset
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.fusion_model import MultiModalSentimentModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model():
    # 设置随机种子
    torch.manual_seed(Config.SEED)

    # 验证路径
    logger.info(f"Training file path: {Config.TRAIN_FILE}")
    if not os.path.exists(Config.TRAIN_FILE):
        raise FileNotFoundError(f"Training file not found at {Config.TRAIN_FILE}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据
    try:
        df = pd.read_csv(Config.TRAIN_FILE)
        logger.info(f"Successfully loaded training data with {len(df)} samples")
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        try:
            df = pd.read_csv(Config.TRAIN_FILE, encoding='gbk')
            logger.info("Successfully loaded training data with GBK encoding")
        except Exception as e:
            logger.error(f"Failed to load training data with GBK encoding: {e}")
            raise

    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=Config.SEED,
        stratify=df['tag']
    )

    # 保存临时文件
    temp_train_path = os.path.join(os.path.dirname(Config.TRAIN_FILE), 'temp_train.csv')
    temp_val_path = os.path.join(os.path.dirname(Config.TRAIN_FILE), 'temp_val.csv')

    train_df.to_csv(temp_train_path, index=False)
    val_df.to_csv(temp_val_path, index=False)

    logger.info(f"Saved temporary training file to {temp_train_path}")

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL)

    # 创建数据集
    try:
        train_dataset = MultiModalDataset(
            csv_file=temp_train_path,
            data_dir=Config.DATA_DIR,
            tokenizer=tokenizer,
            transform=transform
        )

        val_dataset = MultiModalDataset(
            csv_file=temp_val_path,
            data_dir=Config.DATA_DIR,
            tokenizer=tokenizer,
            transform=transform
        )

        logger.info(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}")
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        raise

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 初始化模型
    text_encoder = TextEncoder(Config.BERT_MODEL, Config.HIDDEN_SIZE)
    image_encoder = ImageEncoder(Config.IMAGE_MODEL, Config.HIDDEN_SIZE)
    model = MultiModalSentimentModel(
        text_encoder,
        image_encoder,
        Config.HIDDEN_SIZE,
        Config.NUM_CLASSES
    )
    model = model.to(Config.DEVICE)

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_val_acc = 0
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}"):
            input_ids = batch['text_ids'].to(Config.DEVICE)
            attention_mask = batch['text_mask'].to(Config.DEVICE)
            images = batch['image'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['text_ids'].to(Config.DEVICE)
                attention_mask = batch['text_mask'].to(Config.DEVICE)
                images = batch['image'].to(Config.DEVICE)
                labels = batch['label'].to(Config.DEVICE)

                outputs = model(input_ids, attention_mask, images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        logger.info(f'Epoch: {epoch + 1}')
        logger.info(f'Train Loss: {train_loss / len(train_loader):.4f}, '
                    f'Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss / len(val_loader):.4f}, '
                    f'Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f"Saved new best model with validation accuracy: {val_acc:.2f}%")

    # 清理临时文件
    try:
        os.remove(temp_train_path)
        os.remove(temp_val_path)
    except Exception as e:
        logger.warning(f"Error removing temporary files: {e}")


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
