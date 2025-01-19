import torch
from torch.utils.data import DataLoader
import pandas as pd
from transformers import BertTokenizer
from torchvision import transforms

from config import Config
from dataset import MultiModalDataset
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.fusion_model import MultiModalSentimentModel


def predict():
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL)

    # 加载测试数据
    test_dataset = MultiModalDataset(
        Config.TEST_FILE,
        Config.DATA_DIR,
        tokenizer,
        transform,
        is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
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

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    model = model.to(Config.DEVICE)
    model.eval()

    # 预测
    predictions = []
    guids = []
    label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['text_ids'].to(Config.DEVICE)
            attention_mask = batch['text_mask'].to(Config.DEVICE)
            images = batch['image'].to(Config.DEVICE)
            guid = batch['guid']

            outputs = model(input_ids, attention_mask, images)
            _, predicted = outputs.max(1)

            predictions.extend([label_map[p.item()] for p in predicted])
            guids.extend(guid)

    # 保存预测结果
    results_df = pd.DataFrame({
        'guid': guids,
        'tag': predictions
    })
    results_df.to_csv('test_predictions.txt', index=False)


if __name__ == "__main__":
    predict()
