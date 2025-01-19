import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import logging
import chardet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    def __init__(self, csv_file, data_dir, tokenizer, transform=None, is_test=False):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            data_dir (str): Directory with all the images and text files.
            tokenizer: BERT tokenizer.
            transform (callable, optional): Optional transform to be applied on images.
            is_test (bool): Whether this is test data.
        """
        if not isinstance(csv_file, str):
            raise ValueError(f"csv_file must be a string path, got {type(csv_file)}")

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found at {csv_file}")

        try:
            self.data = pd.read_csv(csv_file)
            logger.info(f"Successfully loaded {len(self.data)} samples from {csv_file}")
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_file}: {e}")
            raise

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_test = is_test

        self.label_map = {
            'positive': 0,
            'neutral': 1,
            'negative': 2
        }

        # 验证数据文件
        self._validate_files()

    def _validate_files(self):
        """验证所需的文件是否存在"""
        missing_files = []
        for idx, row in self.data.iterrows():
            guid = str(row['guid'])
            text_path = os.path.join(self.data_dir, f"{guid}.txt")
            image_path = os.path.join(self.data_dir, f"{guid}.jpg")

            if not os.path.exists(text_path):
                missing_files.append(text_path)
            if not os.path.exists(image_path):
                missing_files.append(image_path)

        if missing_files:
            logger.warning(f"Found {len(missing_files)} missing files")
            logger.warning(f"First few missing files: {missing_files[:5]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        guid = str(int(float(self.data.iloc[idx]['guid'])))  # 将float转为int再转为字符串

        # 加载文本
        text_path = os.path.join(self.data_dir, f"{guid}.txt")
        try:
            with open(text_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'
                text = raw_data.decode(encoding).strip()
        except Exception as e:
            logger.warning(f"Error reading text file {text_path}: {e}")
            text = "[PAD]"

        # 处理文本
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        # 加载图像
        image_path = os.path.join(self.data_dir, f"{guid}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            image = torch.zeros((3, 224, 224))

        # 处理标签
        if self.is_test:
            label = -1
        else:
            label = self.label_map[self.data.iloc[idx]['tag']]

        return {
            'guid': guid,
            'text_ids': encoded['input_ids'].squeeze(),
            'text_mask': encoded['attention_mask'].squeeze(),
            'image': image,
            'label': label
        }
