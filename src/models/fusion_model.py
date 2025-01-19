import torch
import torch.nn as nn


class MultiModalSentimentModel(nn.Module):
    def __init__(self, text_encoder, image_encoder, hidden_size, num_classes):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, images):
        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.image_encoder(images)

        # 特征融合
        combined_features = torch.cat([text_features, image_features], dim=1)
        fused_features = self.fusion(combined_features)

        # 分类
        logits = self.classifier(fused_features)
        return logits
