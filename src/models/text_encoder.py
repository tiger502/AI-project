import torch
import torch.nn as nn
from transformers import BertModel


class TextEncoder(nn.Module):
    def __init__(self, model_name, hidden_size):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        text_features = self.fc(self.dropout(pooled_output))
        return text_features
