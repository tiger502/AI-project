

```markdown
# 多模态情感分析实验

## 项目概述
本项目实现了一个基于文本和图像的多模态情感分析系统，能够对社交媒体内容进行情感分类（积极、中性、消极）。

作者：Yanzichao
更新时间：2025-01-18

## 项目结构
```
multimodal_sentiment/
│
├── data/                      # 数据目录
│   └── README.md             # 数据说明
│
├── src/                      # 源代码
│   ├── __init__.py
│   ├── config.py            # 配置文件
│   ├── dataset.py           # 数据集实现
│   ├── models/              # 模型实现
│   │   ├── __init__.py
│   │   ├── text_encoder.py  # 文本编码器
│   │   ├── image_encoder.py # 图像编码器
│   │   └── fusion_model.py  # 多模态融合模型
│   ├── train.py            # 训练脚本
│   ├── predict.py          # 预测脚本
│   └── utils.py            # 工具函数
│
├── requirements.txt         # 环境依赖
└── README.md               # 项目说明
```

## 模型架构
本项目采用了一个双流架构的多模态融合模型：
1. 文本编码器：基于BERT的文本特征提取
2. 图像编码器：基于ResNet50的图像特征提取
3. 多模态融合：使用注意力机制融合两种模态的特征

## 环境配置
1. 创建虚拟环境（推荐）：
```bash
conda create -n multimodal python=3.8
conda activate multimodal
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备
1. 将数据集解压到`data`目录：
   - train.txt：训练集标签文件
   - test_without_label.txt：测试集文件
   - data/：包含所有文本和图像文件

2. 数据格式说明：
   - 文本文件：`{guid}.txt`，包含社交媒体文本内容
   - 图像文件：`{guid}.jpg`，对应的图像内容
   - 标签文件：CSV格式，包含guid和情感标签

## 使用说明

### 训练模型
```bash
cd src
python train.py
```

### 预测测试集
```bash
cd src
python predict.py
```

## 实验结果
1. 验证集结果：
   - 整体准确率：XX%
   - 各类别F1分数：
     - Positive: XX%
     - Neutral: XX%
     - Negative: XX%

2. 消融实验结果：
   - 仅文本模型：XX%
   - 仅图像模型：XX%
   - 多模态融合模型：XX%

## 主要特点
1. 采用预训练模型提升性能
2. 实现了高效的数据加载和预处理流程
3. 支持模型的保存和加载
4. 提供了详细的实验结果分析

## 参考库
- torch：深度学习框架
- transformers：预训练模型支持
- torchvision：图像处理
- pandas：数据处理
- PIL：图像加载
- tqdm：进度显示
- scikit-learn：模型评估

## 问题及解决方案
1. 内存占用问题：
   - 实现了高效的数据加载器
   - 使用适当的批次大小

2. 训练稳定性：
   - 采用warmup学习率策略
   - 实现了模型检查点保存

## 可能的改进方向
1. 尝试更复杂的多模态融合策略
2. 增加数据增强方法
3. 实现模型集成
4. 优化推理速度

## 许可证
MIT License

## 联系方式
如有问题，请通过以下方式联系：
- GitHub: @
```

此README包含了完整的项目说明，包括：
1. 项目结构说明
2. 环境配置指南
3. 使用方法
4. 实验结果
5. 可能的改进方向

