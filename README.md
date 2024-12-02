# 基于隐式CoT思维链的微调训练

## 项目概述

本项目旨在通过微调预训练语言模型来提高其在逻辑推理任务上的表现。使用了隐式CoT（Chain of Thought）思维链的提示词模板来引导模型进行逐步推理，并选择最合适的答案。项目中包含了训练、评估和日志记录的完整流程。

## 文件结构

- `train.py`: 主训练脚本，负责加载数据、配置模型、执行训练和评估。
- `run_train.sh`: Shell脚本，用于配置和启动训练过程。
- `ds_config.json`: DeepSpeed配置文件，用于优化训练过程。

## 使用方法

### 环境准备

1. 确保已安装必要的依赖项，包括 `torch`, `transformers`, `openmind`, `peft`, `evaluate`, `pandas`, `matplotlib` 等。
2. 激活合适的Python环境（例如通过conda）。

### 运行训练

1. 编辑 `run_train.sh` 文件，配置模型路径、数据路径、输出目录等参数。
2. 运行 `bash run_train.sh` 启动训练。

### 训练参数

- `model_path`: 预训练模型的路径。
- `train_data`: 训练数据集的路径。
- `eval_data`: 评估数据集的路径。
- `output_dir`: 输出目录。
- `num_gpus`: 使用的GPU数量。
- `batch_size`: 每个设备的训练批次大小。
- `num_epochs`: 训练的总轮数。
- `learning_rate`: 学习率。
- `use_peft`: 是否使用参数高效微调（PEFT）。

## 提示词模板

在 `train.py` 中定义了一个提示词模板 `PROMPT_DICT`，用于指导模型进行逻辑推理：

```python
PROMPT_DICT = {
    "prompt_input": (
        "<|im_start|>system\n"
        "You are a logical reasoning expert. For each question, analyze the information and choose the most logical answer.\n"
        "You must format your response as follows:\n"
        "1. Analyze the given information step by step\n"
        "2. Choose and output the answer in format: 'The answer is X' (where X is the letter of your chosen option)\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "{text}\n"
        "Question: {question}\n"
        "Choices:\n"
        "{options}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Let me solve this step by step:\n"
    ),
    "prompt_output": (
        "{answer}<|im_end|>"
    )
}
```

### 作用

- **隐式CoT思维链**: 通过逐步分析问题，引导模型进行逻辑推理，帮助模型在复杂问题上做出更准确的判断。
- **格式化输出**: 要求模型以特定格式输出答案，确保结果的可读性和一致性。

## 训练过程中的指标记录

项目中实现了 `MetricsRecorder` 类，用于记录和绘制训练过程中的各项指标，包括：

- **损失曲线**: 记录训练过程中的损失变化。
- **梯度范数**: 监控梯度的变化，帮助判断训练的稳定性。
- **学习率**: 记录学习率的变化。
- **准确率评估**: 在评估过程中计算模型的准确率，并绘制准确率曲线。
