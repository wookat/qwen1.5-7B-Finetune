#    Copyright (c) Huawei Technologies Co., Ltd. 2024-2024, All rights reserved.
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Any

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
import openmind
from peft import LoraConfig, AdaLoraConfig, IA3Config, get_peft_model, TaskType, PeftModel # type: ignore
import numpy as np
import evaluate
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime
from transformers.trainer_utils import EvalPrediction
import re

# 一些常量
IGNORE_INDEX = -100  # 用于标记不参与损失计算的标签位置
# 定义模型词表中的特殊token
DEFAULT_PAD_TOKEN = "<|endoftext|>"  # 填充token
DEFAULT_BOS_TOKEN = "<|im_start|>"    # 句子开始token 
DEFAULT_EOS_TOKEN = "<|im_end|>"   # 句子结束token
DEFAULT_UNK_TOKEN = "<|extra_0|>"  # 未知词token

# 定义提示模板格式
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
        "{answer}<|im_end|>"  # 只输出选项字母
    )
}




# 配置日志格式
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 创建一个自定义的日志记录器
def log_info(message: str, level: str = "warning"):
    """记录带有时间戳的日志信息
    
    Args:
        message: 要记录的消息
        level: 日志级别，可选值："debug", "info", "warning", "error", "critical"
    """
    if level == "debug":
        logging.debug(message)
    elif level == "info":
        logging.info(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "error":
        logging.error(message)
    elif level == "critical":
        logging.critical(message)


@dataclass
class ModelArguments:
    """模型参数配置类"""
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "预训练模型的路径或名称"})
    # 新增低参微调相关配置
    use_peft: bool = field(default=False, metadata={"help": "是否使用参数高效微调(PEFT)"})
    peft_method: str = field(
        default="lora",
        metadata={
            "help": "PEFT方法选择: 'lora', 'adalora', 或 'ia3'",
            "choices": ["lora", "adalora", "ia3"]
        }
    )
    # LoRA相关参数
    lora_r: int = field(default=16, metadata={"help": "LoRA秩"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    # AdaLoRA相关参数
    adalora_init_r: int = field(default=12, metadata={"help": "AdaLoRA初始秩"})
    adalora_tinit: int = field(default=200, metadata={"help": "AdaLoRA tinit"})
    adalora_tfinal: int = field(default=1000, metadata={"help": "AdaLoRA tfinal"})
    adalora_delta_t: int = field(default=10, metadata={"help": "AdaLoRA deltaT"})


@dataclass
class DataArguments:
    """数据参数配置类"""
    train_data_path: str = field(default=None, metadata={"help": "训练数据的路径"})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "评估数据的路径"})
    eval_num_samples: int = field(
        default=0,
        metadata={
            "help": "评估使用的样本数量。0表示使用全部样本，否则使用指定数量的样本。"
                   "实际使用的样本数会被调整为8的倍数以便于分布式评估。"
        }
    )


@dataclass
class TrainingArguments(openmind.TrainingArguments):
    """训练参数配置类, 继承自openmind的TrainingArguments"""
    cache_dir: Optional[str] = field(default=None, metadata={"help": "缓存目录路径"})
    optim: str = field(default="adamw_torch", metadata={"help": "使用的优化器类型"})
    model_max_length: int = field(
        default=512, 
        metadata={"help": "最大序列长度，超过此长度的序列将被截断"}
    )
    evaluation_strategy: str = field(
        default="no",
        metadata={"help": "评估策略: no, steps, epoch", "choices": ["no", "steps", "epoch"]}
    )
    eval_steps: int = field(
        default=None,
        metadata={"help": "每隔多少步进行评估, 仅在evaluation_strategy='steps'时有效"}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "保存策略: no, steps, 或 epoch", "choices": ["no", "steps", "epoch"]}
    )
    metric_for_best_model: str = field(
        default="accuracy",
        metadata={"help": "用于确定最佳模型的指标"}
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "训练结束时是加载最佳模型"}
    )



def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        f'(?i)ANSWER\s*:\s*([{options}])',
        f'[Tt]he answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        f'(\s|^)[{options}][\s,:\.$]',
        f'1.\s?(.*?)$',
        f'1.\s?([{options}])[.$]?$',
    ]
    cushion_patterns = [
        f'([{options}]):',
        f'([{options}])',
    ]
    # flake8: noqa
    # yapf: enable

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        text = text.strip()
        match = re.search(pattern, text, re.DOTALL)
        if match:
            outputs = match.group(1)
            for i in options:
                if i in outputs:
                    return i
    return ''


def create_eval_examples(model, tokenizer, eval_data) -> List[Dict[str, Any]]:
    """创建评估样本"""
    eval_examples = []
    
    for sample in eval_data:
        # 构建对话格式的提示
        messages = [{
            "role": "user",
            "content": (
                f"{sample['text']}\n"
                f"Question: {sample['question']}\n"
                + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(sample['options'])])
                + "\nAnswer: "
            )
        }]
        
        # 使用chat template处理
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True
        )
        
        eval_examples.append({
            "input_ids": inputs,
            "label": sample["label"],
            "text": sample["text"],
            "question": sample["question"],
            "options": sample["options"]
        })
    
    return eval_examples


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer,
    model,
):
    """调整tokenizer和模型embedding层的大小以适应新增的特殊token
    
    Args:
        special_tokens_dict: 包含要添加的特殊token的字典
        tokenizer: 分词器对象
        model: 模型对象
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer) -> Dict:
    """将文本序列转换为token id
    
    Args:
        strings: 要分词的文本列表
        tokenizer: 分词器对象
    
    Returns:
        包含input_ids等信息的字典
    """
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=True,  # 使用tokenizer的特殊token处理
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def jload(f, mode="r"):
    """加载数据文件为Python对象
    
    Args:
        f: 文件路径或文件对象
        mode: 打开文件的模式
    
    Returns:
        加载的数据列表
    """
    if isinstance(f, io.IOBase):
        return json.load(f)
        
    # 根据文件扩展名选择不同的读取方式
    if f.endswith('.parquet'):
        df = pd.read_parquet(f)
        # 确保将numpy array转换为list
        records = df.to_dict('records')
        for record in records:
            if 'options' in record and hasattr(record['options'], 'tolist'):
                record['options'] = record['options'].tolist()
        return records
    elif f.endswith(('.json', '.jsonl', '.txt')):
        with open(f, mode=mode, encoding='utf-8') as file:
            return [json.loads(line) for line in file]
    else:
        raise ValueError(f"不支持的文件格式: {f}")


def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer) -> Dict:
    """预处理数据"""
    # 分别对输入和目标进行分词
    examples_tokenized = _tokenize_fn([s + t for s, t in zip(sources, targets)], tokenizer)
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    
    input_ids = examples_tokenized["input_ids"]
    # 创建labels的深拷贝，而不是简单的引用
    labels = copy.deepcopy(input_ids)
    
    # 确保标签数据的有效性
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        # 将source部分的label设置为IGNORE_INDEX
        label[:source_len] = IGNORE_INDEX
        # 将所有可能的None值替换为IGNORE_INDEX
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        label[label.isnan()] = IGNORE_INDEX  # 处理NaN值
    
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """用于监督微调的数据集类"""

    def __init__(self, data_path: str, tokenizer):
        super(SupervisedDataset, self).__init__()
        log_info("正在加载数据...")
        
        # 读取数据集
        list_data_dict = []
        data_list = jload(data_path)
        
        for data in data_list:
            if not all(k in data for k in ['text', 'question', 'options', 'label']):
                continue
                
            # 构建输入
            source = PROMPT_DICT["prompt_input"].format(
                text=data['text'],
                question=data['question'],
                options="\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(data['options'])])
            )
            
            # 构建输出
            target = PROMPT_DICT["prompt_output"].format(
                answer=chr(65 + data['label'])
            )
            
            list_data_dict.append({
                "input": source,
                "output": target
            })

        log_info(f"成功处理 {len(list_data_dict)} 条数据")

        log_info("开始格式化输入...")
        sources = [x["input"] for x in list_data_dict]
        targets = [x["output"] for x in list_data_dict]

        # 进行分词，设置add_special_tokens=False因为提示词模板中已包含特殊token
        log_info("开始分词...可能需要一些时间...")
        data_dict = preprocess(sources, targets, tokenizer)

        try:
            self.input_ids = data_dict["input_ids"]
        except KeyError as e:
            raise KeyError("input_ids is invalid") from e
        try:
            self.labels = data_dict["labels"]
        except KeyError as e:
            raise KeyError("labels is invalid") from e

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """用于监督微调的数据整理器"""
    tokenizer: object

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer, data_args) -> Dict:
    """创建用于监督微调的数据模块"""
    log_info("正在创建训练数据集...")
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, 
        data_path=data_args.train_data_path
    )
    
    log_info("正在创建数据整理器...")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # 如果提供了评估数据路径，则创建评估数据集
    eval_dataset = None
    if data_args.eval_data_path is not None:
        log_info("正在创建评估数据集...")
        eval_dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.eval_data_path
        )
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )


class MetricsRecorder:
    """记录训练过程中的各项指标"""
    def __init__(self, output_dir: str):
        self.metrics = {
            'loss': {'steps': [], 'values': []},
            'grad_norm': {'steps': [], 'values': []},
            'learning_rate': {'steps': [], 'values': []},
            'epoch': {'steps': [], 'values': []}
        }
        # 单独记录accuracy
        self.accuracies = []  # 每个元素将包含 {step, epoch, accuracy, timestamp}
        
        # 使用output_dir的父路径作为指标保存路径
        self.metrics_dir = os.path.dirname(output_dir.rstrip('/'))
        # 创建metrics子目录用于存放指标相关文件
        self.metrics_dir = os.path.join(self.metrics_dir, 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        
    def add_metric(self, step: int, metric_name: str, value: float, epoch: float = None):
        """添加一条指标记录"""
        if metric_name == 'accuracy':
            # 记录准确率，添加epoch信息
            accuracy_record = {
                'step': step,
                'accuracy': value,
                'epoch': epoch,  # 添加epoch信息
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.accuracies.append(accuracy_record)
            # 按准确率降序排序
            self.accuracies.sort(key=lambda x: x['accuracy'], reverse=True)
            
            # 保存到json文件
            accuracy_path = os.path.join(self.metrics_dir, 'accuracy.json')
            with open(accuracy_path, 'w', encoding='utf-8') as f:
                json.dump(self.accuracies, f, indent=2, ensure_ascii=False)
            
            # 绘制准确率曲线
            self.plot_accuracy_curve()
        else:
            if metric_name not in self.metrics:
                raise ValueError(f"Unknown metric: {metric_name}")
            self.metrics[metric_name]['steps'].append(step)
            self.metrics[metric_name]['values'].append(value)
        
    def plot_accuracy_curve(self):
        """绘制准确率曲线，带有双横轴（step和epoch）"""
        if not self.accuracies:
            return
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 提取数据并按step排序
        sorted_records = sorted(self.accuracies, key=lambda x: x['step'])
        steps = [record['step'] for record in sorted_records]
        accuracies = [record['accuracy'] for record in sorted_records]
        epochs = [record['epoch'] for record in sorted_records]
        
        # 添加水平网格线
        ax1.grid(True, axis='y', alpha=0.2, zorder=0)
        
        # 绘制主曲线（使用steps作为x轴）
        # 使用drawstyle='default'确保线段是直线连接
        line = ax1.plot(steps, accuracies, color='purple', marker='o', 
                       drawstyle='default', linestyle='-',
                       label='Accuracy', zorder=2)
        
        # 设置y轴范围从0到1，给最大值留一些边距
        max_accuracy = max(accuracies)
        ax1.set_ylim(0, min(1.0, max_accuracy + 0.1))
        
        # 设置主坐标轴标签
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', labelcolor='purple')
        
        # 创建次坐标轴（epoch）
        if any(epoch is not None for epoch in epochs):
            ax2 = ax1.twiny()
            ax2.set_xlim(ax1.get_xlim())
            
            # 选择合适的epoch刻度点
            unique_epochs = sorted(set(epochs))
            if len(unique_epochs) > 1:
                # 计算合适的刻度间隔
                epoch_interval = (unique_epochs[-1] - unique_epochs[0]) / 5
                epoch_ticks = []
                step_ticks = []
                
                # 创建刻度映射
                step_to_epoch = {step: epoch for step, epoch in zip(steps, epochs)}
                
                current_epoch = unique_epochs[0]
                while current_epoch <= unique_epochs[-1]:
                    # 找到最接近当前epoch的step
                    closest_step = min(steps, 
                                    key=lambda x: abs(step_to_epoch[x] - current_epoch))
                    epoch_ticks.append(current_epoch)
                    step_ticks.append(closest_step)
                    current_epoch += epoch_interval
                
                # 设置次坐标轴的刻度
                ax2.set_xticks(step_ticks)
                ax2.set_xticklabels([f'{e:.2f}' for e in epoch_ticks])
            
            ax2.set_xlabel('Epochs')
            ax2.tick_params(axis='x', labelcolor='gray')
        
        # 添加图例
        ax1.legend(loc='lower right')
        
        # 设置标题
        plt.title('Training Accuracy Curve')
        
        # 保存图表
        plt.savefig(os.path.join(self.metrics_dir, 'accuracy_curve.png'))
        plt.close()

    def plot_single_metric(self, metric_name: str, color: str):
        """绘制单个指标的曲线图，带有双横轴和0刻度线"""
        if metric_name == 'epoch':  # 跳过epoch的单独绘图
            return
        
        if len(self.metrics[metric_name]['steps']) == 0:
            return
            
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 添加水平0刻度线
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3, zorder=1)
        
        # 主坐标轴（steps）
        ax1.plot(
            self.metrics[metric_name]['steps'],
            self.metrics[metric_name]['values'],
            color=color,
            label=metric_name,
            zorder=2  # 确保曲线在0刻度线之上
        )
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel(metric_name)
        ax1.tick_params(axis='x', labelcolor=color)
        
        # 添加网格线
        ax1.grid(True, alpha=0.2, zorder=0)  # 确保网格在最底层
        
        # 创建次坐标轴（epoch）
        ax2 = ax1.twiny()
        
        # 获取对应的epoch值
        epoch_values = self.metrics['epoch']['values']
        step_values = self.metrics['epoch']['steps']
        
        if len(epoch_values) > 0:
            # 设置次坐标轴的刻度
            ax2.set_xlim(ax1.get_xlim())
            # 将steps映射到对应的epoch值
            step_to_epoch = {step: epoch for step, epoch in zip(step_values, epoch_values)}
            
            # 选择合适的刻度点
            unique_epochs = sorted(set(epoch_values))
            if len(unique_epochs) > 1:
                epoch_interval = (unique_epochs[-1] - unique_epochs[0]) / 10
                epoch_ticks = []
                step_ticks = []
                
                current_epoch = unique_epochs[0]
                while current_epoch <= unique_epochs[-1]:
                    closest_step = min(step_values, 
                                    key=lambda x: abs(step_to_epoch[x] - current_epoch))
                    epoch_ticks.append(current_epoch)
                    step_ticks.append(closest_step)
                    current_epoch += epoch_interval
                
                ax2.set_xticks(step_ticks)
                ax2.set_xticklabels([f'{e:.1f}' for e in epoch_ticks])
            
        ax2.set_xlabel('Epochs')
        ax2.tick_params(axis='x', labelcolor='gray')
        
        plt.title(f'Training {metric_name.replace("_", " ").title()} Curve')
        plt.savefig(os.path.join(self.metrics_dir, f'{metric_name}_curve.png'))
        plt.close()

    def plot_all_metrics(self):
        """绘制所有指标的组合图"""
        metric_colors = {
            'loss': 'blue',
            'grad_norm': 'red',
            'learning_rate': 'green'
        }
        
        log_info("绘制所有指标的组合图")

        for metric_name, color in metric_colors.items():
            self.plot_single_metric(metric_name, color)
            
        # 绘制2x2子图组合
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        fig.suptitle('Training Metrics', fontsize=16)
        
        # 绘制loss、grad_norm和learning_rate的单独图表
        plot_params = {
            'loss': {'color': 'blue', 'title': 'Training Loss', 'position': (0, 0)},
            'grad_norm': {'color': 'red', 'title': 'Gradient Norm', 'position': (0, 1)},
            'learning_rate': {'color': 'green', 'title': 'Learning Rate', 'position': (1, 0)}
        }
        
        # 绘制每个指标的曲线
        for metric_name, params in plot_params.items():
            row, col = params['position']
            ax1 = fig.add_subplot(gs[row, col])
            
            if len(self.metrics[metric_name]['steps']) > 0:
                # 添加水平0刻度线
                ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3, zorder=1)
                
                # 添加网格线
                ax1.grid(True, alpha=0.2, zorder=0)
                
                # 主坐标轴（steps）
                ax1.plot(
                    self.metrics[metric_name]['steps'],
                    self.metrics[metric_name]['values'],
                    color=params['color'],
                    label=metric_name,
                    zorder=2
                )
                ax1.set_xlabel('Training Steps')
                ax1.set_ylabel(metric_name)
                ax1.tick_params(axis='x', labelcolor=params['color'])
                ax1.legend()
                
                # 添加epoch次坐标轴
                ax2 = ax1.twiny()
                epoch_values = self.metrics['epoch']['values']
                step_values = self.metrics['epoch']['steps']
                
                if len(epoch_values) > 0:
                    ax2.set_xlim(ax1.get_xlim())
                    step_to_epoch = {step: epoch for step, epoch in zip(step_values, epoch_values)}
                    
                    unique_epochs = sorted(set(epoch_values))
                    if len(unique_epochs) > 1:
                        epoch_interval = (unique_epochs[-1] - unique_epochs[0]) / 5
                        epoch_ticks = []
                        step_ticks = []
                        
                        current_epoch = unique_epochs[0]
                        while current_epoch <= unique_epochs[-1]:
                            closest_step = min(step_values, 
                                            key=lambda x: abs(step_to_epoch[x] - current_epoch))
                            epoch_ticks.append(current_epoch)
                            step_ticks.append(closest_step)
                            current_epoch += epoch_interval
                        
                        ax2.set_xticks(step_ticks)
                        ax2.set_xticklabels([f'{e:.1f}' for e in epoch_ticks])
                    
                    ax2.set_xlabel('Epochs')
                    ax2.tick_params(axis='x', labelcolor='gray')
                
                ax1.set_title(params['title'])
        
        # 绘制最近20步的loss和grad_norm组合图（双纵轴）
        ax_combined = fig.add_subplot(gs[1, 1])
        
        # 获取最近20步的数据
        recent_steps = self.metrics['loss']['steps'][-20:]
        recent_loss = self.metrics['loss']['values'][-20:]
        recent_grad_norm = self.metrics['grad_norm']['values'][-20:]
        
        # 检查是否有足够的数据
        if not recent_steps or not recent_loss or not recent_grad_norm:
            # 如果没有数据，显示提示信息
            ax_combined.text(0.5, 0.5, 'No data available for recent steps',
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=ax_combined.transAxes)
            ax_combined.set_title('Recent Loss and Grad Norm (last 20 steps)')
        else:
            # 添加水平0刻度线
            ax_combined.axhline(y=0, color='gray', linestyle='--', alpha=0.3, zorder=1)
            
            # 添加网格线
            ax_combined.grid(True, alpha=0.2, zorder=0)  # 确保网格在最底层
            
            # 绘制loss曲线（使用左侧y轴）
            line1 = ax_combined.plot(recent_steps, recent_loss, color='blue', label='Loss', zorder=2)
            ax_combined.set_xlabel('Training Steps')
            ax_combined.set_ylabel('Loss', color='blue')
            ax_combined.tick_params(axis='y', labelcolor='blue')
            
            # 创建右侧y轴并绘制grad_norm曲线
            ax_combined_right = ax_combined.twinx()
            line2 = ax_combined_right.plot(recent_steps, recent_grad_norm, color='red', label='Grad Norm', zorder=2)
            ax_combined_right.set_ylabel('Gradient Norm', color='red')
            ax_combined_right.tick_params(axis='y', labelcolor='red')
            
            # 设置y轴范围（从0到各自的最大值）
            loss_max = max(recent_loss)
            grad_max = max(recent_grad_norm)
            
            # 为最大值添加10%的边距
            loss_margin = 0.1 * loss_max
            grad_margin = 0.1 * grad_max
            
            # 设置y轴范围：从0到最大值加边距
            ax_combined.set_ylim(0, loss_max + loss_margin)
            ax_combined_right.set_ylim(0, grad_max + grad_margin)
            
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax_combined.legend(lines, labels, loc='upper right')
            
            ax_combined.set_title('Recent Loss and Grad Norm (last 20 steps)')
            
            # 添加epoch次坐标轴（顶部x轴）
            ax_combined_top = ax_combined.twiny()
            recent_epochs = [step_to_epoch.get(step, 0) for step in recent_steps]
            ax_combined_top.set_xlim(ax_combined.get_xlim())
            
            # 设置epoch刻度
            if recent_epochs:
                epoch_min, epoch_max = min(recent_epochs), max(recent_epochs)
                epoch_interval = (epoch_max - epoch_min) / 4  # 使用4个刻度点
                epoch_ticks = []
                step_ticks = []
                
                for i in range(5):  # 5个点（包括起点和终点）
                    target_epoch = epoch_min + i * epoch_interval
                    # 找到最接近的step
                    closest_step_idx = min(range(len(recent_steps)), 
                                        key=lambda x: abs(recent_epochs[x] - target_epoch))
                    epoch_ticks.append(recent_epochs[closest_step_idx])
                    step_ticks.append(recent_steps[closest_step_idx])
                
                ax_combined_top.set_xticks(step_ticks)
                ax_combined_top.set_xticklabels([f'{e:.2f}' for e in epoch_ticks])
            
            ax_combined_top.set_xlabel('Epochs')
            ax_combined_top.tick_params(axis='x', labelcolor='gray')
        
        # 保存组合图
        plt.savefig(os.path.join(self.metrics_dir, 'all_metrics.png'))
        plt.close()
        
        # 保存指标数据到JSON文件
        metrics_data = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {}
        }
        
        all_steps = sorted(list(set(
            step for metric in self.metrics.values() for step in metric['steps']
        )))
        
        for step in all_steps:
            metrics_data['metrics'][str(step)] = {
                metric_name: self.metrics[metric_name]['values'][
                    self.metrics[metric_name]['steps'].index(step)
                ] if step in self.metrics[metric_name]['steps'] else None
                for metric_name in self.metrics
            }
        
        with open(os.path.join(self.metrics_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics_data, f, indent=2)


class PeftTrainer(openmind.Trainer):
    """扩展Trainer以支持指标记录"""
    
    # 在类初始化时就检查XLA可用性
    try:
        from transformers.utils import is_torch_xla_available
        if is_torch_xla_available():
            import torch_xla.core.xla_model as xm
            XLA_AVAILABLE = True
        else:
            XLA_AVAILABLE = False
    except ImportError:
        XLA_AVAILABLE = False
    
    def __init__(self, *args, eval_examples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_recorder = MetricsRecorder(self.args.output_dir)
        self.eval_examples = eval_examples

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            # 使用类级别的XLA检查结果
            if self.XLA_AVAILABLE:
                self.xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            original_loss = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)

            logs["loss"] = round(original_loss, 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            # 记录指标到metrics_recorder
            if self.is_world_process_zero():
                self.metrics_recorder.add_metric(self.state.global_step, 'loss', original_loss)
                if "grad_norm" in logs:
                    self.metrics_recorder.add_metric(self.state.global_step, 'grad_norm', logs["grad_norm"])
                self.metrics_recorder.add_metric(self.state.global_step, 'learning_rate', logs["learning_rate"])
                self.metrics_recorder.add_metric(self.state.global_step, 'epoch', self.state.epoch)
                
                # 实时更新图表
                self.metrics_recorder.plot_all_metrics()
                
                # 打印当前指标
                log_info(
                    f"Step {self.state.global_step}: "
                    f"loss={logs['loss']:.8f}, "
                    f"grad_norm={logs.get('grad_norm', 0.0):.8f}, "
                    f"lr={logs['learning_rate']:.2e}, "
                    f"epoch={self.state.epoch:.2f}"
                )
            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """重写评估方法，支持多NPU并行评估"""
        if self.eval_examples:
            # 获取当前进程的rank和world_size
            local_rank = self.args.local_rank
            world_size = torch.distributed.get_world_size()
            
            log_info(f"进程 {local_rank}/{world_size} 开始评估...")
            total_samples = len(self.eval_examples)
            
            # 计算每个进程应处理的样本范围
            samples_per_process = (total_samples + world_size - 1) // world_size
            start_idx = local_rank * samples_per_process
            end_idx = min(start_idx + samples_per_process, total_samples)
            
            # 获取当前进程负责的样本
            process_examples = self.eval_examples[start_idx:end_idx]
            log_info(f"进程 {local_rank} 负责评估 {len(process_examples)} 个样本 (范围: {start_idx}-{end_idx-1})")
            
            # 创建批次
            batch_size = self.args.per_device_eval_batch_size
            num_batches = (len(process_examples) + batch_size - 1) // batch_size
            
            # 清理GPU缓存
            if torch.npu.is_available():
                torch.npu.empty_cache()
            
            total_correct = 0
            process_total = len(process_examples)
            
            # 在evaluate方法中添加一个计数器来统计无效答案的样本
            invalid_answers = []  # 存储无效答案的样本索引
            
            # 添加两个列表用于记录样本预测内容
            valid_predictions = []    # 存储成功提取答案的预测内容
            invalid_predictions = []  # 存储未能提取答案的预测内容
            
            for batch_idx in range(num_batches):
                try:
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, process_total)
                    batch_examples = process_examples[batch_start:batch_end]
                    
                    log_info(f"进程 {local_rank} 正在评估第 {batch_idx + 1}/{num_batches} 批次...")
                    
                    # 找出当前批次中最长的序列长度
                    max_length = max(example["input_ids"].shape[1] for example in batch_examples)
                    
                    # 准备批次输入，使用pad_token_id进行左侧padding
                    padded_input_ids = []
                    for example in batch_examples:
                        input_ids = example["input_ids"]
                        # 计算需要padding的长度
                        pad_length = max_length - input_ids.shape[1]
                        if pad_length > 0:
                            # 在序列左侧添加padding
                            padding = torch.full(
                                (1, pad_length),
                                self.tokenizer.pad_token_id,
                                dtype=input_ids.dtype,
                                device=input_ids.device
                            )
                            input_ids = torch.cat([padding, input_ids], dim=1)  # 左侧padding
                        padded_input_ids.append(input_ids)
                    
                    # 将padding后的输入拼接成批次
                    batch_input_ids = torch.cat(padded_input_ids, dim=0)
                    
                    # 优化内存：分批移动数据到设备
                    batch_input_ids = batch_input_ids.to(self.model.device)
                    
                    # 创建attention mask
                    attention_mask = (batch_input_ids != self.tokenizer.pad_token_id)
                    
                    # 批量生成答案
                    with torch.no_grad():
                        outputs = self.model.generate(
                            batch_input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=512,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # 处理每个样本的预测结果
                    for i, example in enumerate(batch_examples):
                        try:
                            # 获取原始输入长度
                            orig_length = example["input_ids"].shape[1]
                            # 计算padding后的偏移量
                            pad_offset = max_length - orig_length
                            # 只解码新生成的token，考虑左侧padding的偏移
                            generated_tokens = outputs[i][max_length:]  # 使用max_length而不是orig_length
                            prediction = self.tokenizer.decode(generated_tokens)
                            
                            # 使用first_option_postprocess替换原来的答案提取逻辑
                            pred_letter = first_option_postprocess(prediction, 'ABCDEF')
                            if pred_letter:  # pred_letter已经是A/B/C/D中的一个
                                pred_label = ord(pred_letter) - ord('A')
                                is_correct = pred_label == example["label"]
                                total_correct += int(is_correct)
                                
                                # 记录成功提取答案的预测内容(只记录前3个)
                                if len(valid_predictions) < 3:
                                    valid_predictions.append({
                                        'index': start_idx + batch_idx * batch_size + i,
                                        'prediction': prediction,
                                        'extracted_answer': pred_letter,
                                        'correct_answer': chr(65 + example["label"]),
                                        'is_correct': is_correct
                                    })
                                
                                # 保持现有的进度日志输出...
                            else:
                                # 记录未能提取答案的预测内容(只记录前3个)
                                if len(invalid_predictions) < 3:
                                    invalid_predictions.append({
                                        'index': start_idx + batch_idx * batch_size + i,
                                        'prediction': prediction
                                    })
                                invalid_answers.append(start_idx + batch_idx * batch_size + i)
                        
                        except Exception as e:
                            log_info(f"进程 {local_rank} 处理样本时出错: {str(e)}")
                            continue
                    
                    # 在批次处理完成后输出预测内容分析
                    if valid_predictions:
                        log_info("\n=== 成功提取答案的预测示例(前3个) ===")
                        for pred in valid_predictions:
                            log_info(f"\n样本 {pred['index']}:")
                            log_info(f"预测内容: {pred['prediction']}")
                            log_info(f"提取答案: {pred['extracted_answer']}")
                            log_info(f"正确答案: {pred['correct_answer']}")
                            log_info(f"是否正确: {'✓' if pred['is_correct'] else '✗'}")

                    if invalid_predictions:
                        log_info("\n=== 未能提取答案的预测示例(前3个) ===")
                        for pred in invalid_predictions:
                            log_info(f"\n样本 {pred['index']}:")
                            log_info(f"预测内容: {pred['prediction']}")
                
                    # 保持现有的无效答案统计日志
                    if invalid_answers:
                        log_info(f"\n警告：以下样本未能提取到有效答案字母: {invalid_answers}")
                        log_info(f"共有 {len(invalid_answers)} 个样本未能提取到有效答案字母")
                
                except Exception as e:
                    log_info(f"进程 {local_rank} 处理批次 {batch_idx + 1} 时出错: {str(e)}", level="error")
                    continue
                
                # 定期输出进度
                if (batch_idx + 1) % 5 == 0:
                    current_accuracy = total_correct / ((batch_idx + 1) * batch_size)
                    log_info(f"进程 {local_rank} 当前进度: {batch_idx + 1}/{num_batches} 批次, 准确率: {current_accuracy:.4f}")
            
            # 收集所有进程的结果
            process_metrics = torch.tensor([total_correct, process_total], device=self.model.device)
            gathered_metrics = [torch.zeros_like(process_metrics) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_metrics, process_metrics)
            
            # 在主进程中计算总体指标
            if local_rank == 0:
                total_correct = sum(metrics[0].item() for metrics in gathered_metrics)
                total_samples = sum(metrics[1].item() for metrics in gathered_metrics)
                accuracy = total_correct / total_samples if total_samples > 0 else 0
                
                log_info(f"评估完成! 准确率: {accuracy:.4f} ({total_correct}/{total_samples})")
                
                # 记录评估指标
                self.metrics_recorder.add_metric(
                    self.state.global_step,
                    'accuracy',
                    accuracy,
                    epoch=self.state.epoch  # 添加epoch信息
                )
                
                # 修改返回的指标字典，确保包含eval_前缀
                metrics = {
                    f"{metric_key_prefix}_accuracy": accuracy,  # 添加前缀
                    f"{metric_key_prefix}_loss": 0.0  # 保持一致的前缀
                }
                
                self.log(metrics)
                return metrics
            
            # 非主进程返回空指标，同样使用前缀
            return {
                f"{metric_key_prefix}_accuracy": 0.0,
                f"{metric_key_prefix}_loss": 0.0
            }
        
        log_info("没有找到评估样本，跳过评估...")
        return super().evaluate(
            eval_dataset, 
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )


def print_named_parameters(model):
    """打印模型的命名参数信息
    
    Args:
        model: 要打印参数的模型
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        # 打印每个参数的名称、形状和否需要梯度
        log_info(
            f"Parameter name: {name}, Shape: {param.shape}, "
            f"Requires gradient: {param.requires_grad}, "
            f"Number of parameters: {num_params}"
        )
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    # 打印可训练参数的统计信息
    log_info(
        f"Total parameters: {all_param}\n"
        f"Trainable parameters: {trainable_params}\n"
        f"Percentage of trainable parameters: {100 * trainable_params / all_param:.2f}%"
    )


def check_environment():
    """检查训练环境"""
    log_info("检查PEFT环境...")
    try:
        import peft
        log_info(f"PEFT version: {peft.__version__}")
    except ImportError:
        raise ImportError("PEFT not installed. Please install it first.")
        
    log_info("检查PyTorch环境...")
    log_info(f"PyTorch version: {torch.__version__}")
    
    # 查NPU环境
    log_info("检查NPU环境...")
    try:
        import torch_npu
        log_info("torch_npu imported successfully")
        log_info(f"NPU available: {torch.npu.is_available()}")
        log_info(f"NPU device count: {torch.npu.device_count()}")
    except ImportError:
        raise ImportError("torch_npu not installed. Please install it first.")

    
def train():
    check_environment()
    # 输出提示词模板
    log_info("提示词模板:")
    log_info(PROMPT_DICT["prompt_input"])
    log_info(PROMPT_DICT["prompt_output"])
        
    """训练流程的主函数"""
    # 解析命令行参数
    log_info("开始解析命令行参数...")
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    log_info("命令行参数解析结束...")
    

    log_info("开始加载预训练模型...")
    # 加载预训练模型
    model = openmind.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False
    )
    
        
    log_info("预练模型加载结束...")

    if model_args.use_peft:
        log_info("开始添加PEFT配置...")
        
        # 将模型设置为训练模式
        model.train()
        
        # 冻结基础模型参数
        for param in model.parameters():
            param.requires_grad = False
        
        # 修改target_modules以匹配模型的实际参数名称
        target_modules = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj"
        ]
        
        if model_args.peft_method == "lora":
            log_info("配置LoRA...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=model_args.lora_dropout,
                bias="none",
                init_lora_weights=True,
                modules_to_save=None,
                fan_in_fan_out=False
            )
            
            # 获取PEFT模型
            model = get_peft_model(model, peft_config)
            
            # 确保adapter正确初始化
            if not hasattr(model, 'peft_config'):
                raise ValueError("PEFT model initialization failed")
            
            # 打印PEFT配置信息
            log_info(f"PEFT config: {model.peft_config}")
            
            # 确保adapter被加载
            if not model.active_adapters:
                model.add_adapter('default', peft_config)
            
            # 激活adapter
            model.set_adapter('default')
            
            # 确保LoRA参数是可训练的，并统计参数数量
            trainable_params = 0
            all_param = 0
            for name, param in model.named_parameters():
                num_params = param.numel()
                all_param += num_params
                if "lora" in name:
                    param.requires_grad = True
                    trainable_params += num_params
                    # log_info(f"Set requires_grad=True for {name} with {num_params} parameters")
                else:
                    param.requires_grad = False

            # 验证LoRA参数设置
            if trainable_params == 0:
                raise ValueError("No LoRA parameters found! PEFT configuration might be incorrect.")

            # 打印参数统计
            log_info(f"Total parameters: {all_param:,}")
            log_info(f"Trainable parameters: {trainable_params:,}")
            log_info(f"Percentage of trainable parameters: {100 * trainable_params / all_param:.2f}%")

            # 验证模型是否在训练模式
            if not model.training:
                log_info("Warning: Model not in training mode. Setting to training mode...")
                model.train()
            else:
                log_info("Model is already in training mode.")

    log_info("开始加载分词器...")
    # 加载分词器
    tokenizer = openmind.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True
    )

    # 设置特殊token
    special_tokens_dict = {
        "pad_token": DEFAULT_PAD_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "eos_token": DEFAULT_EOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
        "additional_special_tokens": [DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN]
    }

    log_info("开始调整tokenizer和模型embedding层的大小...")
    # 使用smart_tokenizer_and_embedding_resize函数处理特殊token
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model
    )
    log_info("调整tokenizer和模型embedding层的大小结束...")

    # 打印特殊token信息
    log_info("特殊token信息:")
    log_info(f"PAD token: {tokenizer.pad_token} -> ID: {tokenizer.pad_token_id}")
    log_info(f"EOS token: {tokenizer.eos_token} -> ID: {tokenizer.eos_token_id}")
    log_info(f"BOS token: {tokenizer.bos_token} -> ID: {tokenizer.bos_token_id}")
    log_info(f"UNK token: {tokenizer.unk_token} -> ID: {tokenizer.unk_token_id}")

    # # 打印tokenizer的详细信息以便调试
    # log_info("Tokenizer special tokens:")
    # log_info(f"Special tokens map: {tokenizer.special_tokens_map}")
    # log_info(f"All special tokens: {tokenizer.all_special_tokens}")
    # log_info(f"Vocabulary size: {len(tokenizer)}")


    log_info("开始准备训练数据...")
    # 准备训练数据
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    log_info("数据准备结束...")


    # # 打印前几个input_ids
    # log_info("前几个input_ids示例:")
    # for i in range(min(3, len(data_module['train_dataset']))):
    #     sample = data_module['train_dataset'][i]
    #     log_info(f"Sample {i} input_ids: {sample['input_ids']}")
    #     log_info(f"Sample {i} labels: {sample['labels']}")

    # # 打印前10条提示模板
    # log_info("前10条提示模板:")
    # for i in range(min(3, len(data_module['train_dataset']))):
    #     sample = data_module['train_dataset'][i]
    #     try:
    #         input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    #         output_text = tokenizer.decode([x for x in sample['labels'] if x != IGNORE_INDEX], skip_special_tokens=False)
    #         log_info(f"Sample {i} prompt: \n{input_text}")
    #         log_info(f"Sample {i} output: \n{output_text}")
    #     except Exception as e:
    #         log_info(f"Error processing sample {i}: {str(e)}")
    #         continue


    log_info("开始创建训练器...")
    # 创建评估样本
    eval_examples = None
    if data_args.eval_data_path:
        eval_data = jload(data_args.eval_data_path)
        
        # 处理评估样本数量
        if data_args.eval_num_samples > 0:
            # 确保样本数是8的倍数（为了8卡分布式评估）
            num_samples = ((data_args.eval_num_samples + 7) // 8) * 8
            num_samples = min(num_samples, len(eval_data))
            log_info(f"使用 {num_samples} 个样本进行评估 (调整为8的倍数)")
            # 从头开始选择指定数量的样本
            eval_data = eval_data[:num_samples]
        else:
            # 如果样本总数不是8的倍数，调整为8的倍数
            num_samples = (len(eval_data) // 8) * 8
            if num_samples != len(eval_data):
                log_info(f"调整评估样本数量从 {len(eval_data)} 到 {num_samples} (确保是8的倍数)")
                eval_data = eval_data[:num_samples]
        
        eval_examples = create_eval_examples(model, tokenizer, eval_data)
    
    # 使用自定义的PeftTrainer
    trainer = PeftTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        eval_examples=eval_examples,
        **data_module
    )
    log_info("训练器创建结束...")

    log_info("开始训练...")
    train_result = trainer.train()
    log_info("训练结束...")

    # 训练结束后绘制最终的loss曲线
    trainer.metrics_recorder.plot_all_metrics()

    # 如果使用了PEFT方法，合并并保存模型
    if model_args.use_peft:
        log_info("开始合并PEFT模型...")
        try:
            # 合并模型
            merged_model = model.merge_and_unload()
            
            # 创建合并模型的保存路径
            merged_model_path = os.path.join(training_args.output_dir, "merged_model")
            
            # 保存合并后的模型，使用分片以处理大型模型
            log_info("开始保存合并后的模型...")
            merged_model.save_pretrained(
                merged_model_path,
                safe_serialization=True
            )
            
            # 同时保存分词器到合并模型目录
            log_info("保存分词器到合并模型目录...")
            tokenizer.save_pretrained(merged_model_path)
            
            log_info("合并后的模型和分词器保存完成...")
            print_named_parameters(merged_model) # 打印合并后的模型参数
        except Exception as e:
            log_info(f"合并模型时出错: {str(e)}")
            # 如果合并失败，至少保存当前模型状态和分词器
            log_info("保存未合并的模型和分词器...")
            trainer.save_model(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
    else:
        # 保存原始模型
        log_info("开始保存模型...")
        trainer.save_model(output_dir=training_args.output_dir)
        log_info("模型保存结束...")

    log_info("开始保存分词器...")
    tokenizer.save_pretrained(training_args.output_dir)
    log_info("分词器保存结束...")

    log_info("训练完成...")
    

if __name__ == "__main__":
    train()
