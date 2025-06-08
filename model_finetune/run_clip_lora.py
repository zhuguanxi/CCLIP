#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import json
import wandb
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, ToTensor, Compose
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
    AutoConfig,
    AutoModelForVision2Seq,
    AutoProcessor,
    CLIPConfig,
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# 导入PEFT库
from peft import LoraConfig, TaskType, get_peft_model

# 导入自定义的数据加载和损失函数模块
from data import get_dataloaders, collate_fn
from loss import (
    ablation_caption_clip, ablation_concept_clip,
    ablation_caption_negclip, ablation_concept_negclip,
    ablation_caption_concept_clip, ablation_caption_neg_concept_clip,
    ablation_caption_concept_neg, cultureclip_loss
)

# 在文件开头添加索引位置定义
INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}

# 更新视觉模型的索引位置定义
INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    },
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    },
    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    }
}

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.48.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_vision_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the vision model parameters or not."}
    )
    freeze_text_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the text model parameters or not."}
    )
    # 添加LoRA相关参数
    use_lora: bool = field(
        default=False, metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning."}
    )
    lora_r: int = field(
        default=8, metadata={"help": "LoRA attention dimension."}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "LoRA alpha parameter."}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "LoRA dropout parameter."}
    )
    apply_lora_to_vision: bool = field(
        default=False, metadata={"help": "Whether to apply LoRA to vision model."}
    )
    apply_lora_to_text: bool = field(
        default=False, metadata={"help": "Whether to apply LoRA to text model."}
    )
    position: str = field(
        default="all",
        metadata={"help": "Which layers to apply LoRA to (e.g., 'all', 'half-up', 'half-bottom')"}
    )
    params: str = field(
        default="qv",
        metadata={"help": "Which parameters to apply LoRA to. Can be any combination of 'q', 'k', 'v', 'o' characters (e.g., 'qkv' for all query, key, value projections; 'qv' for only query and value projections)"}
    )
    backbone: str = field(
        default="ViT-L/14",
        metadata={"help": "The backbone architecture to use (ViT-B/16, ViT-B/32, or ViT-L/14)"}
    )
    # 添加损失函数选择参数
    loss_type: str = field(
        default="cultureclip",
        metadata={
            "help": (
                "The type of loss function to use. Options: "
                "clip (standard CLIP loss), "
                "caption_clip (caption only CLIP loss), "
                "concept_clip (concept only CLIP loss), "
                "caption_negclip (caption with negative samples), "
                "concept_negclip (concept with negative samples), "
                "caption_concept_clip (caption + concept CLIP loss), "
                "caption_neg_concept_clip (caption with negatives + concept CLIP), "
                "caption_concept_neg (caption CLIP + concept with negatives), "
                "cultureclip (full CultureCLIP loss with all components)"
            )
        },
    )
    lambda_caption: float = field(
        default=0.5, metadata={"help": "Weight for caption loss component."}
    )
    lambda_concept: float = field(
        default=0.5, metadata={"help": "Weight for concept loss component."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    pos_image_column: Optional[str] = field(
        default="pos_image_path",
        metadata={"help": "The name of the column in the datasets containing the positive image file paths."},
    )
    pos_caption_column: Optional[str] = field(
        default="pos_caption",
        metadata={"help": "The name of the column in the datasets containing the positive image captions."},
    )
    # 添加负样本和概念列名
    neg_image_column: Optional[str] = field(
        default="neg_image_path",
        metadata={"help": "The name of the column in the datasets containing the negative image file paths."},
    )
    neg_caption_column: Optional[str] = field(
        default="neg_caption",
        metadata={"help": "The name of the column in the datasets containing the negative image captions."},
    )
    pos_concept_column: Optional[str] = field(
        default="pos_concept",
        metadata={"help": "The name of the column in the datasets containing the positive concept."},
    )
    neg_concept_column: Optional[str] = field(
        default="neg_concept",
        metadata={"help": "The name of the column in the datasets containing the negative concept."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    max_seq_length: Optional[int] = field(
        default=77,  # CLIP的文本模型最大位置嵌入大小为77
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. "
                "CLIP's text model has a maximum position embedding size of 77."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl"], "`train_file` should be a csv, json, or jsonl file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl"], "`validation_file` should be a csv, json, or jsonl file."
        
        # 确保max_seq_length不超过CLIP的限制
        if self.max_seq_length > 77:
            print(f"Warning: max_seq_length ({self.max_seq_length}) is greater than CLIP's limit (77). Setting to 77.")
            self.max_seq_length = 77
                
        # 验证所有必需列都已指定
        required_columns = [
            self.pos_image_column,       # pos_image_path
            self.pos_caption_column,     # pos_caption
            self.neg_image_column,   # neg_image_path
            self.neg_caption_column, # neg_caption
            self.pos_concept_column, # pos_concept
            self.neg_concept_column  # neg_concept
        ]
        
        missing_columns = [col for col in required_columns if col is None]
        if missing_columns:
            raise ValueError(f"所有实验都需要完整的六个输入列，请为以下列指定值: {missing_columns}")


dataset_name_mapping = {
    "image_caption_dataset.py": ("image_path", "caption"),
}


# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = Compose([
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        ])

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # 提取model_args参数，然后从kwargs中移除它，以避免传递给父类
        self.model_args = kwargs.pop("model_args", None)
        super().__init__(*args, **kwargs)
        
    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        """
        执行一个训练步骤
        
        Args:
            model: 要训练的模型
            inputs: 模型的输入和目标
            num_items_in_batch: 批次中的项目数
            
        Returns:
            torch.Tensor: 包含此批次训练损失的张量
        """
        # 设置模型为训练模式
        model.train()
        
        # 过滤输入，确保不包含inputs_embeds
        filtered_inputs = {}
        for k, v in inputs.items():
            if k not in ["inputs_embeds"]:
                filtered_inputs[k] = v
        
        # 提取正负样本（用于日志记录和调试）
        pos_image = filtered_inputs.get("pos_image")
        neg_image = filtered_inputs.get("neg_image")
        pos_caption = filtered_inputs.get("pos_caption")
        neg_caption = filtered_inputs.get("neg_caption")
        pos_concept = filtered_inputs.get("pos_concept")
        neg_concept = filtered_inputs.get("neg_concept")
        
        # 使用上下文管理器计算损失
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, filtered_inputs, 
                                    pos_image=pos_image, neg_image=neg_image, 
                                    pos_caption=pos_caption, neg_caption=neg_caption,
                                    pos_concept=pos_concept, neg_concept=neg_concept)
        
        # 如果使用多GPU，对损失取平均
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        # 如果使用梯度累积，对损失进行归一化
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        # 执行反向传播
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        
        return loss.detach()

    def evaluation_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        """
        执行一个评估步骤
        
        Args:
            model: 要评估的模型
            inputs: 模型的输入和目标
            num_items_in_batch: 批次中的项目数
            
        Returns:
            Dict: 包含损失和可能的其他指标的字典
        """
        # 设置模型为评估模式
        model.eval()
        
        # 过滤输入，确保不包含inputs_embeds
        filtered_inputs = {}
        for k, v in inputs.items():
            if k not in ["inputs_embeds"]:
                filtered_inputs[k] = v
        
        # 提取正负样本（用于日志记录和调试）
        pos_image = filtered_inputs.get("pos_image")
        neg_image = filtered_inputs.get("neg_image")
        pos_caption = filtered_inputs.get("pos_caption")
        neg_caption = filtered_inputs.get("neg_caption")
        pos_concept = filtered_inputs.get("pos_concept")
        neg_concept = filtered_inputs.get("neg_concept")
        
        # 禁用梯度计算
        with torch.no_grad():
            # 计算损失
            loss = self.compute_loss(model, filtered_inputs, 
                                    pos_image=pos_image, neg_image=neg_image, 
                                    pos_caption=pos_caption, neg_caption=neg_caption,
                                    pos_concept=pos_concept, neg_concept=neg_concept)
            
            # 如果使用多GPU，对损失取平均
            if self.args.n_gpu > 1:
                loss = loss.mean()
        
        return {"loss": loss.detach()}

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        重写prediction_step方法，过滤掉inputs_embeds参数
        """
        # 过滤输入，确保不包含inputs_embeds
        filtered_inputs = {}
        for k, v in inputs.items():
            if k not in ["inputs_embeds"]:
                filtered_inputs[k] = v
        
        # 打印输入键，帮助调试
        print(f"prediction_step input keys: {filtered_inputs.keys()}")
                
        # 设置模型为评估模式
        model.eval()
        
        with torch.no_grad():
            # 计算损失
            loss = self.compute_loss(model, filtered_inputs)
            print(f"prediction_step loss: {loss.item()}")
            
            # 获取图像和文本特征
            if "pos_image" in filtered_inputs and "pos_caption" in filtered_inputs:
                with torch.no_grad():
                    image_features = model.get_image_features(pixel_values=filtered_inputs.get("pos_image"))
                    text_features = model.get_text_features(input_ids=filtered_inputs.get("pos_caption"))
                    
                    # 计算logits
                    logits_per_image = torch.matmul(image_features, text_features.t()) * model.logit_scale.exp()
                    logits_per_text = logits_per_image.t()
                    
                    # 创建标签（对角线为1，其他为0）
                    labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
                    
                    return (loss, (logits_per_image, logits_per_text), labels)
            else:
                print(f"Missing required keys in inputs: pos_image={('pos_image' in filtered_inputs)}, pos_caption={('pos_caption' in filtered_inputs)}")
            
            # 如果没有足够的数据，返回空结果
            return (loss, None, None)

    def compute_loss(self, model, inputs, **kwargs):
        """
        计算损失函数
        
        Args:
            model: 模型
            inputs: 输入数据
            **kwargs: 其他参数，包括正负样本
            
        Returns:
            loss: 损失值
        """
        # 提取正负样本数据
        pos_image = kwargs.get("pos_image", inputs.get("pos_image"))
        neg_image = kwargs.get("neg_image", inputs.get("neg_image"))
        pos_caption = kwargs.get("pos_caption", inputs.get("pos_caption"))
        neg_caption = kwargs.get("neg_caption", inputs.get("neg_caption"))
        pos_concept = kwargs.get("pos_concept", inputs.get("pos_concept"))
        neg_concept = kwargs.get("neg_concept", inputs.get("neg_concept"))
        
        # 打印输入数据的形状，帮助调试
        print(f"compute_loss inputs shapes: pos_image={pos_image.shape if pos_image is not None else None}, "
              f"neg_image={neg_image.shape if neg_image is not None else None}, "
              f"pos_caption={pos_caption.shape if pos_caption is not None else None}, "
              f"neg_caption={neg_caption.shape if neg_caption is not None else None}, "
              f"pos_concept={pos_concept.shape if pos_concept is not None else None}, "
              f"neg_concept={neg_concept.shape if neg_concept is not None else None}")
        
        try:
            # 获取损失函数类型和权重
            if self.model_args is not None:
                loss_type = getattr(self.model_args, "loss_type", "cultureclip")
                lambda_caption = getattr(self.model_args, "lambda_caption", 0.5)
                lambda_concept = getattr(self.model_args, "lambda_concept", 0.5)
                print(f"Loss type: {loss_type}, lambda_caption: {lambda_caption}, lambda_concept: {lambda_concept}")
            else:
                # 如果model_args为None，使用默认值
                loss_type = "cultureclip"
                lambda_caption = 0.5
                lambda_concept = 0.5
                print("Using default loss parameters (model_args is None)")
            
            # 计算所有嵌入 - 无论使用哪种损失函数类型，都提取所有嵌入
            # 获取正样本的嵌入
            pos_img_embs = model.get_image_features(pixel_values=pos_image)
            pos_caption_embs = model.get_text_features(input_ids=pos_caption)
            
            # 获取负样本的嵌入
            neg_img_embs = model.get_image_features(pixel_values=neg_image)
            neg_caption_embs = model.get_text_features(input_ids=neg_caption)
            
            # 获取概念的嵌入
            pos_concept_embs = model.get_text_features(input_ids=pos_concept)
            neg_concept_embs = model.get_text_features(input_ids=neg_concept)
            
            print(f"Embeddings shapes: pos_img={pos_img_embs.shape}, pos_caption={pos_caption_embs.shape}, "
                  f"neg_img={neg_img_embs.shape}, neg_caption={neg_caption_embs.shape}, "
                  f"pos_concept={pos_concept_embs.shape}, neg_concept={neg_concept_embs.shape}")
            
            # 根据loss_type选择损失函数
            if loss_type == "clip":
                # 标准CLIP损失
                loss, _ = clip_loss(pos_img_embs, pos_caption_embs, model.logit_scale.exp())
                print(f"Using clip_loss, result: {loss.item()}")
            elif loss_type == "caption_clip":
                # 只使用caption的CLIP损失
                loss, _ = ablation_caption_clip(pos_caption_embs, pos_img_embs, model.logit_scale.exp())
                print(f"Using ablation_caption_clip, result: {loss.item()}")
            elif loss_type == "concept_clip":
                # 只使用concept的CLIP损失
                loss, _ = ablation_concept_clip(pos_concept_embs, pos_img_embs, model.logit_scale.exp())
                print(f"Using ablation_concept_clip, result: {loss.item()}")
            elif loss_type == "caption_negclip":
                # 使用caption的negclip损失
                loss, _ = ablation_caption_negclip(pos_caption_embs, pos_img_embs, neg_caption_embs, neg_img_embs, model.logit_scale.exp())
                print(f"Using ablation_caption_negclip, result: {loss.item()}")
            elif loss_type == "concept_negclip":
                # 使用concept的negclip损失
                loss, _ = ablation_concept_negclip(pos_concept_embs, pos_img_embs, neg_concept_embs, neg_img_embs, model.logit_scale.exp())
                print(f"Using ablation_concept_negclip, result: {loss.item()}")
            elif loss_type == "caption_concept_clip":
                # 使用caption和concept的CLIP损失
                loss, _ = ablation_caption_concept_clip(pos_concept_embs, pos_caption_embs, pos_img_embs, model.logit_scale.exp(), lambda_caption, lambda_concept)
                print(f"Using ablation_caption_concept_clip, result: {loss.item()}")
            elif loss_type == "caption_neg_concept_clip":
                # 使用caption的negclip和concept的CLIP损失
                loss, _ = ablation_caption_neg_concept_clip(pos_concept_embs, pos_caption_embs, pos_img_embs, neg_caption_embs, neg_img_embs, model.logit_scale.exp(), lambda_caption, lambda_concept)
                print(f"Using ablation_caption_neg_concept_clip, result: {loss.item()}")
            elif loss_type == "caption_concept_neg":
                # 使用caption的CLIP和concept的negclip损失
                loss, _ = ablation_caption_concept_neg(pos_concept_embs, pos_caption_embs, pos_img_embs, neg_concept_embs, neg_img_embs, model.logit_scale.exp(), lambda_caption, lambda_concept)
                print(f"Using ablation_caption_concept_neg, result: {loss.item()}")
            else:  # 默认使用cultureclip
                # 完整的CultureCLIP损失
                print(f"Using cultureclip_loss with parameters: lambda_caption={lambda_caption}, lambda_concept={lambda_concept}")
                loss, _ = cultureclip_loss(
                    pos_concept_embs=pos_concept_embs, 
                    pos_caption_embs=pos_caption_embs, 
                    pos_img_embs=pos_img_embs,
                    neg_concept_embs=neg_concept_embs, 
                    neg_caption_embs=neg_caption_embs, 
                    neg_img_embs=neg_img_embs,
                    logit_scale=model.logit_scale.exp(), 
                    lambda_caption=lambda_caption, 
                    lambda_concept=lambda_concept
                )
                print(f"cultureclip_loss result: {loss.item()}")
            
            return loss
            
        except Exception as e:
            # 如果出现异常，记录错误并尝试基本的CLIP损失
            print(f"Error in compute_loss: {e}")
            print(f"Falling back to standard CLIP loss")
            try:
                # 尝试最基本的CLIP损失计算
                image_features = model.get_image_features(pixel_values=pos_image)
                text_features = model.get_text_features(input_ids=pos_caption)
                
                # 计算相似度和损失
                logit_scale = model.logit_scale.exp()
                loss, _ = clip_loss(image_features, text_features, logit_scale)
                print(f"Using fallback clip_loss, result: {loss.item()}")
                return loss
            except Exception as inner_e:
                print(f"Failed to compute basic CLIP loss: {inner_e}")
                # 如果仍然失败，返回零损失
                print("Returning zero loss due to errors")
                return torch.tensor(0.0, device=self.args.device)


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clip", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    wandb.init(
            project="cultureclip",
            name=f"clip_lora_{model_args.model_name_or_path.split('/')[-1]}_{model_args.loss_type}",
            config={
                "model_name": model_args.model_name_or_path,
                "loss_type": model_args.loss_type,
                "lambda_caption": model_args.lambda_caption,
                "lambda_concept": model_args.lambda_concept,
                "lora_r": model_args.lora_r,
                "lora_alpha": model_args.lora_alpha,
                "lora_dropout": model_args.lora_dropout,
                "position": model_args.position,
                "params": model_args.params,
                "backbone": model_args.backbone,
                "freeze_vision": model_args.freeze_vision_model,
                "freeze_text": model_args.freeze_text_model,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            }
    )

    # 3. Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load model, tokenizer, and processor
    if model_args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir, token=model_args.token
        )
    elif model_args.model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, token=model_args.token
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Load processor and model
    processor = CLIPProcessor.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, token=model_args.token
    )
    
    model = CLIPModel.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, token=model_args.token
    )

    # 根据参数冻结视觉模型
    if model_args.freeze_vision_model:
        logger.info("Freezing vision model parameters")
        for param in model.vision_model.parameters():
            param.requires_grad = False
    
    # 根据参数冻结文本模型
    if model_args.freeze_text_model:
        logger.info("Freezing text model parameters")
        for param in model.text_model.parameters():
            param.requires_grad = False

    # 5. 应用LoRA微调（如果需要）
    if model_args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # 确定要应用LoRA的模块
            target_modules = []
            
            # 为视觉模型应用LoRA
            if model_args.apply_lora_to_vision:
                # 根据backbone和position确定目标层
                try:
                    # 使用预定义的INDEX_POSITIONS_VISION字典获取层索引
                    vision_layers = INDEX_POSITIONS_VISION[model_args.backbone][model_args.position]
                except KeyError:
                    # 如果找不到对应的backbone或position，使用默认值
                    logger.warning(
                        f"Could not find layers for backbone={model_args.backbone} and position={model_args.position}. "
                        f"Using default layers."
                    )
                    # 默认使用所有层
                    if model_args.backbone == "ViT-L/14":
                        vision_layers = list(range(24))  # ViT-L/14有24层
                    else:
                        vision_layers = list(range(12))  # ViT-B默认12层

                # 根据params参数确定需要微调的模块类型
                for i in vision_layers:
                    if 'q' in model_args.params:
                        target_modules.append(f"vision_model.encoder.layers.{i}.self_attn.q_proj")
                    if 'k' in model_args.params:
                        target_modules.append(f"vision_model.encoder.layers.{i}.self_attn.k_proj")
                    if 'v' in model_args.params:
                        target_modules.append(f"vision_model.encoder.layers.{i}.self_attn.v_proj")
                    if 'o' in model_args.params:
                        target_modules.append(f"vision_model.encoder.layers.{i}.self_attn.out_proj")
            
            # 为文本模型应用LoRA
            if model_args.apply_lora_to_text:
                try:
                    # 使用预定义的INDEX_POSITIONS_TEXT字典获取层索引
                    text_layers = INDEX_POSITIONS_TEXT[model_args.position]
                except KeyError:
                    # 如果找不到对应的position，使用默认值
                    logger.warning(
                        f"Could not find layers for position={model_args.position} in text model. "
                        f"Using default layers."
                    )
                    # 默认使用所有层
                    text_layers = list(range(12))
                    
                # 根据params参数确定需要微调的模块类型
                for i in text_layers:
                    if 'q' in model_args.params:
                        target_modules.append(f"text_model.encoder.layers.{i}.self_attn.q_proj")
                    if 'k' in model_args.params:
                        target_modules.append(f"text_model.encoder.layers.{i}.self_attn.k_proj")
                    if 'v' in model_args.params:
                        target_modules.append(f"text_model.encoder.layers.{i}.self_attn.v_proj")
                    if 'o' in model_args.params:
                        target_modules.append(f"text_model.encoder.layers.{i}.self_attn.out_proj")
            
            # 创建LoRA配置
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            
            # 应用LoRA
            model = get_peft_model(model, lora_config)
            logger.info(f"Successfully applied LoRA to model, target modules: {target_modules}")
            
            # 打印可训练参数比例
            model.print_trainable_parameters()
            
        except ImportError:
            logger.warning("PEFT library not found, cannot apply LoRA. Please install PEFT: pip install peft")
        except Exception as e:
            logger.error(f"Error applying LoRA: {str(e)}")
            logger.warning("Will continue training with original model")
    
    # 6. 加载数据
    train_dataloader, val_dataloader = get_dataloaders(
        train_file=data_args.train_file,
        processor=processor,
        tokenizer=tokenizer,
        pos_image_column=data_args.pos_image_column,
        pos_caption_column=data_args.pos_caption_column,
        neg_image_column=data_args.neg_image_column,
        neg_caption_column=data_args.neg_caption_column,
        pos_concept_column=data_args.pos_concept_column,
        neg_concept_column=data_args.neg_concept_column,
        max_seq_length=data_args.max_seq_length,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=data_args.preprocessing_num_workers or 4,
        val_file=data_args.validation_file,
        max_train_samples=data_args.max_train_samples,
        max_eval_samples=data_args.max_eval_samples,
    )
    
    # 7. 初始化训练器
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset if training_args.do_train else None,
        eval_dataset=val_dataloader.dataset if training_args.do_eval and val_dataloader else None,
        data_collator=None,  # 数据集类已经处理了批处理
        model_args=model_args,  # 传递模型参数给CustomTrainer
    )
    
    # 8. 训练
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        wandb.log(metrics)
    
    # 9. 评估
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        wandb.log({"eval_" + k: v for k, v in metrics.items()})
    
    wandb.finish()
    
    # 10. 保存模型和上传
    if model_args.use_lora:
        # 本地只保存adapter
        adapter_dir = os.path.join(training_args.output_dir, "lora_adapter")
        model.save_pretrained(adapter_dir)
        logger.info(f"LoRA adapter saved to: {adapter_dir}")
        
        if training_args.push_to_hub:
            logger.info("Preparing to merge LoRA weights and upload to Hub...")
            # 将模型移到CPU准备合并
            device = model.device
            model = model.to("cpu")
            torch.cuda.empty_cache()
            
            try:
                # 合并模型
                merged_model = model.merge_and_unload()
                logger.info("Successfully merged LoRA weights")
                
                # 移除PEFT配置
                if hasattr(merged_model.config, "peft_config"):
                    delattr(merged_model.config, "peft_config")
                
                # 创建详细的模型卡片
                model_card = f"""
# CultureCLIP Model (LoRA Fine-tuned)

This model is a CLIP model fine-tuned using LoRA method, with LoRA weights merged into the base model.

## Model Details

- **Base Model**: {model_args.model_name_or_path}
- **Task**: Contrastive Image-Text Matching
- **Training Parameters**:
  - Batch Size: {training_args.per_device_train_batch_size}
  - Learning Rate: {training_args.learning_rate}
  - Training Epochs: {training_args.num_train_epochs}
  - Gradient Accumulation Steps: {training_args.gradient_accumulation_steps}
  - Loss Function: {model_args.loss_type}
  - Caption Loss Weight: {model_args.lambda_caption}
  - Concept Loss Weight: {model_args.lambda_concept}

## LoRA Configuration
- LoRA Rank (r): {model_args.lora_r}
- LoRA Alpha: {model_args.lora_alpha}
- LoRA Dropout: {model_args.lora_dropout}
- Apply to Vision Model: {model_args.apply_lora_to_vision}
- Apply to Text Model: {model_args.apply_lora_to_text}
- Target Position: {model_args.position}
- Target Parameters: {model_args.params}
- Backbone: {model_args.backbone}

## Freezing Settings
- Freeze Vision Model: {model_args.freeze_vision_model}
- Freeze Text Model: {model_args.freeze_text_model}

## Dataset Information
- Training File: {data_args.train_file}
- Validation File: {data_args.validation_file}
- Maximum Sequence Length: {data_args.max_seq_length}

## Usage

```python
from transformers import CLIPModel, CLIPProcessor

# Load model and processor
model = CLIPModel.from_pretrained("{training_args.hub_model_id}")
processor = CLIPProcessor.from_pretrained("{training_args.hub_model_id}")

# Process text and images
inputs = processor(
    text=["A photo of a cat", "A photo of a dog"], 
    images=image, 
    return_tensors="pt", 
    padding=True
)

# Get outputs
outputs = model(**inputs)
```
"""
                # Save model card
                readme_path = os.path.join(training_args.output_dir, "README.md")
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(model_card)

                # Save all components
                merged_model.save_pretrained(training_args.output_dir)
                tokenizer.save_pretrained(training_args.output_dir)
                processor.save_pretrained(training_args.output_dir)

                # Upload to Hub
                merged_model.push_to_hub(
                    repo_id=training_args.hub_model_id,
                    token=model_args.token,
                    commit_message="Upload merged LoRA model with training details"
                )
                logger.info("Successfully uploaded merged model to Hub")

                # If evaluation needed, move model back to original device
                if training_args.do_eval:
                    model = model.to(device)
                    
            except Exception as e:
                logger.error(f"Error merging or uploading model: {str(e)}")
                logger.error("Please check your HF credentials or upload model manually")
    else:
        # 非LoRA模型的处理保持不变
        if training_args.push_to_hub:
            trainer.push_to_hub()
            logger.info("Model uploaded to Hub")


if __name__ == "__main__":
    main()
    