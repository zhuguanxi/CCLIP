#!/bin/bash

# Basic configuration
# Replace these paths with your own paths
output_dir="./outputs/cultureclip_lora"  # Output directory for model checkpoints and logs
model_name_or_path="openai/clip-vit-base-patch32"  # Base model to use
# model_name_or_path="openai/clip-vit-large-patch14"  # Alternative model option
# tokenizer_name="openai/clip-vit-large-patch14"  # Optional: specify tokenizer if different from model

# Replace these paths with your own data paths
train_file="./data/train.jsonl"  # Path to your training data
validation_file="./data/val.jsonl"  # Path to your validation data

# Column names in your data files
# Modify these if your data uses different column names
pos_image_column="pos_image_path"  # Column name for positive image paths
pos_caption_column="pos_caption"   # Column name for positive captions
neg_image_column="neg_image_path"  # Column name for negative image paths
neg_caption_column="neg_caption"   # Column name for negative captions
pos_concept_column="pos_concept"   # Column name for positive concepts
neg_concept_column="neg_concept"   # Column name for negative concepts

# Loss function configuration
loss_type="cultureclip"       # Loss function type, options:
                              # clip (standard CLIP loss)
                              # caption_clip (caption only CLIP loss)
                              # concept_clip (concept only CLIP loss)
                              # caption_negclip (caption with negative samples)
                              # concept_negclip (concept with negative samples)
                              # caption_concept_clip (caption + concept CLIP loss)
                              # caption_neg_concept_clip (caption with negatives + concept CLIP)
                              # caption_concept_neg (caption CLIP + concept with negatives)
                              # cultureclip (full CultureCLIP loss)
lambda_caption=0.7            # Caption loss weight
lambda_concept=0.3            # Concept loss weight

# Freeze model parameters
freeze_vision_model=true      # Whether to freeze vision model parameters
freeze_text_model=true        # Whether to freeze text model parameters
apply_lora_to_vision=true     # Whether to apply LoRA to vision model
apply_lora_to_text=true       # Whether to apply LoRA to text model

# LoRA specific parameters
backbone="ViT-B/32"     # Options: ViT-B/16, ViT-B/32, ViT-L/14
# backbone="ViT-L/14"   # Alternative backbone option
position="all"          # Options: 
                        # For ViT-L/14: all, half-up, half-bottom
                        # For ViT-B/16: all, half-up, half-bottom, top, top3, bottom, mid, up
                        # For ViT-B/32: all, half-up, half-bottom, bottom, mid, up
params="qv"            # Options: q, k, v, o (which attention parameters to apply LoRA to)
lora_r=4                # LoRA rank
lora_alpha=16           # LoRA alpha
lora_dropout=0.1        # LoRA dropout

# Training configuration
batch_size=128          # Batch size per GPU
learning_rate=3e-6      # Learning rate
warmup_steps=100        # Number of warmup steps
weight_decay=0.1        # Weight decay
num_epochs=10           # Number of training epochs
eval_steps=500          # Evaluation frequency in steps
save_steps=500          # Checkpoint saving frequency in steps
gradient_accumulation_steps=16  # Gradient accumulation steps, can reduce GPU memory usage

# Input parameter list - includes all required input column names
input_args="--pos_image_column $pos_image_column --pos_caption_column $pos_caption_column --neg_image_column $neg_image_column --neg_caption_column $neg_caption_column --pos_concept_column $pos_concept_column --neg_concept_column $neg_concept_column"

# Execute training
# Replace CUDA_VISIBLE_DEVICES with your desired GPU ID
CUDA_VISIBLE_DEVICES=0 python run_clip_lora.py \
    --output_dir $output_dir \
    --model_name_or_path $model_name_or_path \
    --train_file $train_file \
    --validation_file $validation_file \
    $input_args \
    --loss_type $loss_type \
    --lambda_caption $lambda_caption \
    --lambda_concept $lambda_concept \
    --remove_unused_columns=False \
    --do_train \
    --do_eval \
    --per_device_train_batch_size=$batch_size \
    --gradient_accumulation_steps=$gradient_accumulation_steps \
    --learning_rate=$learning_rate \
    --warmup_steps=$warmup_steps \
    --weight_decay=$weight_decay \
    --num_train_epochs=$num_epochs \
    --eval_steps=$eval_steps \
    --save_steps=$save_steps \
    --overwrite_output_dir \
    --use_lora \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --apply_lora_to_vision \
    --apply_lora_to_text \
    --backbone $backbone \
    --position $position \
    --params $params \
    --push_to_hub  # Remove this if you don't want to push to Hugging Face Hub 