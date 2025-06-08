#!/bin/bash

# Input path to the SD format data
path="sd_input.jsonl"

# Column names for captions (these are the default values in the code)
pos_caption_column="pos_caption"
neg_caption_column="neg_caption"

# Model settings
model_name_or_path="stabilityai/stable-diffusion-3.5-large-turbo"
use_safetensors="True"

# Output settings
output_dir="/data/yuchen/CultureCLIP_data/sd_generated_images"
batch_size_per_device=16

# The script will automatically detect previously generated images and continue from the last checkpoint
# If output.jsonl exists in output_dir, the program will automatically skip processed data
# No need to modify any parameters, just run to continue generation

python image_gen.py \
    --path $path \
    --pos_caption_column $pos_caption_column \
    --neg_caption_column $neg_caption_column \
    --model_name_or_path $model_name_or_path \
    --use_safetensors $use_safetensors \
    --output_dir $output_dir \
    --batch_size_per_device $batch_size_per_device

