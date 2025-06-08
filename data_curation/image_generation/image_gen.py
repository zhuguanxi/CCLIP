import os
import json
import torch
from diffusers import DiffusionPipeline
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict

@dataclass
class InputData:
    path: str = field(metadata={"help": "Path to the dataset"})
    pos_caption_column: str = field(default="pos_caption", metadata={"help": "Column name of the positive caption"})
    neg_caption_column: str = field(default="neg_caption", metadata={"help": "Column name of the negative caption"})

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Model name or path"})
    use_safetensors: bool = field(default=True, metadata={"help": "Use SafeTensors"})

@dataclass 
class InferArguments:
    batch_size_per_device: int = field(default=8, metadata={"help": "Batch size per device"})
    num_devices: int = field(default=1, metadata={"help": "Number of devices"})
    height: int = field(default=512, metadata={"help": "Height of the image"})
    width: int = field(default=512, metadata={"help": "Width of the image"})
    steps: int = field(default=4, metadata={"help": "Number of steps"})
    output_dir: str = field(default="output", metadata={"help": "Output directory"})

class PromptDataset(Dataset):
    def __init__(self, data, pos_caption_column="pos_caption", neg_caption_column="neg_caption"):
        self.data = data
        self.pos_caption_column = pos_caption_column
        self.neg_caption_column = neg_caption_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Ensure both captions exist
        if self.pos_caption_column not in item or self.neg_caption_column not in item:
            raise ValueError(f"Missing required caption columns in item {idx}")
        return item

    def collate_fn(self, batch):
        return {
            "pos_caption": [item[self.pos_caption_column] for item in batch],
            "neg_caption": [item[self.neg_caption_column] for item in batch],
            "original_data": batch  # Keep original data for output
        }

def load_data(args):
    data = []
    with open(args.path, "r") as f:
        if args.path.endswith('.jsonl'):
            for line in f:
                data.append(json.loads(line.strip()))
        else:  # .json format
            data = json.load(f)
    return data

def create_output(orig_data, pos_caption, neg_caption, pos_image_path, neg_image_path):
    """Create output dictionary with image paths appended."""
    output = orig_data.copy()
    # Add image paths while preserving all original fields
    output.update({
        "pos_image_path": pos_image_path,
        "neg_image_path": neg_image_path
    })
    return output

def main():
    parser = HfArgumentParser((InputData, ModelArguments, InferArguments))
    data_args, model_args, infer_args = parser.parse_args_into_dataclasses()

    output_dir = infer_args.output_dir
    output_image_dir = os.path.join(output_dir, "image")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    # 检查是否有已生成的输出文件
    output_jsonl = os.path.join(output_dir, "output.jsonl")
    processed_ids = set()
    processed_data = []
    
    if os.path.exists(output_jsonl):
        print(f"找到已有输出文件: {output_jsonl}")
        with open(output_jsonl, "r") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    processed_data.append(item)
                    # 从文件路径中提取ID
                    if "pos_image_path" in item:
                        img_path = item["pos_image_path"]
                        img_id = int(os.path.basename(img_path).split("_")[1].split(".")[0])
                        processed_ids.add(img_id)
                except Exception as e:
                    print(f"处理已有输出时出错: {e}")
        
        print(f"已处理 {len(processed_ids)} 个图像对，将从下一个继续")

    dataset = load_data(data_args)

    prompt_dataset = PromptDataset(
        dataset, 
        data_args.pos_caption_column,
        data_args.neg_caption_column
    )

    dataloader = DataLoader(
        prompt_dataset,
        batch_size=infer_args.batch_size_per_device * infer_args.num_devices,
        shuffle=False,
        collate_fn=prompt_dataset.collate_fn
    )

    pipeline = DiffusionPipeline.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        use_safetensors=model_args.use_safetensors,
        variant="fp16"
    )
    pipeline.to("cuda")

    output = processed_data.copy()
    # 确定起始ID
    id = max(processed_ids) if processed_ids else 0
    
    # 计算要跳过的批次数
    skip_batches = id // (infer_args.batch_size_per_device * infer_args.num_devices)
    
    print(f"从ID {id+1} 开始继续生成，跳过前 {skip_batches} 批次")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx < skip_batches:
            continue
            
        pos_captions = batch["pos_caption"]
        neg_captions = batch["neg_caption"]
        original_data = batch["original_data"]
        print(f"处理批次 {batch_idx+1}/{len(dataloader)}，包含 {len(pos_captions)} 个样本")
        
        # Generate positive images
        pos_images = pipeline(prompt=pos_captions, num_inference_steps=infer_args.steps, guidance_scale=0.0).images
        # Generate negative images
        neg_images = pipeline(prompt=neg_captions, num_inference_steps=infer_args.steps, guidance_scale=0.0).images

        for pos_caption, neg_caption, pos_image, neg_image, orig_data in zip(
            pos_captions, neg_captions, pos_images, neg_images, original_data
        ):
            id = id + 1

            pos_image_path = f"{infer_args.output_dir}/image/pos_{id:06d}.png"
            neg_image_path = f"{infer_args.output_dir}/image/neg_{id:06d}.png"

            # 创建输出字典，在最后添加图片路径
            output_entry = create_output(
                orig_data,
                pos_caption,
                neg_caption,
                pos_image_path,
                neg_image_path
            )
            
            output.append(output_entry)
            pos_image.save(pos_image_path)
            neg_image.save(neg_image_path)
            # write single line caption_id_pair to a jsonl file
            with open(f"{infer_args.output_dir}/output.jsonl", "a") as f:
                json.dump(output_entry, f)
                f.write("\n")
        
        print(f"已完成到ID: {id}")

    with open(f"{infer_args.output_dir}/output.json", "w") as f:
        json.dump(output, f)
    
    print(f"生成完成，共处理 {id} 个图像对")

if __name__ == "__main__":
    main()