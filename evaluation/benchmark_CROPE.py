from datasets import load_dataset
import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from tqdm import tqdm
import json
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run CROPE benchmark")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to test")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cache_dir", type=str, default="/data/yuchen/huggingface", help="Cache directory for datasets")
    parser.add_argument("--output_dir", type=str, default="/home/yuchen/project/cultureCLIP/evaluator/results", help="Base output directory")
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = os.path.join(args.output_dir, "benchmark_CROPE")
    output_file = os.path.join(output_dir, f"{args.model_name.replace('/', '_')}.json")
    
    # 设置随机种子
    random.seed(args.random_seed)
    
    # 加载数据集并过滤
    dataset = load_dataset("Malvinan/CROPE", cache_dir=args.cache_dir)["test"]
    
    # 筛选 answer 为 "no" 的数据
    filtered_dataset = dataset.filter(
        lambda x: x["answer"].lower() == "no",  # 确保大小写不敏感
        batched=False
    )
    
    # 二次过滤无效图片
    valid_data = []
    for item in tqdm(filtered_dataset, desc="Validating images"):
        try:
            # 确保主图片对象有效
            image = item["image"]
            
            # 确保定义图片有效
            def_images = item["definition_images"]
            
            valid_data.append({
                "image": image,
                "definition_images": def_images,
                "metadata": item["metadata"]
            })
        except (IOError, ValueError, RuntimeError) as e:
            print(f"Skipping invalid image: {e}")
            continue
    
    # 加载 CLIP 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForZeroShotImageClassification.from_pretrained(args.model_name)
    model = model.to(device)

    def generate_questions(item, item_id):
        questions = []
        item_id = item_id + 1
        metadata = item.get("metadata", "")
        tar_concept = metadata.get("image_en_concept", "")
        def_concept = metadata.get("definition_concept", "")

        if not tar_concept or not def_concept:
            return questions

        tar_text = f"There is {tar_concept} in the image."
        def_text = f"There is {def_concept} in the image."
        options = [tar_text, def_text]

        # Question Type 1:
        random.shuffle(options)
        correct_idx_1 = options.index(tar_text)
        questions.append({
            "id": item_id,
            "image": item["image"],
            "options": options,
            "correct_idx": correct_idx_1,
            "type": "main_image"
        })

        # Question Type 2:
        for def_img in item["definition_images"]:
            correct_idx_2 = options.index(def_text)
            questions.append({
                "id": item_id,
                "image": def_img,
                "options": options,
                "correct_idx": correct_idx_2,
                "type": "definition_image"
            })
        
        return questions

    # 评估逻辑
    results = []
    correct = 0
    total = 0
    item_id = 0

    for item in tqdm(valid_data, desc="Evaluating"):
        try:
            # 生成问题
            questions = generate_questions(item, item_id)
            if not questions:
                continue
            
            # 评估每个问题
            for q in questions:
                # 准备模型输入
                inputs = processor(
                    text=q["options"],
                    images=q["image"],
                    return_tensors="pt",
                    padding=True
                ).to(device)
                
                # 模型推理
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # 解析结果
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1)
                pred_idx = probs.argmax().item()
                
                # 记录结果
                total += 1
                if pred_idx == q["correct_idx"]:
                    correct += 1
                    
                results.append({
                    "id": q["id"],
                    "options": q["options"],
                    "predicted": pred_idx,
                    "correct": q["correct_idx"],
                    "type": q["type"],
                    "concepts": {
                        "correct": q["options"][q["correct_idx"]],
                        "wrong": q["options"][1 - q["correct_idx"]]
                    }
                })
        except Exception as e:
            print(f"Error processing item: {e}")
            continue

    # 计算准确率
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct}/{total})")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "accuracy": f"{accuracy:.2f}%",
            "correct": correct,
            "total": total,
            "details": results
        }, f, indent=2)

if __name__ == "__main__":
    main()