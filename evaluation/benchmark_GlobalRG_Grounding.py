from datasets import load_dataset
import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from tqdm import tqdm
import json
import random
from collections import defaultdict
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run GlobalRG Grounding benchmark")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to test")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cache_dir", type=str, default="/data/yuchen/huggingface", help="Cache directory for datasets")
    parser.add_argument("--output_dir", type=str, default="/home/yuchen/project/cultureCLIP/evaluator/results", help="Base output directory")
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = os.path.join(args.output_dir, "benchmark_GlobalRG_Grounding")
    output_file = os.path.join(output_dir, f"{args.model_name.replace('/', '_')}.json")
    
    # 设置随机种子
    random.seed(args.random_seed)
    
    # Load the dataset (only test split)
    dataset = load_dataset("UBC-VL/globalrg-grounding-task", cache_dir=args.cache_dir)["test"]
    
    # Filter out invalid images
    valid_indices = []
    for idx in range(len(dataset)):
        try:
            image = dataset[idx]["image"]  # Ensure this is a PIL.Image.Image object
            image.verify()  # Verify the image is valid
            valid_indices.append(idx)  # Keep track of valid indices
        except (IOError, ValueError, RuntimeError) as e:
            print(f"Error with image at index {idx}: {e}. Skipping...")
    
    # Create a filtered dataset with only valid images
    filtered_dataset = dataset.select(valid_indices)
    
    # Load the CLIP model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForZeroShotImageClassification.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model.to(device)
    
    # Build a mapping of regions to concepts from the filtered dataset
    region_concepts = defaultdict(set)
    
    for idx in range(len(filtered_dataset)):
        region = filtered_dataset[idx]["region"]
        concept = filtered_dataset[idx]["concept"]
        region_concepts[region].add(concept)
    
    # Filter regions with fewer than 4 concepts
    valid_regions = {region for region, concepts in region_concepts.items() if len(concepts) >= 4}
    
    # Generate options and evaluate the model
    results = []
    total_questions = 0
    correct_answers = 0
    
    # Iterate through the filtered dataset for inference
    for idx in tqdm(range(len(filtered_dataset))):
        if filtered_dataset[idx]["region"] not in valid_regions:
            continue
    
        try:
            # Get the image object (already a PIL.Image.Image)
            image = filtered_dataset[idx]["image"]  # Ensure this is a PIL.Image.Image object
    
            # If the image is valid, proceed with evaluation
            region = filtered_dataset[idx]["region"]
            concept = filtered_dataset[idx]["concept"]
    
            # Generate options
            other_concepts = list(region_concepts[region] - {concept})
            if len(other_concepts) < 3:
                continue
    
            wrong_concepts = random.sample(other_concepts, 3)
            options = [f"The item in the picture is {concept} in {region}"] + \
                      [f"The item in the picture is {c} in {region}" for c in wrong_concepts]
            random.shuffle(options)
            correct_idx = options.index(f"The item in the picture is {concept} in {region}")
    
            # Prepare inputs
            inputs = processor(
                text=options,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)
    
            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            pred_idx = probs.argmax(dim=1).item()
    
            # Check if the prediction is correct
            total_questions += 1
            if pred_idx == correct_idx:
                correct_answers += 1
    
            results.append({
                "region": region,
                "concept": concept,
                "options": options,
                "predicted_idx": pred_idx,
                "correct_idx": correct_idx
            })
    
        except (IOError, ValueError) as e:
            print(f"Error processing item {idx}: {e}. Skipping...")
            continue
    
    # Calculate accuracy
    accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    print(f"Accuracy: {accuracy:.2f}% ({correct_answers}/{total_questions})")
    
    # Save results to a file
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "accuracy": f"{accuracy:.2f}%",
            "correct": correct_answers,
            "total": total_questions,
            "details": results
        }, f, indent=4)

if __name__ == "__main__":
    main()