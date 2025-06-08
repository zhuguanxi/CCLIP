import os
import json
from tqdm import tqdm

def adapt_for_stable_diffusion(input_file, output_file):
    """Convert diverse caption format to stable diffusion input format with sample IDs."""
    
    # Load input data
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(data)} samples from {input_file}")
    
    # Convert format
    adapted_data = []
    for idx, item in enumerate(tqdm(data, desc="Converting format")):
        # Create new entry with sample ID
        sample_id = f"id_{idx:06d}"
        
        # For each sample, create entries for each caption pair
        # Now we use nested loops to get all combinations of pos and neg captions
        for pos_caption_idx, pos_caption in enumerate(item["pos_all_captions"]):
            for neg_caption_idx, neg_caption in enumerate(item["neg_all_captions"]):
                adapted_item = {
                    "sample_id": sample_id,
                    "pos_caption_id": pos_caption_idx,
                    "neg_caption_id": neg_caption_idx,
                    "pos_concept": item["pos_concept"],
                    "pos_context": item["pos_context"],
                    "pos_caption": pos_caption,
                    "neg_concept": item["neg_concept"],
                    "neg_context": item["neg_context"],
                    "neg_caption": neg_caption,
                    "metadata": {
                        "category": item["category"],
                        "pos_features": item["pos_features"],
                        "neg_features": item["neg_features"]
                    }
                }
                adapted_data.append(adapted_item)
    
    # Save adapted data
    with open(output_file, 'w') as f:
        for item in adapted_data:
            json.dump(item, f)
            f.write("\n")
    
    print(f"Converted {len(adapted_data)} caption pairs saved to {output_file}")
    print(f"Generated {len(data)} unique sample IDs")
    print(f"Each sample has {len(data[0]['pos_all_captions'])} × {len(data[0]['neg_all_captions'])} = {len(data[0]['pos_all_captions']) * len(data[0]['neg_all_captions'])} caption pairs")

if __name__ == "__main__":
    input_file = "../demo_diverse_captions.jsonl"
    output_file = "sd_input.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        exit(1)
        
    adapt_for_stable_diffusion(input_file, output_file)
