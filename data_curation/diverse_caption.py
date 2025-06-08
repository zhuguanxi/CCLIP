import os
import json
import random
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm

# Set cache directories
cache_dir = "/data/yuchen/huggingface/"

def clean_text(text):
    """Clean generated text by removing extra spaces and newlines."""
    return ' '.join(text.strip().split())

def generate_diverse_captions(model, processor, concept, context, visual_features):
    """Generate 10 diverse captions for a given concept."""
    
    prompt_template = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": None}
            ]
        }
    ]

    text_template = (
        "Given a cultural concept, context, and key visual features, your task is to generate 10 different captions that describe the concept in various scenarios while preserving its cultural significance. The captions must reflect different styles or settings, but they should all clearly include the key visual features. Follow these guidelines:\n\n"
        "- Emphasize the key visual differences in the scenes (e.g., shape, size, cooking method, setting).\n"
        "- Retain the cultural or functional context (e.g., everyday use, ceremonial purpose, tradition).\n"
        "- Ensure the differentiating features (e.g., shape, texture, size, material, use) are clearly reflected in each caption.\n"
        "- Each caption should be under 15 words.\n"
        "- Each caption should be unique, showing different perspectives or settings, but should always include the key visual features.\n\n"
        "Examples:\n"
        "Input:\n"
        "Concept: Xiaolongbao (Soup Dumplings)\n"
        "Context: Steamed soup dumplings with delicate, thin wrappers, filled with savory broth and pork, typically steamed in bamboo baskets.\n"
        "Key Visual Features: Delicate, thin wrappers and steamed in bamboo baskets.\n\n"
        "Output:\n"
        "1. A chef carefully places Xiaolongbao in a bamboo steamer, showcasing their thin, translucent wrappers and savory broth inside.\n"
        "2. A steaming bamboo basket of Xiaolongbao, delicate wrappers holding savory broth, served in an elegant Shanghai restaurant.\n"
        "3. Steamed xiaolongbao resting in bamboo baskets, ready to be served during a family meal.\n"
        "4. Crispy fried xiaolongbao, golden-brown and served with dipping sauce, sitting in bamboo baskets.\n"
        "5. Miniature xiaolongbao filled with crab roe, elegantly presented in bamboo baskets at a Cantonese restaurant.\n"
        "6. Steaming xiaolongbao with delicate skin, served in bamboo baskets during a traditional Chinese New Year meal.\n"
        "7. Bamboo-steamed xiaolongbao, filled with savory broth, served alongside hot tea in a Beijing teahouse.\n"
        "8. Translucent, plump xiaolongbao, freshly steamed in bamboo baskets for a cozy brunch setting.\n"
        "9. Steamed xiaolongbao with pork filling, served in bamboo baskets with chili oil at a street food stall.\n"
        "10. Elegant xiaolongbao, arranged in bamboo baskets, presented at a lavish festive feast.\n\n"
        "Current Task:\n"
        f"Concept: {concept}\n"
        f"Context: {context}\n"
        f"Key Visual Features: {visual_features}\n\n"
        "Generate 10 different captions, each reflecting a different style or scene, but all incorporating the key visual features:"
    )

    prompt = prompt_template.copy()
    prompt[0]["content"][0]["text"] = text_template
    
    text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(prompt)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    ).to(model.device)
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )
    
    full_text = processor.batch_decode(
        [generated_ids[0][len(inputs.input_ids[0]):]], 
        skip_special_tokens=True
    )[0]
    
    # Parse the generated captions
    captions = []
    for line in full_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() and '. ' in line:
            caption = line.split('. ', 1)[1].strip()
            captions.append(caption)
    
    # If we didn't get exactly 10 captions, pad or truncate
    while len(captions) < 10:
        captions.append(captions[-1] if captions else "")
    captions = captions[:10]
    
    # Randomly select one caption as the main caption
    main_caption = random.choice(captions)
    
    return {
        "main_caption": main_caption,
        "all_captions": captions
    }

def generate_diverse_pairs(input_file, output_file):
    """Generate diverse captions for twin concept pairs."""
    
    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        cache_dir=cache_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Load input pairs
    pairs = []
    with open(input_file, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    
    print(f"Loaded {len(pairs)} concept pairs")
    
    # Generate diverse captions for each pair
    with open(output_file, 'w') as f:
        for pair in tqdm(pairs, desc="Generating diverse captions"):
            # Generate captions for positive concept
            pos_captions = generate_diverse_captions(
                model,
                processor,
                pair['pos_concept'],
                pair['pos_context'],
                pair['pos_features']
            )
            
            # Generate captions for negative concept
            neg_captions = generate_diverse_captions(
                model,
                processor,
                pair['neg_concept'],
                pair['neg_context'],
                pair['neg_features']
            )
            
            # Create output with all fields
            output = {
                "category": pair['category'],
                "pos_concept": pair['pos_concept'],
                "pos_context": pair['pos_context'],
                "pos_features": pair['pos_features'],
                "pos_caption": pos_captions['main_caption'],
                "pos_all_captions": pos_captions['all_captions'],
                "neg_concept": pair['neg_concept'],
                "neg_context": pair['neg_context'],
                "neg_features": pair['neg_features'],
                "neg_caption": neg_captions['main_caption'],
                "neg_all_captions": neg_captions['all_captions']
            }
            
            json.dump(output, f)
            f.write("\n")
    
    print(f"Generated diverse captions saved to {output_file}")

if __name__ == "__main__":
    input_file = "demo_twin_pairs.jsonl"
    output_file = "demo_diverse_captions.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        exit(1)
        
    generate_diverse_pairs(input_file, output_file)
