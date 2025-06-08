import os
import json
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm

# Set cache directories
cache_dir = "/data/yuchen/huggingface/"

def clean_text(text):
    """Clean generated text by removing extra spaces and newlines."""
    return ' '.join(text.strip().split())

def generate_twin_concept(model, processor, category, concept, context, visual_features):
    """Generate a similar-looking but culturally different concept."""
    
    prompt_template = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": None}
            ]
        }
    ]

    text_template = (
        "Given a cultural concept from a certain category, its context and its visual features, your task is to generate the following:\n"
        "- New Concept: A visually similar but culturally different concept from the same category.\n"
        "- New Context: A short 20-word description in English that provides insight into the cultural and functional use of the generatedconcept.\n"
        "- New Key Visual Features: The visual features that distinguish the concept (e.g., shape, material, color, size) from the original concept.\n\n"
        "Examples:\n"
        "Input:\n"
        "Category: Art\n"
        "Concept: Erhu\n"
        "Context: A two-stringed Chinese musical instrument played with a bow, often used in traditional Chinese music.\n"
        "Key Visual Features: Two strings, a bow, a wooden body, and a horsehair bow.\n\n"
        "Output:\n"
        "New Concept: Guzheng\n"
        "New Context: A Chinese zither-like instrument with a large, rectangular wooden body and multiple strings, played with plucking.\n"
        "New Key Visual Features: A large, rectangular wooden body, multiple strings, and a plucking mechanism.\n\n"
        "Current Task:\n"
        f"Category: {category}\n"
        f"Concept: {concept}\n"
        f"Context: {context}\n"
        f"Visual Features: {visual_features}\n\n"
        "Generate a new concept, context and visual features:\n"
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
    
    # Parse the generated text
    new_concept = ""
    new_context = ""
    new_visual_features = ""
    
    for line in full_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('New Concept:'):
            new_concept = line.replace('New Concept:', '').strip()
        elif line.startswith('New Context:'):
            new_context = line.replace('New Context:', '').strip()
        elif line.startswith('New Visual Features:'):
            new_visual_features = line.replace('New Visual Features:', '').strip()
    
    return {
        "concept": new_concept,
        "context": new_context,
        "visual_features": new_visual_features
    }

def generate_twin_pairs(input_file, output_file):
    """Generate twin concept pairs from the input concepts."""
    
    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        cache_dir=cache_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Load input concepts
    concepts = []
    with open(input_file, 'r') as f:
        for line in f:
            concepts.append(json.loads(line))
    
    print(f"Loaded {len(concepts)} concepts")
    
    # Generate twin pairs
    with open(output_file, 'w') as f:
        for concept in tqdm(concepts, desc="Generating twin pairs"):
            # Generate twin concept
            twin = generate_twin_concept(
                model, 
                processor,
                concept['category'],
                concept['concept'],
                concept['context'],
                concept['visual_features']
            )
            
            # Create output pair
            output = {
                "category": concept['category'],
                "pos_concept": concept['concept'],
                "pos_context": concept['context'],
                "pos_features": concept['visual_features'],
                "neg_concept": twin['concept'],
                "neg_context": twin['context'],
                "neg_features": twin['visual_features']
            }
            
            json.dump(output, f)
            f.write("\n")
    
    print(f"Generated twin pairs saved to {output_file}")

if __name__ == "__main__":
    input_file = "./top_down/demo_concepts.jsonl"
    output_file = "demo_twin_pairs.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        exit(1)
        
    generate_twin_pairs(input_file, output_file)
