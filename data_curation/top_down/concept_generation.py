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

def generate_cultural_concepts(input_file, output_file):
    """Generate cultural concepts for each country-category pair."""
    
    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        cache_dir=cache_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    prompt_template = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": None}
            ]
        }
    ]

    text_template = (
        "Given a country and cultural category, your task is to generate the following:\n\n"
        "- Concept: A cultural concept that fits the provided country and category.\n"
        "- Context: A short 20-word description that provides insight into the cultural and functional use of the concept.\n"
        "- Key Visual Features: The visual features that distinguish the concept (e.g., shape, material, color, size).\n\n"
        "Examples:\n"
        "Input:\n"
        "Country: China\n"
        "Cultural Category: Food\n"
        "Output:\n"
        "Concept: Mantou\n"
        "Context: Steamed wheat bun symbolizing prosperity and wisdom, commonly served during festivals and family gatherings.\n"
        "Key Visual Features: Pillowy white appearance, round or rectangular shape, smooth surface, typically palm-sized.\n\n"
        "Input:\n"
        "Country: Japan\n"
        "Cultural Category: Art\n"
        "Output:\n"
        "Concept: Ukiyo-e\n"
        "Context: Traditional woodblock prints depicting scenes from everyday life, nature, and historical events.\n"
        "Key Visual Features: Flat color blocks, bold outlines, vibrant pigments, rectangular format, detailed patterns.\n\n"
        "Current Task:\n"
        "Country: {country}\n"
        "Cultural Category: {category}\n\n"
        "Generate the concept, context, and key visual features:"
    )

    # Load input data
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} country-category pairs")
    
    # Process each sample
    with open(output_file, 'w') as f:
        for sample in tqdm(samples, desc="Generating cultural concepts"):
            country = sample['country']
            category = sample['category']
            
            # Generate concept
            prompt = prompt_template.copy()
            prompt[0]["content"][0]["text"] = text_template.format(country=country, category=category)
            
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
            
            # Decode and parse the output
            full_text = processor.batch_decode(
                [generated_ids[0][len(inputs.input_ids[0]):]], 
                skip_special_tokens=True
            )[0]
            
            # Parse the generated text
            concept = ""
            context = ""
            visual_features = ""
            
            current_section = None
            for line in full_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('Concept:'):
                    concept = line.replace('Concept:', '').strip()
                elif line.startswith('Context:'):
                    context = line.replace('Context:', '').strip()
                elif line.startswith('Key Visual Features:'):
                    visual_features = line.replace('Key Visual Features:', '').strip()
            
            # Save results
            output = {
                "country": country,
                "category": category,
                "concept": concept,
                "context": context,
                "visual_features": visual_features,
                "metadata": sample.get('metadata', {})
            }
            json.dump(output, f)
            f.write("\n")
    
    print(f"Generated concepts saved to {output_file}")

if __name__ == "__main__":
    # Use the pre-generated country-category pairs
    pairs_file = "demo_pairs.jsonl"
    # pairs_file = "country_category_pairs.jsonl"
    if not os.path.exists(pairs_file):
        print(f"Error: {pairs_file} not found. Please run country_taxonomy.py first to generate the pairs.")
        exit(1)
        
    # Generate cultural concepts using these pairs
    output_file = "demo_concepts.jsonl"
    # output_file = "generated_cultural_concepts.jsonl"
    generate_cultural_concepts(pairs_file, output_file)
