import os
import json
import jsonlines
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
from io import BytesIO
import re

# cache_dir = "/data/yuchen/huggingface/"
cache_dir = "/mnt/M3_Lab/hsi/huggingface_cache/"
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_dir)
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_dir, device_map="auto", torch_dtype="auto")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define culture categories
culture_category = {
    "Cuisine": "Refers to the foods, culinary practices, and cooking methods that are unique to specific regions or cultures. This includes iconic dishes, preparation techniques, and the cultural background behind eating habits, as well as the importance of food in social and religious practices.",
    
    "Clothing": "Encompasses traditional garments, accessories, and adornments from various cultures. It includes not only clothing but also items like jewelry, headwear, and footwear that hold cultural significance, reflecting identity, status, and traditions.",
    
    #"Animal & Plants": "Describes the native species, both fauna and flora, that hold cultural importance. This category includes the use of animals and plants in mythology, cuisine, traditional medicine, and environmental practices, as well as their roles in folklore and symbolism.",
    
    "Art": "Includes visual arts, sculptures, and other forms of artistic expression that represent a culture's aesthetic and artistic heritage. This encompasses paintings, sculptures, performance arts, and crafts that reflect the identity, beliefs, and historical evolution of a community.",
    
    "Architecture": "Refers to the design, style, and structures built by a particular culture. This includes traditional houses, temples, monuments, and public buildings that showcase the engineering, material use, and aesthetic values of the culture.",
    
    #"Daily Life": "Covers the everyday activities, routines, and practices that define how people in a particular culture live. This includes family roles, work habits, and leisure activities, as well as practices around health, education, and community.",
    
    "Symbol": "Involves the symbols, logos, and imagery that carry cultural meaning. This category includes national flags, religious icons, mythological figures, and colors that convey beliefs, values, and identity in various contexts.",
    
    "Festival": "Encompasses cultural festivals, holidays, and ceremonies, along with the associated customs, rituals, and practices. Examples include events like Chinese New Year, Diwali, and Christmas, each rich in traditions, foods, and rituals that symbolize community and heritage."
}

# Define valid categories
VALID_CATEGORIES = {
    "Cuisine", "Clothing", "Art", 
    "Architecture", "Symbol", "Festival"
}

# Define prompt templates
prompt_template = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": None},
            {"type": "image_url", "image_url": None}
        ]
    }
]

# Format category descriptions for the prompt
category_descriptions = "\n".join([f"- {cat}: {desc}" for cat, desc in culture_category.items()])

text_template = (
    "Given a cultural concept, definition, definition caption, and definition image, your task is to classify and extract the following information in English:\n\n"
    "- Country: The country or region associated with the concept.\n"
    "- Cultural Category: IMPORTANT - You MUST choose exactly ONE category from these eight predefined categories:\n"
    "  * Cuisine\n"
    "  * Clothing\n"
    # "  * Animal & Plants\n"
    "  * Art\n"
    "  * Architecture\n"
    # "  * Daily Life\n"
    "  * Symbol\n"
    "  * Festival\n"
    "  Do not use any other categories. If the concept doesn't clearly fit into one of these categories, choose the closest match.\n"
    "- Context: A short 20-word description in English that provides insight into the cultural and functional use of the concept.\n"
    "- Key Visual Features: The visual features that distinguish the concept (e.g., shape, material, color, size) described in English.\n\n"
    "Examples:\n"
    "Input:\n"
    "Concept: Kimono\n"
    "Definition: A traditional Japanese garment with long, wide sleeves and an obi sash, worn for formal occasions and festivals.\n"
    "Definition Caption: A colorful kimono with crane patterns, displayed at Kyoto's textile museum.\n"
    "Definition Image: example_image.jpg\n"
    "Output:\n"
    "Country: Japan\n"
    "Category: Clothing\n"
    "Context: A traditional Japanese garment with long, wide sleeves and an obi sash, worn for formal occasions and festivals.\n"
    "Key Visual Features: Long, wide sleeves, obi sash, colorful patterns, traditional Japanese style.\n\n"
    "Current Task:\n"
    "Concept: {concept}\n"
    "Definition: {definition}\n"
    "Definition Caption: {caption}\n"
    "Definition Image: {image_url}\n\n"
    "Generate the Country, Cultural Category (MUST be one of the eight predefined categories), Context, and Key Visual Features in English:"
)

def preprocess_image(image, max_size=1024):
    """Resize image while maintaining aspect ratio."""
    if image is None:
        return None
        
    # Calculate new dimensions while maintaining aspect ratio
    ratio = min(max_size / image.width, max_size / image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    
    # Resize image
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    return resized_image

def load_image_from_url(image_url):
    """Load image from URL and return PIL Image object."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return preprocess_image(image)
    except Exception as e:
        print(f"Error loading image from {image_url}: {e}")
        return None

def analyze_concepts_batch(items):
    """Batch analyze cultural concepts using Qwen model."""
    prompts = []
    images = []
    valid_indices = []

    print(f"\nProcessing batch of {len(items)} items...")

    # Prepare prompts and images
    for i, item in enumerate(items):
        print(f"\nProcessing item {i}:")
        print(f"Title: {item['page_title']}")
        print(f"Image URL: {item['image_url']}")
        
        image = load_image_from_url(item['image_url'])
        if image is None:
            print(f"Failed to load image for item {i}")
            continue

        # Format text prompt
        text = text_template.format(
            concept=item['page_title'],
            definition=item['context_page_description'],
            caption=item['caption_attribution_description'],
            image_url=item['image_url']
        )

        # Create prompt using template
        prompt = prompt_template.copy()
        prompt[0]["content"][0]["text"] = text
        prompt[0]["content"][1]["image_url"] = item['image_url']

        prompts.append(prompt)
        images.append(image)
        valid_indices.append(i)

    if not prompts:
        print("No valid prompts generated!")
        return [], []

    print(f"\nProcessing {len(prompts)} prompts with model...")

    # Process inputs in batch
    texts = [
        processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        for message in prompts
    ]

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)

    # Generate results with adjusted parameters
    print("Generating model outputs...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,  # Increase max tokens for longer outputs
        min_new_tokens=32,   # Ensure minimum output length
        do_sample=True,      # Enable sampling
        temperature=0.7,     # Slightly lower temperature for more focused outputs
        top_p=0.9,          # Nucleus sampling
        repetition_penalty=1.2,  # Prevent repetition
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    # Decode outputs
    responses = processor.batch_decode(outputs, skip_special_tokens=True)
    print(f"\nGot {len(responses)} responses from model")

    # Process responses
    results = []
    for idx, response in zip(valid_indices, responses):
        try:
            print(f"\nProcessing response for item {idx}:")
            print("Raw response:")
            print(response)
            print("-" * 50)

            # Initialize variables
            country = None
            category = None
            context = None
            visual_features = None

            # Split response into lines and clean
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Find the start of the actual response (after the prompt and examples)
            start_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('Country:') and not any('Japan' in l for l in lines[i:i+4]):
                    start_idx = i
                    break
            
            # Only process lines from the start of the actual response
            response_lines = lines[start_idx:]
            
            # Process each field
            current_field = None
            field_content = []
            
            for line in response_lines:
                if line.startswith('Country:'):
                    if current_field and field_content:
                        if current_field == 'Country':
                            country = ' '.join(field_content)
                        elif current_field == 'Category':
                            category = ' '.join(field_content)
                        elif current_field == 'Context':
                            context = ' '.join(field_content)
                        elif current_field == 'Visual Features':
                            visual_features = ' '.join(field_content)
                        field_content = []
                    current_field = 'Country'
                    field_content.append(line.replace('Country:', '').strip())
                elif line.startswith('Category:') or line.startswith('Cultural Category:'):
                    if current_field and field_content:
                        if current_field == 'Country':
                            country = ' '.join(field_content)
                        elif current_field == 'Category':
                            category = ' '.join(field_content)
                        elif current_field == 'Context':
                            context = ' '.join(field_content)
                        elif current_field == 'Visual Features':
                            visual_features = ' '.join(field_content)
                        field_content = []
                    current_field = 'Category'
                    field_content.append(line.replace('Category:', '').replace('Cultural Category:', '').strip())
                elif line.startswith('Context:'):
                    if current_field and field_content:
                        if current_field == 'Country':
                            country = ' '.join(field_content)
                        elif current_field == 'Category':
                            category = ' '.join(field_content)
                        elif current_field == 'Context':
                            context = ' '.join(field_content)
                        elif current_field == 'Visual Features':
                            visual_features = ' '.join(field_content)
                        field_content = []
                    current_field = 'Context'
                    field_content.append(line.replace('Context:', '').strip())
                elif line.startswith('Key Visual Features:'):
                    if current_field and field_content:
                        if current_field == 'Country':
                            country = ' '.join(field_content)
                        elif current_field == 'Category':
                            category = ' '.join(field_content)
                        elif current_field == 'Context':
                            context = ' '.join(field_content)
                        elif current_field == 'Visual Features':
                            visual_features = ' '.join(field_content)
                        field_content = []
                    current_field = 'Visual Features'
                    field_content.append(line.replace('Key Visual Features:', '').strip())
                elif current_field:
                    field_content.append(line)
            
            # Save the last field
            if current_field and field_content:
                if current_field == 'Country':
                    country = ' '.join(field_content)
                elif current_field == 'Category':
                    category = ' '.join(field_content)
                elif current_field == 'Context':
                    context = ' '.join(field_content)
                elif current_field == 'Visual Features':
                    visual_features = ' '.join(field_content)

            # Validate category
            if category:
                # Clean and normalize category
                category = category.strip()
                # Try to find the closest valid category
                if category not in VALID_CATEGORIES:
                    print(f"Invalid category '{category}' for item {idx}, attempting to map to valid category...")
                    # Simple mapping for common variations
                    category_mapping = {
                        "Mythology": "Symbol",
                        "Myth": "Symbol",
                        "Religion": "Symbol",
                        "Religious": "Symbol",
                        "Ceremony": "Festival",
                        "Celebration": "Festival",
                        "Food": "Cuisine",
                        "Dish": "Cuisine",
                        "Meal": "Cuisine",
                        "Garment": "Clothing",
                        "Costume": "Clothing",
                        "Dress": "Clothing",
                        "Building": "Architecture",
                        "Structure": "Architecture",
                        "Painting": "Art",
                        "Sculpture": "Art",
                        "Craft": "Art",
                        # "Creature": "Animal & Plants",
                        # "Plant": "Animal & Plants",
                        # "Flora": "Animal & Plants",
                        # "Fauna": "Animal & Plants",
                        # "Custom": "Daily Life",
                        # "Tradition": "Daily Life",
                        # "Practice": "Daily Life"
                    }
                    category = category_mapping.get(category, "Symbol")  # Default to Symbol if no mapping found
                    print(f"Mapped category to '{category}'")

            # Check if essential fields were found
            if not all([country, category, context, visual_features]):
                print(f"Failed to extract essential fields from response for item {idx}")
                print(f"Extracted fields: Country={country}, Category={category}, Context={context}, Visual Features={visual_features}")
                continue

            # Verify category is valid
            if category not in VALID_CATEGORIES:
                print(f"Invalid category '{category}' for item {idx} after mapping")
                continue

            # Add category description to metadata
            metadata = {
                "category_description": culture_category.get(category, "Detailed description of the cultural category and its significance in the cultural context.")
            }

            # Create output dictionary
            output = {
                "country": country,
                "category": category,
                "concept": items[idx]['page_title'],
                "context": context,
                "visual_features": visual_features,
                "metadata": metadata
            }
            results.append((idx, output))
            print(f"Successfully processed item {idx}")
        except Exception as e:
            print(f"Error processing response for item {idx}: {e}")
            print(f"Response: {response}")
            continue

    print(f"\nSuccessfully processed {len(results)} items out of {len(valid_indices)} valid items")
    return valid_indices, results

def process_dataset(input_file, output_file, progress_file='classification_progress.txt', batch_size=4):
    """Process the dataset and classify cultural concepts."""
    processed_count = 0

    print(f"\nStarting dataset processing...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Progress file: {progress_file}")

    # Check progress file
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as progress_f:
            processed_count = int(progress_f.read().strip())
        print(f"Resuming from processed count: {processed_count}")

    # Read input file
    print(f"\nReading input file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        items = list(jsonlines.Reader(f))
    
    total_items = len(items)
    print(f"Found {total_items} items in input file")
    
    # Process items in batches
    with open(output_file, 'a', encoding='utf-8') as out_f:
        for i in range(processed_count, total_items, batch_size):
            print(f"\nProcessing batch starting at index {i}")
            batch_items = items[i:i + batch_size]
            
            # Analyze batch
            valid_indices, results = analyze_concepts_batch(batch_items)
            
            # Write results
            for idx, result in results:
                jsonlines.Writer(out_f).write(result)
                out_f.flush()
                print(f"Wrote result for item {idx}")
            
            # Update progress
            with open(progress_file, 'w') as progress_f:
                progress_f.write(str(i + batch_size))
                progress_f.flush()
            
            print(f"Processed {min(i + batch_size, total_items)} / {total_items} items...")
            
            # Clear GPU memory
            torch.cuda.empty_cache()

    print(f"\nClassification completed. Results saved to {output_file}")

if __name__ == "__main__":
    process_dataset(
        input_file='demo_raw_data.jsonl',
        output_file='demo_classified_concepts.jsonl'
    )
