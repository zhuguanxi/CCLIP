import os
# 強制讓 os.chmod 變成什麼都不做的函數，避開 /mnt 權限錯誤
def mock_chmod(path, mode, *args, **kwargs):
    pass
os.chmod = mock_chmod

import os
import re
import ast
import json
import jsonlines
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

# cache_dir = "/data/yuchen/huggingface/"
cache_dir = "/mnt/M3_Lab/hsi/huggingface_cache/"
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_dir)
processor.tokenizer.padding_side = 'left'

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_dir, device_map={"": 0}, torch_dtype=torch.bfloat16, trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# Extract the dictionary from the text output
def extract_and_convert(text_output):
    match = re.search(r'{(.*)}', text_output, re.DOTALL)
    if match:
        dict_content = match.group(0).strip()
        
        # 先将双引号转为 $$，避免后续的替换冲突
        temp_content = re.sub(r'"', '$$', dict_content)

        formatted_text = temp_content.replace("\'s", '##s')
        
        # 将单引号转义符 \' 替换为特殊标记 @@
        temp_content = temp_content.replace("\\'", "@@") 

        # 将单引号转为双引号
        formatted_text = temp_content.replace("'", '"')
        
        # 再把 $$ 转回单引号
        formatted_text = formatted_text.replace('$$', "'")

        # 最后将 @@ 转回 \'
        formatted_text = formatted_text.replace("@@", "\\'")

        formatted_text = formatted_text.replace("##s", "\'s")

        #print('dict_content:', formatted_text)
        try:
            result_dict = ast.literal_eval(formatted_text)
            return result_dict
        except (SyntaxError, ValueError) as e:
            print("Error converting text to dictionary:", e)
            return None
    else:
        print("No dictionary found in the text.")
        return None
 
def analyze_page_titles_batch(titles):
    """
    Batch analyze page titles using Qwen model to determine if they are cultural concepts.
    """
    prompts = [
        f"""
        Please determine whether the concept '{title}' clearly and unambiguously belongs to one of the six cultural categories (Cuisine, Clothing, Art, Architecture, Symbol, or Festival). If the concept is only loosely related, culturally ambiguous, or does not strongly align with any of the categories, please select 'A' to ensure strict filtering. The concept does not clearly belong to any of the eight categories. The concept is clearly and directly related to one of the eight categories. Answer only with 'A' or 'B'.

        Output the result in JSON format as follows:
        {{
            "concept_type": "A" or "B",
        }}
        """
        for title in titles
        '''
        f"""
        Please determine whether the concept '{title}' clearly and unambiguously belongs to one of the SIX cultural categories: 
        1. Cuisine (Food/Cooking)
        2. Clothing (Traditional garments/Accessories)
        3. Art (Fine arts/Traditional crafts)
        4. Architecture (Buildings/Structures)
        5. Symbol (Religious icons, mythological figures, cultural symbols)
        6. Festival (Holidays/Ceremonies).

        If the concept is loosely related, modern, or does not strongly align with these traditional cultural categories, select 'A'. 
        If it is a clear, culturally significant concept from the 6 categories above, select 'B'.

        Answer ONLY in the following JSON format:
        {{
            "concept_type": "A" or "B"
        }}
        """
        '''
    ]

    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        for prompt in prompts
    ]

    # Process inputs in batch
    texts = [
        processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        for message in messages
    ]

    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)

    # Generate results
    outputs = model.generate(**inputs, max_new_tokens=128)

    # Decode outputs
    responses = processor.batch_decode(outputs, skip_special_tokens=True)

    # Extract JSON from responses
    results = []
    for response_text in responses:
        match = re.search(r"assistant\s+({.*?})", response_text, re.DOTALL)
        if match:
            assistant_response = match.group(1)
            results.append(assistant_response.strip())
        else:
            results.append(None)
    
    return results


def process_wit_dataset(output_file, progress_file='wit_progress.txt', batch_size=16):
    """
    Process WIT dataset to filter cultural concepts.
    """
    processed_count = 0

    # Load WIT dataset
    # ds = load_dataset("google/wit")
    ds = load_dataset(
        "google/wit", 
        cache_dir="/mnt/M3_Lab/hsi/huggingface_cache/datasets", 
        trust_remote_code=True,
        keep_in_memory=False
    )
    train_dataset = ds['train']

    # Check progress file
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as progress_f:
            processed_count = int(progress_f.read().strip())

    total_samples = len(train_dataset)  # 3700萬筆資料
    limit_samples = 100000  
    
    end_index = min(total_samples, limit_samples)

    with open(output_file, 'a', encoding='utf-8') as out_f:
        for i in range(processed_count, end_index, batch_size):
            batch_data = train_dataset[i:i + batch_size]
            titles = batch_data['page_title']

            # Analyze titles
            results = analyze_page_titles_batch(titles)
            
            keys = batch_data.keys()
            batch_items = [
                {key: batch_data[key][j] for key in keys} 
                for j in range(len(titles))
            ]

            for item, result in zip(batch_items, results):
                if not result:
                    continue
                    
                result_dict = extract_and_convert(result)
                concept_type = result_dict.get("concept_type", "Unknown")

                if concept_type == "B":  # Cultural concept
                    # Create a filtered entry with relevant fields
                    filtered_entry = {
                        "page_title": item["page_title"],
                        "page_url": item["page_url"],
                        "image_url": item["image_url"],
                        "caption_reference_description": item["caption_reference_description"],
                        "caption_attribution_description": item["caption_attribution_description"],
                        "caption_alt_text_description": item["caption_alt_text_description"],
                        "context_page_description": item["context_page_description"],
                        "context_section_description": item["context_section_description"],
                        "language": item["language"],
                        "mime_type": item["mime_type"],
                        "original_height": item["original_height"],
                        "original_width": item["original_width"],
                        "is_main_image": item["is_main_image"]
                    }
                    jsonlines.Writer(out_f).write(filtered_entry)

            # Update progress
            with open(progress_file, 'w') as progress_f:
                progress_f.write(str(i + batch_size))
                progress_f.flush()
            print(f"Processed {min(i + batch_size, end_index)} / {end_index} entries...")
    
    print(f"Processing completed. Results saved to {output_file}")


if __name__ == "__main__":
    process_wit_dataset('wit_cultural_concepts.jsonl')
