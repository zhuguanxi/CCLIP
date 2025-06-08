import os
import json
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
import datetime
import argparse

cache_dir = "/data/huggingface/"

def load_model():
    """Load the Qwen2.5-VL model for image-text matching."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        cache_dir=cache_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_dir)
    return model, processor

def image_resize(image):
    """Resize image to fit model requirements."""
    width, height = image.size
    pixels = width * height
    max_pixels = 640 * 28 * 28

    if pixels > max_pixels:
        scale_factor = (max_pixels / pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

# Define prompt templates for each criterion
AUTHENTICITY_TEMPLATE = """
Please analyze this image and rate its authenticity on a scale of 1 to 5. You can refer to the context to help you make the decision. Focus on whether the concept shown is realistic and follows common sense.

Consider:
- Are all elements anatomically and physically correct?
- Does everything look natural and possible in the real world?
- Are there any unrealistic or deformed features?

Examples:
Input: Erhu (二胡); An image showing a person playing Erhu with three hands in mid air;
Output: 1
Input: Erhu (二胡); An image showing a person playing Erhu with two hands sitting on the chair;
Output: 5

Now, give you the image, concept and its corresponding context:
Concept: {concept}
Context: {context}
Output only the score:
"""

CONSISTENCY_TEMPLATE = """
Please analyze this image and rate its consistency with the concept on a scale of 1 to 5. You can refer to the context to help you make the decision. Focus on whether the image accurately depicts the specified concept without showing wrong concepts.

Consider:
- Does the image show exactly the concept mentioned?
- Are there any mismatched or wrong elements?
- Is the concept clearly and accurately represented?

Examples:
Input: Tang Sancai (唐三彩); An image showing a blue-and-white porcelain (青花瓷) bowl;
Output: 1
Input: Tang Sancai (唐三彩); An image showing exactly a Tang Sancai (唐三彩) horse;
Output: 5

Now, give you the image, concept and its corresponding context:
Concept: {concept}
Context: {context}
Output only the score:
"""

CULTURAL_FIDELITY_TEMPLATE = """
Please analyze this image and rate its cultural fidelity on a scale of 1 to 5. Focus on whether the cultural elements are accurate and appropriate for the specific context.

Consider:
- Are all cultural elements accurate for this context?
- Are there any mixed or incorrect cultural elements?
- Does everything align with the cultural background specified?

Examples:
Input: Mexican Day of the Dead (墨西哥亡灵节); An image showing marigold flowers, skull paintings, and candle altars;
Output: 5
Input: Mexican Day of the Dead (墨西哥亡灵节); An image showing showing marigold flowers, skull paintings, and Chinese joss paper money (中式纸钱);
Output: 1

Now, give you the image, concept and its corresponding context:
Concept: {concept}
Context: {context}
Output only the score:
"""

def evaluate_batch(model, processor, image_paths, concepts, contexts, batch_size=1):
    """Evaluate a batch of images."""
    results = []
    total_images = len(image_paths)
    processed_images = 0
    
    print(f"Starting evaluation of {total_images} images...")
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_concepts = concepts[i:i + batch_size]
        batch_contexts = contexts[i:i + batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}: images {i+1}-{min(i+batch_size, total_images)}")
        
        # Load images
        batch_images = []
        valid_indices = []
        for idx, path in enumerate(batch_paths):
            try:
                image = Image.open(path)
                image = image_resize(image)
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                batch_images.append(image)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error opening image {path}: {e}")
                results.append({
                    'authenticity': 0,
                    'consistency': 0,
                    'cultural_fidelity': 0,
                    'passed': False,
                    'error': str(e)
                })
        
        if not batch_images:
            continue
            
        # Process each criterion for the batch
        scores = {idx: {} for idx in valid_indices}
        for criterion in ['authenticity', 'consistency', 'cultural_fidelity']:
            print(f"  Evaluating {criterion}...")
            
            # Prepare prompts for the batch
            batch_prompts = []
            for idx, image in zip(valid_indices, batch_images):
                # Get the appropriate template and format it
                if criterion == 'authenticity':
                    text = AUTHENTICITY_TEMPLATE.format(concept=batch_concepts[idx], context=batch_contexts[idx])
                elif criterion == 'consistency':
                    text = CONSISTENCY_TEMPLATE.format(concept=batch_concepts[idx], context=batch_contexts[idx])
                elif criterion == 'cultural_fidelity':
                    text = CULTURAL_FIDELITY_TEMPLATE.format(concept=batch_concepts[idx], context=batch_contexts[idx])
                
                # Create prompt in conversation format
                prompt_template = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": None},
                            {"type": "text", "text": None}
                        ]
                    }
                ]
                
                # Fill in the template
                prompt = prompt_template.copy()
                prompt[0]["content"][0]["image"] = image
                prompt[0]["content"][1]["text"] = text
                
                batch_prompts.append(prompt)
            
            try:
                # Process each image-prompt pair individually
                for idx_in_batch, (idx, prompt) in enumerate(zip(valid_indices, batch_prompts)):
                    print(f"    Processing image {i + idx_in_batch + 1}/{total_images} for {criterion}...")
                    
                    # Process the prompt using the same approach as in generate_captions.py
                    text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(prompt)
                    inputs = processor(
                        text=text,
                        images=image_inputs,
                        videos=video_inputs,
                        return_tensors="pt"
                    ).to(model.device)
                    
                    # Generate response
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50
                    )
                    
                    # Decode the response
                    response = processor.batch_decode(
                        [outputs[0][len(inputs.input_ids[0]):]], 
                        skip_special_tokens=True
                    )[0]
                    
                    # Clear CUDA cache after generation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Delete tensors explicitly
                    del inputs
                    del outputs
                    
                    # Parse score
                    try:
                        score = float(response.strip().split()[0])
                        # Ensure score is between 1 and 5
                        if score < 1:
                            score = 1
                        elif score > 5:
                            score = 5
                        else:
                            score = round(score)  # Round to nearest integer
                        scores[idx][criterion] = score
                        print(f"    Score: {score}")
                    except ValueError:
                        print(f"    Error parsing score for {criterion} from response: {response}")
                        scores[idx][criterion] = 0
                        
            except Exception as e:
                print(f"Error processing batch: {e}")
                for idx in valid_indices:
                    scores[idx][criterion] = 0
        
        # Clean up batch images
        for img in batch_images:
            img.close()
        del batch_images
        
        # Compile results for the batch
        for idx in valid_indices:
            # Calculate average score across all criteria
            avg_score = sum(scores[idx].values()) / len(scores[idx])

            # Check if any score is 1 (automatic fail) or if average is <= 3
            has_score_one = any(score == 1 for score in scores[idx].values())
            passed = not has_score_one and avg_score > 3
            
            results.append({
                **scores[idx],
                'avg_score': avg_score,
                'has_score_one': has_score_one,
                'passed': passed
            })
            processed_images += 1
            
        print(f"Processed {processed_images}/{total_images} images ({processed_images/total_images*100:.1f}%)")
            
        # Force garbage collection after each batch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"Evaluation complete. Processed {processed_images}/{total_images} images.")
    return results

def evaluate_all_pairs(input_jsonl, output_scored_jsonl, model, processor, batch_size=1, image_batch_size=8):
    """Evaluate all image pairs and save their scores to a new JSONL file.
    
    Args:
        input_jsonl: Path to input JSONL file containing image pairs
        output_scored_jsonl: Path to output JSONL file with added scores
        model: The VL model for evaluation
        processor: The processor for the VL model
        batch_size: Number of pairs to process before writing to output
        image_batch_size: Batch size for image evaluation within evaluate_batch
    """
    # Read all pairs from input file
    with open(input_jsonl, 'r') as f:
        pairs = [json.loads(line) for line in f]
    
    total_pairs = len(pairs)
    print(f"Total pairs to process: {total_pairs}")
    
    # Check if output file exists and count processed pairs
    processed_count = 0
    if os.path.exists(output_scored_jsonl):
        with open(output_scored_jsonl, 'r') as f:
            processed_count = sum(1 for _ in f)
        print(f"Found {processed_count} already processed pairs. Resuming from pair {processed_count + 1}...")
    
    # Create or append to the output file
    mode = 'a' if processed_count > 0 else 'w'
    
    # Process pairs in batches, skipping already processed ones
    for batch_start in range(processed_count, total_pairs, batch_size):
        batch_end = min(batch_start + batch_size, total_pairs)
        print(f"\nProcessing batch: pairs {batch_start+1}-{batch_end}")
        
        batch_pairs = pairs[batch_start:batch_end]
        
        # Prepare batches for positive and negative images in this batch
        pos_paths = [pair['pos_image_path'] for pair in batch_pairs]
        pos_concepts = [pair['pos_concept'] for pair in batch_pairs]
        pos_contexts = [pair['pos_context'] for pair in batch_pairs]
        
        neg_paths = [pair['neg_image_path'] for pair in batch_pairs]
        neg_concepts = [pair['neg_concept'] for pair in batch_pairs]
        neg_contexts = [pair['neg_context'] for pair in batch_pairs]
        
        # Process positive images for this batch
        print("Evaluating positive images...")
        pos_results = evaluate_batch(model, processor, pos_paths, pos_concepts, pos_contexts, image_batch_size)
        
        # Process negative images for this batch
        print("Evaluating negative images...")
        neg_results = evaluate_batch(model, processor, neg_paths, neg_concepts, neg_contexts, image_batch_size)
        
        # Add scores to pairs and write to output
        with open(output_scored_jsonl, mode) as f:
            for i, (pair, pos_scores, neg_scores) in enumerate(zip(batch_pairs, pos_results, neg_results)):
                # Add scores to the pair
                pair['pos_scores'] = pos_scores
                pair['neg_scores'] = neg_scores
                
                # Write to output file
                json.dump(pair, f)
                f.write('\n')
        
        # Switch to append mode after first batch
        mode = 'a'
        
        # Print batch statistics
        print(f"\n----- Batch {batch_start//batch_size + 1} Statistics -----")
        print(f"Batch pairs processed: {len(batch_pairs)}")
        print(f"Pairs written to: {output_scored_jsonl}")
        print("-------------------------------------\n")
        
        # Force garbage collection after each batch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\nEvaluation complete. All scored pairs saved to: {output_scored_jsonl}")

def filter_scored_pairs(input_scored_jsonl, output_filtered_jsonl):
    """Filter pairs based on their scores and save filtered pairs to a new JSONL file.
    
    Args:
        input_scored_jsonl: Path to input JSONL file with scores
        output_filtered_jsonl: Path to output JSONL file for filtered pairs
    """
    # Initialize statistics
    stats = {
        'total_pairs': 0,
        'passed_pairs': 0,
        'failed_pairs': 0,
        'failed_score_one': 0,
        'failed_low_avg': 0,
        'error_pairs': 0
    }
    
    # For calculating standard deviation and averages
    all_scores = {
        'authenticity': [],
        'consistency': [],
        'cultural_fidelity': []
    }
    
    filtered_scores = {
        'authenticity': [],
        'consistency': [],
        'cultural_fidelity': []
    }
    
    # Read all scored pairs
    print(f"Reading scored pairs from {input_scored_jsonl}...")
    all_pairs = []
    with open(input_scored_jsonl, 'r') as f:
        for line in f:
            all_pairs.append(json.loads(line))
    
    # Process and filter pairs
    filtered_pairs = []
    for pair in all_pairs:
        stats['total_pairs'] += 1
        
        # Check for errors
        if ('error' in pair.get('pos_scores', {}) or 'error' in pair.get('neg_scores', {})):
            stats['error_pairs'] += 1
            continue
        
        # Calculate pair scores for each criterion (average of pos and neg)
        for criterion in ['authenticity', 'consistency', 'cultural_fidelity']:
            pos_score = pair['pos_scores'].get(criterion, 0)
            neg_score = pair['neg_scores'].get(criterion, 0)
            pair_criterion_score = (pos_score + neg_score) / 2
            
            # Add to all scores
            all_scores[criterion].append(pair_criterion_score)
        
        # Check if passed
        pos_passed = pair['pos_scores'].get('passed', False)
        neg_passed = pair['neg_scores'].get('passed', False)
        
        # Check reason for failure
        pos_has_one = pair['pos_scores'].get('has_score_one', False)
        neg_has_one = pair['neg_scores'].get('has_score_one', False)
        
        if pos_passed and neg_passed:
            stats['passed_pairs'] += 1
            filtered_pairs.append(pair)
            
            # Add to filtered scores
            for criterion in ['authenticity', 'consistency', 'cultural_fidelity']:
                pos_score = pair['pos_scores'].get(criterion, 0)
                neg_score = pair['neg_scores'].get(criterion, 0)
                pair_criterion_score = (pos_score + neg_score) / 2
                filtered_scores[criterion].append(pair_criterion_score)
        else:
            stats['failed_pairs'] += 1
            if pos_has_one or neg_has_one:
                stats['failed_score_one'] += 1
            else:
                stats['failed_low_avg'] += 1
    
    # Write filtered pairs to output
    with open(output_filtered_jsonl, 'w') as f:
        for pair in filtered_pairs:
            json.dump(pair, f)
            f.write('\n')
    
    # Calculate statistics
    # Calculate average scores
    avg_scores = {criterion: np.mean(scores) if scores else 0 
                 for criterion, scores in all_scores.items()}
    
    # Calculate standard deviations
    std_devs = {
        'all': {criterion: np.std(scores) if scores else 0 
               for criterion, scores in all_scores.items()},
        'filtered': {criterion: np.std(scores) if scores else 0 
                    for criterion, scores in filtered_scores.items()}
    }
    
    # Calculate average scores for filtered data
    filtered_avg_scores = {criterion: np.mean(scores) if scores else 0 
                          for criterion, scores in filtered_scores.items()}
    
    # Calculate overall average scores
    overall_avg_score = 0
    if all([all_scores[criterion] for criterion in ['authenticity', 'consistency', 'cultural_fidelity']]):
        all_avg_scores = []
        for i in range(len(all_scores['authenticity'])):
            pair_avg = (all_scores['authenticity'][i] + 
                       all_scores['consistency'][i] + 
                       all_scores['cultural_fidelity'][i]) / 3
            all_avg_scores.append(pair_avg)
        overall_avg_score = np.mean(all_avg_scores)
    
    filtered_overall_avg_score = 0
    if all([filtered_scores[criterion] for criterion in ['authenticity', 'consistency', 'cultural_fidelity']]):
        filtered_avg_scores_list = []
        for i in range(len(filtered_scores['authenticity'])):
            pair_avg = (filtered_scores['authenticity'][i] + 
                       filtered_scores['consistency'][i] + 
                       filtered_scores['cultural_fidelity'][i]) / 3
            filtered_avg_scores_list.append(pair_avg)
        filtered_overall_avg_score = np.mean(filtered_avg_scores_list)
    
    # Print filtering statistics
    print("\n===== Filtering Statistics =====")
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"Passed pairs: {stats['passed_pairs']} ({stats['passed_pairs']/stats['total_pairs']*100:.2f}% pass rate)")
    print(f"Failed pairs: {stats['failed_pairs']} ({stats['failed_pairs']/stats['total_pairs']*100:.2f}%)")
    print(f"  - Failed due to score of 1: {stats['failed_score_one']} ({stats['failed_score_one']/stats['total_pairs']*100:.2f}%)")
    print(f"  - Failed due to low average: {stats['failed_low_avg']} ({stats['failed_low_avg']/stats['total_pairs']*100:.2f}%)")
    print(f"Pairs with errors: {stats['error_pairs']} ({stats['error_pairs']/stats['total_pairs']*100:.2f}%)")
    
    print(f"\nAverage scores (all data):")
    for criterion in ['authenticity', 'consistency', 'cultural_fidelity']:
        print(f"  {criterion}: {avg_scores[criterion]:.3f} ± {std_devs['all'][criterion]:.3f}")
    print(f"Overall average score: {overall_avg_score:.3f}")
    
    print(f"\nAverage scores (filtered data):")
    for criterion in ['authenticity', 'consistency', 'cultural_fidelity']:
        print(f"  {criterion}: {filtered_avg_scores[criterion]:.3f} ± {std_devs['filtered'][criterion]:.3f}")
    print(f"Overall average score: {filtered_overall_avg_score:.3f}")
    print("===============================\n")
    
    # Generate statistics file path for TXT file
    stats_file = os.path.splitext(output_filtered_jsonl)[0] + "_stats.txt"
    
    # Save statistics to txt file
    with open(stats_file, 'w') as f:
        f.write(f"Quality Filtering Statistics\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Input file: {input_scored_jsonl}\n")
        f.write(f"Output file: {output_filtered_jsonl}\n\n")
        
        f.write(f"Total pairs: {stats['total_pairs']}\n")
        f.write(f"Passed pairs: {stats['passed_pairs']} ({stats['passed_pairs']/stats['total_pairs']*100:.2f}% pass rate)\n")
        f.write(f"Failed pairs: {stats['failed_pairs']} ({stats['failed_pairs']/stats['total_pairs']*100:.2f}%)\n")
        f.write(f"  - Failed due to score of 1: {stats['failed_score_one']} ({stats['failed_score_one']/stats['total_pairs']*100:.2f}%)\n")
        f.write(f"  - Failed due to low average: {stats['failed_low_avg']} ({stats['failed_low_avg']/stats['total_pairs']*100:.2f}%)\n")
        f.write(f"Pairs with errors: {stats['error_pairs']} ({stats['error_pairs']/stats['total_pairs']*100:.2f}%)\n\n")
        
        f.write(f"Average scores (all data):\n")
        for criterion in ['authenticity', 'consistency', 'cultural_fidelity']:
            f.write(f"  {criterion}: {avg_scores[criterion]:.3f} ± {std_devs['all'][criterion]:.3f}\n")
        f.write(f"Overall average score: {overall_avg_score:.3f}\n\n")
        
        f.write(f"Average scores (filtered data):\n")
        for criterion in ['authenticity', 'consistency', 'cultural_fidelity']:
            f.write(f"  {criterion}: {filtered_avg_scores[criterion]:.3f} ± {std_devs['filtered'][criterion]:.3f}\n")
        f.write(f"Overall average score: {filtered_overall_avg_score:.3f}\n")
    
    # Save JSON statistics
    stats_json_file = os.path.splitext(output_filtered_jsonl)[0] + "_stats.json"
    json_stats = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_pairs': stats['total_pairs'],
        'passed_pairs': stats['passed_pairs'],
        'failed_pairs': stats['failed_pairs'],
        'pass_rate': stats['passed_pairs'] / stats['total_pairs'] * 100 if stats['total_pairs'] > 0 else 0,
        'failed_due_to_score_one': stats['failed_score_one'],
        'failed_due_to_low_avg': stats['failed_low_avg'],
        'error_pairs': stats['error_pairs'],
        'averages': {
            'all_data': {
                criterion: {
                    'mean': float(avg_scores[criterion]),
                    'std': float(std_devs['all'][criterion])
                } for criterion in ['authenticity', 'consistency', 'cultural_fidelity']
            },
            'filtered_data': {
                criterion: {
                    'mean': float(filtered_avg_scores[criterion]),
                    'std': float(std_devs['filtered'][criterion])
                } for criterion in ['authenticity', 'consistency', 'cultural_fidelity']
            }
        },
        'overall_avg_score': float(overall_avg_score),
        'filtered_overall_avg_score': float(filtered_overall_avg_score)
    }
    
    with open(stats_json_file, 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    print(f"Filtering complete. Filtered pairs saved to: {output_filtered_jsonl}")
    print(f"Statistics saved to: {stats_file} and {stats_json_file}")
    
    return stats

def main():
    """Main function to run the filtering process."""
    parser = argparse.ArgumentParser(description='Quality filter for image pairs')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'filter', 'both'], default='both',
                        help='Mode to run: evaluate only, filter only, or both')
    parser.add_argument('--input', type=str, default=None,
                        help='Input JSONL file with image pairs')
    parser.add_argument('--output-scored', type=str, default=None,
                        help='Output JSONL file with scored pairs')
    parser.add_argument('--output-filtered', type=str, default=None,
                        help='Output JSONL file with filtered pairs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Number of pairs to process before writing to output')
    parser.add_argument('--image-batch-size', type=int, default=4,
                        help='Batch size for image evaluation')
    
    args = parser.parse_args()
    
    # Default paths if not specified
    if args.input is None:
        args.input = "/data/yuchen/CultureCLIP_data/pos_neg_crope/categorized_data/Cuisine.jsonl"
    
    if args.output_scored is None:
        args.output_scored = os.path.splitext(args.input)[0] + "_scored.jsonl"
    
    if args.output_filtered is None:
        args.output_filtered = os.path.splitext(args.input)[0] + "_filtered.jsonl"
    
    print(f"Input file: {args.input}")
    print(f"Output scored file: {args.output_scored}")
    print(f"Output filtered file: {args.output_filtered}")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'evaluate' or args.mode == 'both':
        print("Loading model...")
        model, processor = load_model()
        
        # Evaluate all pairs and save scores
        evaluate_all_pairs(
            args.input,
            args.output_scored,
            model,
            processor,
            batch_size=args.batch_size,
            image_batch_size=args.image_batch_size
        )
    
    if args.mode == 'filter' or args.mode == 'both':
        # Filter pairs based on scores
        filter_scored_pairs(args.output_scored, args.output_filtered)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
