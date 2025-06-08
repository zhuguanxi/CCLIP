import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPTokenizer

logger = logging.getLogger(__name__)

class CultureClipDataset(Dataset):
    """Dataset for cultureCLIP model training with positive and negative image/caption/concept pairs."""
    
    def __init__(
        self,
        data_path: str, 
        processor: CLIPProcessor,
        tokenizer: CLIPTokenizer,
        pos_image_column: str = "pos_image_path",
        pos_caption_column: str = "pos_caption",
        neg_image_column: str = "neg_image_path", 
        neg_caption_column: str = "neg_caption",
        pos_concept_column: str = "pos_concept",
        neg_concept_column: str = "neg_concept",
        max_seq_length: int = 77,
        batch_size: int = 8,
        max_train_samples: Optional[int] = None,
        mode: str = "train",
        **kwargs
    ):
        """
        Dataset for cultureCLIP model training.
        
        Args:
            data_path: Path to data file
            processor: CLIP processor
            tokenizer: CLIP tokenizer
            pos_image_column: Positive image column name
            pos_caption_column: Positive text column name
            neg_image_column: Negative image column name
            neg_caption_column: Negative text column name
            pos_concept_column: Positive concept column name
            neg_concept_column: Negative concept column name
            max_seq_length: Maximum sequence length
            batch_size: Batch size
            max_train_samples: Maximum number of samples
            mode: Mode, "train" or "valid"
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_column = pos_image_column  # store as image_column internally for backward compatibility
        self.caption_column = pos_caption_column  # store as caption_column internally for backward compatibility
        self.neg_image_column = neg_image_column
        self.neg_caption_column = neg_caption_column
        self.pos_concept_column = pos_concept_column
        self.neg_concept_column = neg_concept_column
        self.max_seq_length = min(max_seq_length, 77)  # Ensure not exceeding CLIP's limit
        self.batch_size = batch_size
        self.mode = mode
        
        # Load data
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                if data_path.endswith('.jsonl'):
                    # JSONL format, one JSON object per line
                    data = [json.loads(line) for line in f]
                else:
                    # Standard JSON format
                    data = json.load(f)
            self.data = data
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # If max_train_samples is specified, truncate the data
        if max_train_samples is not None and max_train_samples < len(self.data):
            self.data = self.data[:max_train_samples]
            
        # Validate data format
        self._validate_data()
    
    def _validate_data(self):
        """Validate that all required columns are present in the data"""
        required_columns = [
            self.image_column,         # pos_image_path
            self.caption_column,       # pos_caption
            self.neg_image_column,     # neg_image_path
            self.neg_caption_column,   # neg_caption
            self.pos_concept_column,   # pos_concept
            self.neg_concept_column    # neg_concept
        ]
        
        # Check if the first sample contains all required columns
        if len(self.data) > 0:
            sample = self.data[0]
            missing_columns = [col for col in required_columns if col not in sample]
            if missing_columns:
                logger.warning(f"Data missing required columns: {missing_columns}")
                raise ValueError(f"All experiments require complete six input columns, missing: {missing_columns}")
    
    def __len__(self):
        """Return dataset size"""
        return len(self.data)
    
    def _process_image(self, image_path):
        """Process image"""
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            # Process image using CLIP processor
            processed_image = self.processor.image_processor(image, return_tensors="pt")["pixel_values"][0]
            return processed_image
        except Exception as e:
            logger.warning(f"Error processing image: {image_path}, error: {str(e)}")
            # Return blank image as fallback
            blank_image = torch.zeros(3, 224, 224)
            return blank_image
    
    def _process_text(self, text):
        """Process text"""
        # Process text using CLIP tokenizer
        encoded_text = self.tokenizer(
            text,
            padding="max_length",
            max_length=min(self.max_seq_length, 77),  # Ensure not exceeding CLIP's 77 token limit
            truncation=True,
            return_tensors="pt"
        )
        return encoded_text
    
    def __getitem__(self, index):
        """Get data sample"""
        item = self.data[index]
        
        # Process positive image
        pos_image = self._process_image(item[self.image_column])
        
        # Process positive text
        pos_caption_data = self._process_text(item[self.caption_column])
        pos_caption_ids = pos_caption_data["input_ids"][0]
        pos_caption_attention_mask = pos_caption_data["attention_mask"][0]
        
        # Process negative image
        neg_image = self._process_image(item[self.neg_image_column])
        
        # Process negative text
        neg_caption_data = self._process_text(item[self.neg_caption_column])
        neg_caption_ids = neg_caption_data["input_ids"][0]
        neg_caption_attention_mask = neg_caption_data["attention_mask"][0]
        
        # Process positive concept
        pos_concept_data = self._process_text(item[self.pos_concept_column])
        pos_concept_ids = pos_concept_data["input_ids"][0]
        pos_concept_attention_mask = pos_concept_data["attention_mask"][0]
        
        # Process negative concept
        neg_concept_data = self._process_text(item[self.neg_concept_column])
        neg_concept_ids = neg_concept_data["input_ids"][0]
        neg_concept_attention_mask = neg_concept_data["attention_mask"][0]
        
        # Return complete sample with all six inputs
        sample = {
            "pos_image": pos_image,
            "pos_caption": pos_caption_ids,
            "pos_caption_attention_mask": pos_caption_attention_mask,
            "neg_image": neg_image,
            "neg_caption": neg_caption_ids,
            "neg_caption_attention_mask": neg_caption_attention_mask,
            "pos_concept": pos_concept_ids,
            "pos_concept_attention_mask": pos_concept_attention_mask,
            "neg_concept": neg_concept_ids,
            "neg_concept_attention_mask": neg_concept_attention_mask
        }
        
        return sample

class CultureClipValDataset(CultureClipDataset):
    """Validation dataset for cultureCLIP model."""
    
    def __init__(
        self,
        data_path: str, 
        processor: CLIPProcessor,
        tokenizer: CLIPTokenizer,
        pos_image_column: str = "pos_image_path",
        pos_caption_column: str = "pos_caption",
        neg_image_column: str = "neg_image_path", 
        neg_caption_column: str = "neg_caption",
        pos_concept_column: str = "pos_concept",
        neg_concept_column: str = "neg_concept",
        max_seq_length: int = 77,
        batch_size: int = 8,
        max_train_samples: int = 1000,
        **kwargs
    ):
        # Initialize parent class, set mode to "valid"
        super().__init__(
            data_path=data_path,
            processor=processor,
            tokenizer=tokenizer,
            pos_image_column=pos_image_column,
            pos_caption_column=pos_caption_column,
            neg_image_column=neg_image_column,
            neg_caption_column=neg_caption_column,
            pos_concept_column=pos_concept_column,
            neg_concept_column=neg_concept_column,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            max_train_samples=max_train_samples,
            mode="valid",
            **kwargs
        )

def collate_fn(examples):
    """
    Batch processing function to combine multiple samples into a batch
    """
    # Basic fields
    pos_image = torch.stack([example["pos_image"] for example in examples])
    pos_caption = torch.stack([example["pos_caption"] for example in examples])
    pos_caption_attention_mask = torch.stack([example["pos_caption_attention_mask"] for example in examples])
    
    batch = {
        "pos_image": pos_image,
        "pos_caption": pos_caption,
        "pos_caption_attention_mask": pos_caption_attention_mask,
    }
    
    # Check if negative samples and concepts exist
    has_neg_samples = all("neg_image" in example for example in examples)
    has_concept = all("pos_concept" in example for example in examples)
    
    # If negative samples exist, add to batch
    if has_neg_samples:
        batch["neg_image"] = torch.stack([example["neg_image"] for example in examples])
        batch["neg_caption"] = torch.stack([example["neg_caption"] for example in examples])
        batch["neg_caption_attention_mask"] = torch.stack([example["neg_caption_attention_mask"] for example in examples])
    
    # If concepts exist, add to batch
    if has_concept:
        batch["pos_concept"] = torch.stack([example["pos_concept"] for example in examples])
        batch["pos_concept_attention_mask"] = torch.stack([example["pos_concept_attention_mask"] for example in examples])
        
        if has_neg_samples:
            batch["neg_concept"] = torch.stack([example["neg_concept"] for example in examples])
            batch["neg_concept_attention_mask"] = torch.stack([example["neg_concept_attention_mask"] for example in examples])
    
    return batch

def get_dataloaders(
    train_file: str,
    processor: CLIPProcessor,
    tokenizer: CLIPTokenizer,
    pos_image_column: str = "pos_image_path",
    pos_caption_column: str = "pos_caption",
    neg_image_column: str = "neg_image_path",
    neg_caption_column: str = "neg_caption",
    pos_concept_column: str = "pos_concept",
    neg_concept_column: str = "neg_concept",
    max_seq_length: int = 77,
    batch_size: int = 32,
    num_workers: int = 4,
    val_file: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
):
    """
    Get training and validation dataloaders
    
    Args:
        train_file: Path to training data file
        processor: CLIP processor
        tokenizer: CLIP tokenizer
        pos_image_column: Positive image path column name
        pos_caption_column: Positive text description column name
        neg_image_column: Negative image path column name
        neg_caption_column: Negative text description column name
        pos_concept_column: Positive concept column name
        neg_concept_column: Negative concept column name
        max_seq_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of dataloader worker processes
        val_file: Path to validation data file
        max_train_samples: Maximum number of training samples
        max_eval_samples: Maximum number of evaluation samples
    
    Returns:
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader (if validation file provided)
    """
    # Check if max_seq_length exceeds CLIP's limit
    if max_seq_length > 77:
        print(f"Warning: max_seq_length ({max_seq_length}) is greater than CLIP's limit (77). Will be capped at 77.")
        max_seq_length = 77
    
    # Create training dataset
    train_dataset = CultureClipDataset(
        data_path=train_file,
        processor=processor,
        tokenizer=tokenizer,
        pos_image_column=pos_image_column,
        pos_caption_column=pos_caption_column,
        neg_image_column=neg_image_column,
        neg_caption_column=neg_caption_column,
        pos_concept_column=pos_concept_column,
        neg_concept_column=neg_concept_column,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        max_train_samples=max_train_samples,
        mode="train",
    )
    
    # Create training dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # If validation file provided, create validation dataloader
    val_dataloader = None
    if val_file:
        val_dataset = CultureClipValDataset(
            data_path=val_file,
            processor=processor,
            tokenizer=tokenizer,
            pos_image_column=pos_image_column,
            pos_caption_column=pos_caption_column,
            neg_image_column=neg_image_column,
            neg_caption_column=neg_caption_column,
            pos_concept_column=pos_concept_column,
            neg_concept_column=neg_concept_column,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            max_train_samples=max_eval_samples,
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    return train_dataloader, val_dataloader 