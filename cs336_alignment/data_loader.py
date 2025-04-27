"""
Data loading utilities for instruction fine-tuning.
"""
import os
import json
import gzip
import random
import torch
from torch.utils.data import Dataset, DataLoader


class PackedSFTDataset(Dataset):
    """Dataset for packed sequence fine-tuning."""
    
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle=True):
        """
        Initialize a dataset for packed SFT data.
        
        Args:
            tokenizer: HuggingFace tokenizer for tokenizing text
            dataset_path: Path to the dataset file
            seq_length: The desired sequence length for each example
            shuffle: Whether to shuffle the order of examples
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.shuffle = shuffle
        
        # Load and tokenize the data from the dataset path
        self.token_ids = self._load_and_tokenize_data(dataset_path)
        
        # Split the token ids into chunks of size seq_length
        self.chunks = [
            self.token_ids[i:i+seq_length] 
            for i in range(0, len(self.token_ids) - seq_length + 1, seq_length)
        ]
        
        print(f"Created {len(self.chunks)} chunks of size {seq_length}")
    
    def _load_and_tokenize_data(self, dataset_path):
        """Load and tokenize the data from the dataset path."""
        # Function to format an example as a prompt
        def format_example(example):
            prompt = example.get("prompt", "")
            response = example.get("response", "")
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n{response}{self.tokenizer.eos_token}"
        
        # Open the dataset file (handling gzipped files)
        if dataset_path.endswith('.gz'):
            open_fn = gzip.open
        else:
            open_fn = open
        
        # Load the examples from the dataset
        examples = []
        with open_fn(dataset_path, 'rt') as f:
            for line in f:
                example = json.loads(line)
                examples.append(example)
        
        print(f"Loaded {len(examples)} examples from {dataset_path}")
        
        # Shuffle the examples if requested
        if self.shuffle:
            random.shuffle(examples)
        
        # Format each example and tokenize
        all_token_ids = []
        
        for example in examples:
            # Format the example as a text prompt
            text = format_example(example)
            
            # Tokenize the text
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Add to the list of all token ids
            all_token_ids.extend(token_ids)
        
        print(f"Total tokens: {len(all_token_ids)}")
        return all_token_ids
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.chunks)
    
    def __getitem__(self, i):
        """Return the i-th example from the dataset."""
        # Get the i-th chunk of token ids
        chunk = self.chunks[i]
        
        # Pad the chunk if it's shorter than seq_length
        if len(chunk) < self.seq_length:
            chunk = chunk + [self.tokenizer.pad_token_id] * (self.seq_length - len(chunk))
        
        # Get the input ids (all tokens except the last one)
        input_ids = chunk[:-1]
        
        # Get the labels (all tokens except the first one)
        labels = chunk[1:]
        
        # Add a padding token to the end for both input_ids and labels
        input_ids = input_ids + [self.tokenizer.pad_token_id]
        labels = labels + [self.tokenizer.pad_token_id]
        
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels)
        }


def get_sft_dataloader(dataset, batch_size, shuffle=True):
    """
    Create a DataLoader for the SFT dataset.
    
    Args:
        dataset: The dataset to load
        batch_size: The batch size to use
        shuffle: Whether to shuffle the data
    
    Returns:
        A DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )


def get_packed_sft_dataset(tokenizer, dataset_path, seq_length, shuffle=True):
    """
    Create a PackedSFTDataset.
    
    Args:
        tokenizer: HuggingFace tokenizer for tokenizing text
        dataset_path: Path to the dataset file
        seq_length: The desired sequence length for each example
        shuffle: Whether to shuffle the order of examples
    
    Returns:
        A PackedSFTDataset
    """
    return PackedSFTDataset(tokenizer, dataset_path, seq_length, shuffle)


def run_iterate_batches(dataset, batch_size, shuffle=True):
    """
    Adapter function for testing batch iteration.
    
    Args:
        dataset: The dataset to iterate over
        batch_size: The batch size to use
        shuffle: Whether to shuffle the data
    
    Returns:
        A list of all batches from one epoch of the data
    """
    dataloader = get_sft_dataloader(dataset, batch_size, shuffle)
    return list(dataloader)