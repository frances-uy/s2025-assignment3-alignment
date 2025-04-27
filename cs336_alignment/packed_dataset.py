import torch
from torch.utils.data import Dataset
import json
import random
import os

class PackedSFTDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Load the JSONL dataset
        assert dataset_path.endswith(".jsonl") or dataset_path.endswith(".jsonl.gz"), "Dataset must be a JSONL file"
        if dataset_path.endswith(".gz"):
            import gzip
            open_fn = gzip.open
        else:
            open_fn = open

        documents = []
        with open_fn(dataset_path, "rt", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                instruction = example.get("prompt", "")
                response = example.get("response", "")
                # Format following the Alpaca template
                formatted = (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
                    "### Instruction:\n"
                    f"{instruction}\n"
                    "### Response:\n"
                    f"{response}\n"
                )
                documents.append(formatted)

        if shuffle:
            random.shuffle(documents)
        
        # Tokenize all documents at once
        tokenized = tokenizer(
            documents,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
            padding=False
        )

        self.input_ids = tokenized.input_ids.view(-1)  # flatten to a long 1D tensor

        # Calculate number of complete sequences
        n_tokens = self.input_ids.size(0)
        self.n_sequences = n_tokens // self.seq_length
        self.input_ids = self.input_ids[: self.n_sequences * self.seq_length]  # truncate to fit

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = (idx + 1) * self.seq_length
        input_ids = self.input_ids[start:end]
        labels = input_ids.clone()  # Labels are just shifted input IDs

        return {
            "input_ids": input_ids,
            "labels": labels
        }
