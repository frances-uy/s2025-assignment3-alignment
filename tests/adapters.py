#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

from cs336_alignment.data_loader import PackedSFTDataset


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    return PackedSFTDataset(tokenizer, dataset_path, seq_length, shuffle)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    return list(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True))


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    # Use regex to find patterns like "The correct answer is X" or just "X"
    patterns = [
        r"[Tt]he correct answer is ([A-Da-d])",  # Match "The correct answer is X"
        r"[Tt]he answer is ([A-Da-d])",          # Match "The answer is X"
        r"[Aa]nswer:? ([A-Da-d])",               # Match "Answer: X" or "answer X"
        r"^\s*([A-Da-d])\.?\s*$"                 # Match just "X" or "X." on a line
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, model_output)
        if matches:
            # Return the first match, converted to uppercase
            return matches[0].upper()
    
    # If we couldn't parse a letter, return None
    return None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    # Remove commas from numbers (e.g., "1,234" -> "1234")
    model_output = model_output.replace(",", "")
    
    # Use regex to find all numbers in the response
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", model_output)
    
    # If numbers are found, return the last one as the final answer
    if numbers:
        return numbers[-1]
    
    # If no numbers found, return None
    return None


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    # Format using the Alpaca template
    prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"
    formatted_prompt = prompt_template.format(prompt)
    
    # Add EOS token to responses
    chosen_with_eos = response_chosen + tokenizer.eos_token
    rejected_with_eos = response_rejected + tokenizer.eos_token
    
    # Get devices
    policy_device = next(lm.parameters()).device
    ref_device = next(lm_ref.parameters()).device
    
    # Tokenize inputs
    prompt_input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False, return_tensors="pt").to(policy_device)
    chosen_input_ids = tokenizer.encode(chosen_with_eos, add_special_tokens=False, return_tensors="pt").to(policy_device)
    rejected_input_ids = tokenizer.encode(rejected_with_eos, add_special_tokens=False, return_tensors="pt").to(policy_device)
    
    # Compute log probabilities for chosen response
    with torch.no_grad():
        # Concatenate prompt and chosen response
        chosen_input_ids_full = torch.cat([prompt_input_ids, chosen_input_ids], dim=1)
        chosen_attention_mask = torch.ones_like(chosen_input_ids_full)
        
        # Get logits from policy model
        chosen_outputs_policy = lm(input_ids=chosen_input_ids_full, attention_mask=chosen_attention_mask)
        chosen_logits_policy = chosen_outputs_policy.logits
        
        # Shift for next-token prediction
        shift_chosen_logits_policy = chosen_logits_policy[:, :-1, :]
        shift_chosen_ids = chosen_input_ids_full[:, 1:]
        
        # Compute log probs for policy model
        chosen_log_probs_policy = torch.log_softmax(shift_chosen_logits_policy, dim=-1)
        chosen_log_probs_policy = torch.gather(
            chosen_log_probs_policy, 
            dim=2, 
            index=shift_chosen_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask to only include response tokens (excluding prompt)
        response_mask = torch.zeros_like(shift_chosen_ids)
        response_mask[:, prompt_input_ids.shape[1]-1:] = 1
        chosen_log_probs_policy = (chosen_log_probs_policy * response_mask).sum() / response_mask.sum()
        
        # Move tensor to reference model device if needed
        if policy_device != ref_device:
            chosen_input_ids_full = chosen_input_ids_full.to(ref_device)
            chosen_attention_mask = chosen_attention_mask.to(ref_device)
        
        # Get logits from reference model
        chosen_outputs_ref = lm_ref(input_ids=chosen_input_ids_full, attention_mask=chosen_attention_mask)
        chosen_logits_ref = chosen_outputs_ref.logits
        
        # Shift for next-token prediction
        shift_chosen_logits_ref = chosen_logits_ref[:, :-1, :]
        shift_chosen_ids_ref = chosen_input_ids_full[:, 1:]
        
        # Compute log probs for reference model
        chosen_log_probs_ref = torch.log_softmax(shift_chosen_logits_ref, dim=-1)
        chosen_log_probs_ref = torch.gather(
            chosen_log_probs_ref, 
            dim=2, 
            index=shift_chosen_ids_ref.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask to only include response tokens (excluding prompt)
        response_mask_ref = torch.zeros_like(shift_chosen_ids_ref)
        response_mask_ref[:, prompt_input_ids.shape[1]-1:] = 1
        chosen_log_probs_ref = (chosen_log_probs_ref * response_mask_ref).sum() / response_mask_ref.sum()
        
        # Move tensor back to policy device if needed
        if policy_device != ref_device:
            chosen_log_probs_ref = chosen_log_probs_ref.to(policy_device)
        
        # Process rejected response similarly
        rejected_input_ids_full = torch.cat([prompt_input_ids, rejected_input_ids], dim=1)
        rejected_attention_mask = torch.ones_like(rejected_input_ids_full)
        
        # Get logits from policy model
        rejected_outputs_policy = lm(input_ids=rejected_input_ids_full, attention_mask=rejected_attention_mask)
        rejected_logits_policy = rejected_outputs_policy.logits
        
        # Shift for next-token prediction
        shift_rejected_logits_policy = rejected_logits_policy[:, :-1, :]
        shift_rejected_ids = rejected_input_ids_full[:, 1:]
        
        # Compute log probs for policy model
        rejected_log_probs_policy = torch.log_softmax(shift_rejected_logits_policy, dim=-1)
        rejected_log_probs_policy = torch.gather(
            rejected_log_probs_policy, 
            dim=2, 
            index=shift_rejected_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask to only include response tokens (excluding prompt)
        response_mask = torch.zeros_like(shift_rejected_ids)
        response_mask[:, prompt_input_ids.shape[1]-1:] = 1
        rejected_log_probs_policy = (rejected_log_probs_policy * response_mask).sum() / response_mask.sum()
        
        # Move tensor to reference model device if needed
        if policy_device != ref_device:
            rejected_input_ids_full = rejected_input_ids_full.to(ref_device)
            rejected_attention_mask = rejected_attention_mask.to(ref_device)
        
        # Get logits from reference model
        rejected_outputs_ref = lm_ref(input_ids=rejected_input_ids_full, attention_mask=rejected_attention_mask)
        rejected_logits_ref = rejected_outputs_ref.logits
        
        # Shift for next-token prediction
        shift_rejected_logits_ref = rejected_logits_ref[:, :-1, :]
        shift_rejected_ids_ref = rejected_input_ids_full[:, 1:]
        
        # Compute log probs for reference model
        rejected_log_probs_ref = torch.log_softmax(shift_rejected_logits_ref, dim=-1)
        rejected_log_probs_ref = torch.gather(
            rejected_log_probs_ref, 
            dim=2, 
            index=shift_rejected_ids_ref.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask to only include response tokens (excluding prompt)
        response_mask_ref = torch.zeros_like(shift_rejected_ids_ref)
        response_mask_ref[:, prompt_input_ids.shape[1]-1:] = 1
        rejected_log_probs_ref = (rejected_log_probs_ref * response_mask_ref).sum() / response_mask_ref.sum()
        
        # Move tensor back to policy device if needed
        if policy_device != ref_device:
            rejected_log_probs_ref = rejected_log_probs_ref.to(policy_device)
    
    # Compute chosen and rejected rewards (log(π(y|x)) - log(πref(y|x)))
    chosen_reward = chosen_log_probs_policy - chosen_log_probs_ref
    rejected_reward = rejected_log_probs_policy - rejected_log_probs_ref
    
    # Compute the DPO loss: -log(σ(β * (r_w - r_l)))
    loss = -torch.nn.functional.logsigmoid(beta * (chosen_reward - rejected_reward))
    
    return loss