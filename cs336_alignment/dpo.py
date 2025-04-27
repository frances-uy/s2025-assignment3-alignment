"""
Direct Preference Optimization (DPO) implementation.
"""
import os
import json
import gzip
import random
import torch
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hh_dataset(hh_dir="/home/shared/hh", split_ratio=0.95):
    """
    Load the Anthropic Helpful and Harmless (HH) dataset.
    
    Args:
        hh_dir: Directory containing the HH dataset files
        split_ratio: Ratio of data to use for training (remainder used for validation)
    
    Returns:
        train_examples: List of training examples
        val_examples: List of validation examples
    """
    filenames = [
        "harmless-base.jsonl.gz",
        "helpful-base.jsonl.gz",
        "helpful-online.jsonl.gz",
        "helpful-rejection-sampled.jsonl.gz"
    ]
    
    all_examples = []
    
    for filename in filenames:
        file_path = os.path.join(hh_dir, filename)
        print(f"Loading data from {file_path}")
        
        # Open the gzipped file
        with gzip.open(file_path, 'rt') as f:
            for line in f:
                # Parse the JSON data
                data = json.loads(line)
                
                # Extract the human and assistant messages
                chosen_conversation = data.get("chosen", [])
                rejected_conversation = data.get("rejected", [])
                
                # Check if it's a single-turn conversation
                if len(chosen_conversation) < 2 or len(rejected_conversation) < 2:
                    continue
                
                # Skip if they aren't simple single-turn conversations
                if len(chosen_conversation) > 2 or len(rejected_conversation) > 2:
                    continue
                
                # Extract the first human message (instruction)
                human_msg_chosen = chosen_conversation[0].get("text", "")
                human_msg_rejected = rejected_conversation[0].get("text", "")
                
                # Verify the human messages are the same
                if human_msg_chosen != human_msg_rejected:
                    continue
                
                # Extract the assistant responses
                chosen_response = chosen_conversation[1].get("text", "")
                rejected_response = rejected_conversation[1].get("text", "")
                
                # Skip if either response is empty
                if not chosen_response or not rejected_response:
                    continue
                
                # Create example
                example = {
                    "instruction": human_msg_chosen,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                    "source": os.path.basename(file_path).split(".")[0]  # Track which file it came from
                }
                
                all_examples.append(example)
    
    print(f"Loaded {len(all_examples)} examples from HH dataset")
    
    # Shuffle the examples
    random.shuffle(all_examples)
    
    # Split into train and validation sets
    split_idx = int(len(all_examples) * split_ratio)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    print(f"Split into {len(train_examples)} training examples and {len(val_examples)} validation examples")
    
    return train_examples, val_examples


class HHPreferenceDataset(Dataset):
    """Dataset for HH preference data."""
    
    def __init__(self, examples, tokenizer, max_length=512):
        """
        Initialize the HH preference dataset.
        
        Args:
            examples: List of preference examples
            tokenizer: The tokenizer to use
            max_length: Maximum sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Return the idx-th example from the dataset."""
        example = self.examples[idx]
        
        # Extract fields
        instruction = example["instruction"]
        chosen = example["chosen"]
        rejected = example["rejected"]
        
        # Format using the Alpaca template
        prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"
        prompt = prompt_template.format(instruction)
        
        # Tokenize prompt, chosen, and rejected
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        chosen_tokens = self.tokenizer.encode(chosen + self.tokenizer.eos_token, add_special_tokens=False)
        rejected_tokens = self.tokenizer.encode(rejected + self.tokenizer.eos_token, add_special_tokens=False)
        
        # Truncate if necessary
        if len(prompt_tokens) + max(len(chosen_tokens), len(rejected_tokens)) > self.max_length:
            # Truncate prompt to leave room for the responses
            max_prompt_length = self.max_length - max(len(chosen_tokens), len(rejected_tokens))
            prompt_tokens = prompt_tokens[:max_prompt_length]
        
        # Combine prompt with chosen and rejected responses
        chosen_input_ids = prompt_tokens + chosen_tokens
        rejected_input_ids = prompt_tokens + rejected_tokens
        
        # Pad sequences if needed
        if len(chosen_input_ids) < self.max_length:
            chosen_input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(chosen_input_ids))
        else:
            chosen_input_ids = chosen_input_ids[:self.max_length]
            
        if len(rejected_input_ids) < self.max_length:
            rejected_input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(rejected_input_ids))
        else:
            rejected_input_ids = rejected_input_ids[:self.max_length]
        
        # Create attention masks (1 for tokens, 0 for padding)
        chosen_attention_mask = [1] * len(prompt_tokens + chosen_tokens)
        chosen_attention_mask += [0] * (self.max_length - len(chosen_attention_mask))
        chosen_attention_mask = chosen_attention_mask[:self.max_length]
        
        rejected_attention_mask = [1] * len(prompt_tokens + rejected_tokens)
        rejected_attention_mask += [0] * (self.max_length - len(rejected_attention_mask))
        rejected_attention_mask = rejected_attention_mask[:self.max_length]
        
        # Convert to tensors
        chosen_input_ids = torch.tensor(chosen_input_ids)
        rejected_input_ids = torch.tensor(rejected_input_ids)
        chosen_attention_mask = torch.tensor(chosen_attention_mask)
        rejected_attention_mask = torch.tensor(rejected_attention_mask)
        
        return {
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_attention_mask": rejected_attention_mask,
            "prompt_length": len(prompt_tokens),
        }


def compute_per_instance_dpo_loss(
    policy_model, 
    reference_model, 
    tokenizer, 
    prompt, 
    chosen_response, 
    rejected_response, 
    beta=0.1, 
    max_length=512
):
    """
    Compute the DPO loss for a single example.
    
    Args:
        policy_model: The policy model being trained
        reference_model: The reference model
        tokenizer: The tokenizer
        prompt: The prompt text
        chosen_response: The preferred response
        rejected_response: The dispreferred response
        beta: The regularization strength
        max_length: Maximum sequence length
    
    Returns:
        loss: The DPO loss for this example
    """
    # Format using the Alpaca template
    prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"
    formatted_prompt = prompt_template.format(prompt)
    
    # Add EOS token to responses
    chosen_with_eos = chosen_response + tokenizer.eos_token
    rejected_with_eos = rejected_response + tokenizer.eos_token
    
    # Get devices
    policy_device = next(policy_model.parameters()).device
    ref_device = next(reference_model.parameters()).device
    
    # Tokenize inputs
    prompt_input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False, return_tensors="pt").to(policy_device)
    chosen_input_ids = tokenizer.encode(chosen_with_eos, add_special_tokens=False, return_tensors="pt").to(policy_device)
    rejected_input_ids = tokenizer.encode(rejected_with_eos, add_special_tokens=False, return_tensors="pt").to(policy_device)
    
    # Compute log probabilities for chosen response
    chosen_input_ids_full = torch.cat([prompt_input_ids, chosen_input_ids], dim=1)
    chosen_log_probs_policy = compute_log_probs(policy_model, chosen_input_ids_full, prompt_input_ids.shape[1])
    
    # Move to reference model device if different
    if policy_device != ref_device:
        chosen_input_ids_full = chosen_input_ids_full.to(ref_device)
    
    chosen_log_probs_ref = compute_log_probs(reference_model, chosen_input_ids_full, prompt_input_ids.shape[1])
    
    # Compute log probabilities for rejected response
    rejected_input_ids_full = torch.cat([prompt_input_ids, rejected_input_ids], dim=1)
    rejected_log_probs_policy = compute_log_probs(policy_model, rejected_input_ids_full, prompt_input_ids.shape[1])
    
    # Move to reference model device if different
    if policy_device != ref_device:
        rejected_input_ids_full = rejected_input_ids_full.to(ref_device)
    
    rejected_log_probs_ref = compute_log_probs(reference_model, rejected_input_ids_full, prompt_input_ids.shape[1])
    
    # Move reference model outputs to policy device if needed
    if policy_device != ref_device:
        chosen_log_probs_ref = chosen_log_probs_ref.to(policy_device)
        rejected_log_probs_ref = rejected_log_probs_ref.to(policy_device)
    
    # Compute chosen and rejected rewards (log(π(y|x)) - log(πref(y|x)))
    chosen_reward = chosen_log_probs_policy - chosen_log_probs_ref
    rejected_reward = rejected_log_probs_policy - rejected_log_probs_ref
    
    # Compute the DPO loss: -log(σ(β * (r_w - r_l)))
    loss = -F.logsigmoid(beta * (chosen_reward - rejected_reward))
    
    return loss


def compute_log_probs(model, input_ids, prompt_length):
    """
    Compute the log probabilities of a sequence.
    
    Args:
        model: The model to use
        input_ids: The input sequence
        prompt_length: The length of the prompt (to exclude from loss computation)
    
    Returns:
        log_probs: The sum of log probabilities for the sequence (excluding the prompt)
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        
        # Shift logits and input_ids for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_input_ids = input_ids[:, 1:]
        
        # Only compute log probs for the response (i.e., exclude the prompt)
        shift_logits = shift_logits[:, (prompt_length-1):, :]
        shift_input_ids = shift_input_ids[:, (prompt_length-1):]
        
        # Compute log-softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather the log probs at the positions of the input IDs
        token_log_probs = log_probs.gather(-1, shift_input_ids.unsqueeze(-1)).squeeze(-1)
        
        # Sum the log probs for the entire response
        return token_log_probs.sum()


def per_instance_dpo(policy_model, reference_model, tokenizer, prompt, chosen, rejected, beta=0.1):
    """
    Adapter function for testing DPO loss computation.
    
    Args:
        policy_model: The policy model being trained
        reference_model: The reference model
        tokenizer: The tokenizer
        prompt: The prompt text
        chosen: The preferred response
        rejected: The dispreferred response
        beta: The regularization strength
    
    Returns:
        loss: The DPO loss for this example
    """
    return compute_per_instance_dpo_loss(
        policy_model, reference_model, tokenizer, prompt, chosen, rejected, beta
    )


def train_dpo(
    policy_model_path,
    reference_model_path,
    hh_dir,
    output_dir,
    batch_size=1,
    gradient_accumulation_steps=64,
    epochs=1,
    learning_rate=1e-6,
    beta=0.1,
    max_length=512,
    log_interval=1,
    eval_interval=100,
    save_interval=500,
    policy_device="cuda:0",
    reference_device="cuda:1"
):
    """
    Train a model with Direct Preference Optimization (DPO).
    
    Args:
        policy_model_path: Path to the policy model
        reference_model_path: Path to the reference model
        hh_dir: Directory containing HH dataset files
        output_dir: Directory to save the trained model
        batch_size: The batch size for training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        epochs: Number of epochs to train for
        learning_rate: The learning rate for training
        beta: The regularization strength for DPO
        max_length: Maximum sequence length
        log_interval: How often to log training progress (in steps)
        eval_interval: How often to evaluate on validation data (in steps)
        save_interval: How often to save model checkpoints (in steps)
        policy_device: Device for the policy model
        reference_device: Device for the reference model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from {policy_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(policy_model_path)
    
    if tokenizer.pad_token_id is None:
        # If the tokenizer doesn't have a pad token, use the eos token
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load models
    print(f"Loading policy model from {policy_model_path}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    policy_model.to(policy_device)
    
    print(f"Loading reference model from {reference_model_path}")
    reference_model = AutoModelForCausalLM.from_pretrained(
        reference_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    reference_model.to(reference_device)
    
    # Set reference model to evaluation mode
    reference_model.eval()
    
    # Load HH dataset
    train_examples, val_examples = load_hh_dataset(hh_dir)
    
    # Create datasets
    train_dataset = HHPreferenceDataset(train_examples, tokenizer, max_length)
    val_dataset = HHPreferenceDataset(val_examples, tokenizer, max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create optimizer (RMSprop as recommended in the DPO paper)
    optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=learning_rate)
    
    # Training loop
    policy_model.train()
    step = 0
    total_loss = 0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    
    print(f"Starting DPO training for {epochs} epochs")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to correct device
            chosen_input_ids = batch["chosen_input_ids"].to(policy_device)
            rejected_input_ids = batch["rejected_input_ids"].to(policy_device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(policy_device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(policy_device)
            prompt_length = batch["prompt_length"]
            
            # Forward pass for chosen response
            chosen_logits = policy_model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask
            ).logits
            
            chosen_logits_ref = reference_model(
                input_ids=chosen_input_ids.to(reference_device),
                attention_mask=chosen_attention_mask.to(reference_device)
            ).logits.to(policy_device)
            
            # Forward pass for rejected response
            rejected_logits = policy_model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask
            ).logits
            
            rejected_logits_ref = reference_model(
                input_ids=rejected_input_ids.to(reference_device),
                attention_mask=rejected_attention_mask.to(reference_device)
            ).logits.to(policy_device)
            
            # Compute log probabilities for chosen and rejected
            chosen_log_probs = compute_response_log_probs(chosen_logits, chosen_input_ids, chosen_attention_mask, prompt_length)
            chosen_log_probs_ref = compute_response_log_probs(chosen_logits_ref, chosen_input_ids, chosen_attention_mask, prompt_length)
            
            rejected_log_probs = compute_response_log_probs(rejected_logits, rejected_input_ids, rejected_attention_mask, prompt_length)
            rejected_log_probs_ref = compute_response_log_probs(rejected_logits_ref, rejected_input_ids, rejected_attention_mask, prompt_length)
            
            # Compute rewards
            chosen_rewards = chosen_log_probs - chosen_log_probs_ref
            rejected_rewards = rejected_log_probs - rejected_log_probs_ref
            
            # Compute loss
            loss = -F.logsigmoid(beta * (chosen_rewards - rejected_rewards)).mean()
            
            # Scale loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update stats
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Update parameters if we've accumulated enough gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                step += 1
                
                # Log progress
                if step % log_interval == 0:
                    avg_loss = total_loss / log_interval
                    print(f"Step {step} | Loss: {avg_loss:.4f}")
                    total_loss = 0
                
                # Evaluate on validation set
                if step % eval_interval == 0:
                    val_loss, val_acc = evaluate_dpo(
                        policy_model, 
                        reference_model, 
                        val_dataloader, 
                        beta, 
                        policy_device, 
                        reference_device
                    )
                    print(f"Validation | Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
                    
                    # Save the best model
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_val_loss = val_loss
                        print(f"New best validation accuracy: {best_val_acc:.4f}")
                        
                        # Save the model
                        model_save_path = os.path.join(output_dir, "best_model")
                        os.makedirs(model_save_path, exist_ok=True)
                        policy_model.save_pretrained(model_save_path)
                        tokenizer.save_pretrained(model_save_path)
                        print(f"Saved best model to {model_save_path}")
                
                # Save a checkpoint
                if step % save_interval == 0:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    policy_model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save the final model
    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    policy_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    print("DPO training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return best_val_acc, best_val_loss


def compute_response_log_probs(logits, input_ids, attention_mask, prompt_length):
    """
    Compute log probabilities for a response, excluding the prompt.
    
    Args:
        logits: Model logits
        input_ids: Input token IDs
        attention_mask: Attention mask
        prompt_length: Length of the prompt to exclude
    
    Returns:
        log_probs: Sum of log probabilities for the response
    """
    # Get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Shift for next-token prediction
    shift_log_probs = log_probs[:, :-1, :]
    shift_input_ids = input_ids[:, 1:]
    shift_attention_mask = attention_mask[:, 1:]
    
    # Get the token log probs
    token_log_probs = torch.gather(
        shift_log_probs, 
        dim=2, 
        index=shift_input_ids.unsqueeze(-1)
    ).squeeze(-1)
    
    # Apply attention mask and prompt mask
    batch_size = token_log_probs.shape[0]
    seq_length = token_log_probs.shape[1]
    
    response_mask = torch.zeros_like(shift_attention_mask)
    for i in range(batch_size):
        if isinstance(prompt_length, int):
            response_mask[i, prompt_length-1:] = 1
        else:
            response_mask[i, prompt_length[i]-1:] = 1
    
    # Combine with attention mask
    combined_mask = shift_attention_mask * response_mask
    
    # Mask and sum log probs
    masked_log_probs = token_log_probs * combined_mask
    return masked_log_probs.sum(dim=1) / combined_mask.sum(dim=1).clamp(min=1)


def evaluate_dpo(
    policy_model, 
    reference_model, 
    dataloader, 
    beta, 
    policy_device, 
    reference_device
):
    """
    Evaluate a model with DPO on a validation set.
    
    Args:
        policy_model: The policy model
        reference_model: The reference model
        dataloader: The validation dataloader
        beta: The regularization strength
        policy_device: Device for the policy model
        reference_device: Device for the reference model
    
    Returns:
        avg_loss: The average validation loss
        accuracy: The classification accuracy
    """
    policy_model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to correct device
            chosen_input_ids = batch["chosen_input_ids"].to(policy_device)
            rejected_input_ids = batch["rejected_input_ids"].to(policy_device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(policy_device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(policy_device)
            prompt_length = batch["prompt_length"]
            
            # Forward pass for chosen response
            chosen_logits = policy_model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask
            ).logits
            
            chosen_logits_ref = reference_model(
                input_ids=chosen_input_ids.to(reference_device),
                attention_mask=chosen_attention_mask.to(reference_device)
            ).logits.to(policy_device)
            
            # Forward pass for rejected response
            rejected_logits = policy_model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask
            ).logits
            
            rejected_logits_ref = reference_model(
                input_ids=rejected_input_ids.to(reference_device),
                attention_mask=rejected_attention_mask.to(reference_device)
            ).logits.to(policy_device)
            
            # Compute log probabilities for chosen and rejected
            chosen_log_probs = compute_response_log_probs(chosen_logits, chosen_input_ids, chosen_attention_mask, prompt_length)
            chosen_log_probs_ref = compute_response_log_probs(chosen_logits_ref, chosen_input_ids, chosen_attention_mask, prompt_length)
            
            rejected_log_probs = compute_response_log_probs(rejected_logits, rejected_input_ids, rejected_attention_mask, prompt_length)
            rejected_log_probs_ref = compute_response_log_probs(rejected_logits_ref, rejected_input_ids, rejected_attention_mask, prompt_length)
            
            # Compute rewards
            chosen_rewards = chosen_log_probs - chosen_log_probs_ref
            rejected_rewards = rejected_log_probs - rejected_log_probs_ref
            
            # Compute loss
            losses = -F.logsigmoid(beta * (chosen_rewards - rejected_rewards))
            
            # Compute accuracy (do we prefer the chosen response?)
            correct = (chosen_log_probs > rejected_log_probs).float()
            
            # Update stats
            total_loss += losses.sum().item()
            total_correct += correct.sum().item()
            total_examples += chosen_input_ids.size(0)
    
    # Reset model to training mode
    policy_model.train()
    
    # Compute average loss and accuracy
    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    
    return avg_loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with Direct Preference Optimization")
    parser.add_argument("--policy-model-path", type=str, required=True, help="Path to the policy model")
    parser.add_argument("--reference-model-path", type=str, required=True, help="Path to the reference model")
    parser.add_argument("--hh-dir", type=str, default="/home/shared/hh", help="Directory containing HH dataset files")
    parser.add_argument("--output-dir", type=str, default="outputs/dpo", help="Directory to save the trained model")
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size for training")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64, help="Number of steps to accumulate gradients")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="The learning rate for training")
    parser.add_argument("--beta", type=float, default=0.1, help="The regularization strength for DPO")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--log-interval", type=int, default=1, help="How often to log training progress (in steps)")
    parser.add_argument("--eval-interval", type=int, default=100, help="How often to evaluate on validation data (in steps)")
    parser.add_argument("--save-interval", type=int, default=500, help="How often to save model checkpoints (in steps)")
    parser.add_argument("--policy-device", type=str, default="cuda:0", help="Device for the policy model")
    parser.add_argument("--reference-device", type=str, default="cuda:1", help="Device for the reference model")
    
    args = parser.parse_args()
    
    best_val_acc, best_val_loss = train_dpo(
        args.policy_model_path,
        args.reference_model_path,
        args.hh_dir,
        args.output_dir,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.epochs,
        args.learning_rate,
        args.beta,
        args.max_length,
        args.log_interval,
        args.eval_interval,
        args.save_interval,
        args.policy_device,
        args.reference_device
    )