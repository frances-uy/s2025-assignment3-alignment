"""
Training script for supervised fine-tuning (SFT) of language models.
"""
import os
import math
import time
import torch
import argparse
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_loader import get_packed_sft_dataset, get_sft_dataloader


def get_linear_warmup_cosine_decay_scheduler(optimizer, warmup_steps, total_steps):
    """Create a learning rate scheduler with linear warmup and cosine decay."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay after warmup
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)


def train(
    model_path,
    train_path,
    val_path,
    output_dir,
    seq_length=512,
    batch_size=2,
    gradient_accumulation_steps=16,
    epochs=1,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    log_interval=10,
    eval_interval=100,
    save_interval=500,
    device="cuda"
):
    """
    Train a language model with supervised fine-tuning.
    
    Args:
        model_path: Path to the pre-trained model
        train_path: Path to the training data
        val_path: Path to the validation data
        output_dir: Directory to save the trained model
        seq_length: The sequence length to use
        batch_size: The batch size for training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        epochs: Number of epochs to train for
        learning_rate: The learning rate for training
        warmup_ratio: Ratio of total steps to use for warmup
        log_interval: How often to log training progress (in steps)
        eval_interval: How often to evaluate on validation data (in steps)
        save_interval: How often to save model checkpoints (in steps)
        device: The device to train on
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token_id is None:
        # If the tokenizer doesn't have a pad token, use the eos token
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    
    # Load training and validation data
    print(f"Loading training data from {train_path}")
    train_dataset = get_packed_sft_dataset(tokenizer, train_path, seq_length, shuffle=True)
    train_dataloader = get_sft_dataloader(train_dataset, batch_size, shuffle=True)
    
    print(f"Loading validation data from {val_path}")
    val_dataset = get_packed_sft_dataset(tokenizer, val_path, seq_length, shuffle=False)
    val_dataloader = get_sft_dataloader(val_dataset, batch_size, shuffle=False)
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Compute total steps
    total_steps = len(train_dataloader) * epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = get_linear_warmup_cosine_decay_scheduler(
        optimizer, warmup_steps, total_steps
    )
    
    # Training loop
    model.train()
    step = 0
    total_loss = 0
    best_val_loss = float("inf")
    start_time = time.time()
    
    print(f"Starting training with {total_steps} total steps, {warmup_steps} warmup steps")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            
            # Compute loss
            # Shift logits and labels for next-token prediction
            shift_logits = logits
            shift_labels = labels
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            # Scale loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update stats
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Update parameters if we've accumulated enough gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                step += 1
                
                # Log progress
                if step % log_interval == 0:
                    avg_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    print(f"Step {step}/{total_steps} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.2f}s")
                    total_loss = 0
                    start_time = time.time()
                
                # Evaluate on validation set
                if step % eval_interval == 0:
                    val_loss = evaluate(model, val_dataloader, tokenizer, device)
                    print(f"Validation loss: {val_loss:.4f}")
                    
                    # Save the best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(f"New best validation loss: {best_val_loss:.4f}")
                        
                        # Save the model
                        model_save_path = os.path.join(output_dir, "best_model")
                        os.makedirs(model_save_path, exist_ok=True)
                        model.save_pretrained(model_save_path)
                        tokenizer.save_pretrained(model_save_path)
                        print(f"Saved best model to {model_save_path}")
                
                # Save a checkpoint
                if step % save_interval == 0:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save the final model
    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return best_val_loss


def evaluate(model, dataloader, tokenizer, device):
    """
    Evaluate a model on a validation set.
    
    Args:
        model: The model to evaluate
        dataloader: The validation dataloader
        tokenizer: The tokenizer
        device: The device to evaluate on
    
    Returns:
        The average validation loss
    """
    model.eval()
    total_loss = 0
    total_examples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            
            # Compute loss
            # Shift logits and labels for next-token prediction
            shift_logits = logits
            shift_labels = labels
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            # Update stats
            total_loss += loss.item() * input_ids.size(0)
            total_examples += input_ids.size(0)
    
    # Reset model to training mode
    model.train()
    
    # Compute average loss
    avg_loss = total_loss / total_examples
    return avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model with supervised fine-tuning")
    parser.add_argument("--model-path", type=str, default="/data/Meta-Llama-3-8B", help="Path to the pre-trained model")
    parser.add_argument("--train-path", type=str, default="/home/shared/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz", help="Path to the training data")
    parser.add_argument("--val-path", type=str, default="/home/shared/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz", help="Path to the validation data")
    parser.add_argument("--output-dir", type=str, default="outputs/sft", help="Directory to save the trained model")
    parser.add_argument("--seq-length", type=int, default=512, help="The sequence length to use")
    parser.add_argument("--batch-size", type=int, default=2, help="The batch size for training")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Number of steps to accumulate gradients")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="The learning rate for training")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Ratio of total steps to use for warmup")
    parser.add_argument("--log-interval", type=int, default=10, help="How often to log training progress (in steps)")
    parser.add_argument("--eval-interval", type=int, default=100, help="How often to evaluate on validation data (in steps)")
    parser.add_argument("--save-interval", type=int, default=500, help="How often to save model checkpoints (in steps)")
    parser.add_argument("--device", type=str, default="cuda", help="The device to train on")
    
    args = parser.parse_args()
    
    best_val_loss = train(
        args.model_path,
        args.train_path,
        args.val_path,
        args.output_dir,
        args.seq_length,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.epochs,
        args.learning_rate,
        args.warmup_ratio,
        args.log_interval,
        args.eval_interval,
        args.save_interval,
        args.device
    )