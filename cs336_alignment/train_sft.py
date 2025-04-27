# filename: cs336_alignment/train_sft.py

import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from tqdm import tqdm

from cs336_alignment.adapters import get_packed_sft_dataset, run_iterate_batches

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
    )
    model.to(device)

    # Load dataset
    train_dataset = get_packed_sft_dataset(
        tokenizer, args.train_dataset, args.seq_length, shuffle=True
    )
    val_dataset = get_packed_sft_dataset(
        tokenizer, args.val_dataset, args.seq_length, shuffle=False
    )

    train_loader = run_iterate_batches(train_dataset, args.batch_size, shuffle=True)
    val_loader = run_iterate_batches(val_dataset, args.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_loader) * args.num_epochs // args.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.03 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    model.train()
    step = 0

    for epoch in range(args.num_epochs):
        total_loss = 0.0
        for idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            logits = outputs.logits

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            loss = loss / args.grad_accum_steps
            loss.backward()

            if (idx + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            step += 1

        print(f"Epoch {epoch} - Training Loss: {total_loss/len(train_loader):.4f}")

        # Validation loss
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} - Validation Loss: {val_loss:.4f}")

    # Save model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            logits = outputs.logits

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            total_loss += loss.item()

    return total_loss / len(val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--train-dataset", type=str, required=True)
    parser.add_argument("--val-dataset", type=str, required=True)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()

    train(args)
