"""
MMLU evaluation script for both zero-shot and fine-tuned models.
"""
import os
import json
import re
import random
import time
import argparse
import torch
from vllm import LLM, SamplingParams


def load_mmlu_data(dataset_path="data/mmlu"):
    """Load MMLU evaluation data."""
    examples = []
    
    # Iterate through all files in the dataset directory
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    examples.extend(data)
    
    print(f"Loaded {len(examples)} MMLU examples")
    return examples


def format_mmlu_prompt(example, system_prompt, use_sft_format=False, alpaca_prompt=None):
    """Format an MMLU example as a prompt for the model."""
    # Extract fields from the example
    subject = example.get("subject", "")
    question = example.get("question", "")
    options = example.get("options", [])
    
    # Create the formatted question with options
    formatted_question = f"Answer the following multiple choice question about {subject}. Respond with a single sentence of the form \"The correct answer is _\", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).\n\nQuestion: {question}\n"
    
    # Add options with letters
    for i, option in enumerate(options):
        formatted_question += f"{chr(65 + i)}. {option}\n"
    
    formatted_question += "Answer:"
    
    if use_sft_format and alpaca_prompt:
        # If using SFT format, use the Alpaca template
        prompt = alpaca_prompt.replace("{prompt}", formatted_question).replace("{response}", "")
        full_prompt = f"# Query:\n{prompt}\n# Answer:"
    else:
        # Otherwise, use the standard format
        full_prompt = f"# Query:\n{formatted_question}\n# Answer:"
    
    # Prepend the system prompt
    return system_prompt + "\n" + full_prompt


def parse_mmlu_response(response_text):
    """Parse the model's response to extract the predicted answer (A, B, C, or D)."""
    # Use regex to find patterns like "The correct answer is X" or just "X"
    patterns = [
        r"[Tt]he correct answer is ([A-Da-d])",  # Match "The correct answer is X"
        r"[Tt]he answer is ([A-Da-d])",          # Match "The answer is X"
        r"[Aa]nswer:? ([A-Da-d])",               # Match "Answer: X" or "answer X"
        r"^\s*([A-Da-d])\.?\s*$"                 # Match just "X" or "X." on a line
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text)
        if matches:
            # Return the first match, converted to uppercase
            return matches[0].upper()
    
    # If we couldn't parse a letter, return None
    return None


def evaluate_mmlu(
    model_path, 
    output_dir, 
    system_prompt_path, 
    use_sft_format=False, 
    alpaca_prompt_path=None, 
    max_examples=None
):
    """Evaluate a model on MMLU."""
    # Load the system prompt
    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read().strip()
    
    # Load alpaca prompt if using SFT format
    alpaca_prompt = None
    if use_sft_format and alpaca_prompt_path:
        with open(alpaca_prompt_path, 'r') as f:
            alpaca_prompt = f.read().strip()
    
    # Load MMLU data
    examples = load_mmlu_data()
    
    # Limit number of examples if specified
    if max_examples is not None:
        examples = examples[:max_examples]
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["# Query:"]  # Stop when the model starts a new query
    )
    
    # Initialize the model
    print(f"Loading model from {model_path}")
    llm = LLM(model=model_path)
    
    # Format prompts for all examples
    prompts = [format_mmlu_prompt(
        example, system_prompt, use_sft_format, alpaca_prompt
    ) for example in examples]
    
    # Generate responses
    print("Generating responses...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    total_time = end_time - start_time
    examples_per_second = len(examples) / total_time
    print(f"Generated {len(examples)} responses in {total_time:.2f} seconds")
    print(f"Throughput: {examples_per_second:.2f} examples/second")
    
    # Process responses and evaluate
    correct = 0
    unparseable = 0
    results = []
    
    for i, output in enumerate(outputs):
        example = examples[i]
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        # Parse the response
        predicted_answer = parse_mmlu_response(generated_text)
        
        # Get the correct answer (convert to letter)
        correct_idx = example.get("answer", 0)
        correct_answer = chr(65 + correct_idx)  # Convert to A, B, C, D
        
        # Check if the prediction is correct
        is_correct = False
        if predicted_answer is not None:
            is_correct = predicted_answer == correct_answer
            if is_correct:
                correct += 1
        else:
            unparseable += 1
        
        # Store the result
        results.append({
            "example": example,
            "prompt": prompt,
            "generated_text": generated_text,
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        })
    
    # Calculate accuracy
    accuracy = correct / len(examples)
    
    # Print results
    print(f"MMLU Accuracy: {accuracy:.4f}")
    print(f"Unparseable responses: {unparseable} ({unparseable/len(examples):.2%})")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "mmlu_results.json"), 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "unparseable": unparseable,
            "examples_per_second": examples_per_second,
            "results": results
        }, f, indent=2)
    
    # Find incorrect examples for analysis
    incorrect = [r for r in results if not r["is_correct"] and r["predicted_answer"] is not None]
    if incorrect:
        # Sample 10 random incorrect examples
        sampled = random.sample(incorrect, min(10, len(incorrect)))
        print("\nSample of incorrect predictions:")
        for i, ex in enumerate(sampled):
            print(f"\nExample {i+1}:")
            print(f"Subject: {ex['example']['subject']}")
            print(f"Question: {ex['example']['question']}")
            options = ex['example']['options']
            for j, opt in enumerate(options):
                letter = chr(65 + j)
                print(f"{letter}. {opt}")
            print(f"Correct answer: {ex['correct_answer']}")
            print(f"Predicted answer: {ex['predicted_answer']}")
            print(f"Model response: {ex['generated_text'][:100]}...")
    
    return results, accuracy, unparseable, examples_per_second


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on MMLU")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output-dir", type=str, default="outputs/mmlu", help="Directory to save results")
    parser.add_argument("--system-prompt", type=str, default="cs336_alignment/prompts/system_prompt.txt", help="Path to system prompt")
    parser.add_argument("--alpaca-prompt", type=str, default="cs336_alignment/prompts/alpaca_prompt.txt", help="Path to Alpaca prompt template")
    parser.add_argument("--use-sft-format", action="store_true", help="Use SFT format for prompts")
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum number of examples to evaluate")
    
    args = parser.parse_args()
    
    results, accuracy, unparseable, examples_per_second = evaluate_mmlu(
        args.model_path,
        args.output_dir,
        args.system_prompt,
        args.use_sft_format,
        args.alpaca_prompt,
        args.max_examples
    )