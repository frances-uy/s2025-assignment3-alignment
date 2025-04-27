"""
GSM8K evaluation script for both zero-shot and fine-tuned models using Qwen.
"""
import os
import json
import re
import random
import time
import argparse
from vllm import LLM, SamplingParams


def load_gsm8k_data(dataset_path="data/gsm8k", split="test.jsonl"):
    """Load GSM8K evaluation data from a .jsonl file."""
    examples = []
    file_path = os.path.join(dataset_path, split)

    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    print(f"Loaded {len(examples)} GSM8K examples from {split}")
    return examples


def format_gsm8k_prompt(example, system_prompt, use_sft_format=False, alpaca_prompt=None):
    """Format a GSM8K example as a prompt for the model."""
    question = example.get("question", "")
    formatted_question = f"{question}\nAnswer:"

    if use_sft_format and alpaca_prompt:
        prompt = alpaca_prompt.replace("{prompt}", formatted_question).replace("{response}", "")
        full_prompt = f"# Query:\n{prompt}\n# Answer:"
    else:
        full_prompt = f"# Query:\n{formatted_question}\n# Answer:"

    return system_prompt + "\n" + full_prompt


def parse_gsm8k_response(response_text):
    """Parse the model's response to extract the final numerical answer."""
    response_text = response_text.replace(",", "")
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response_text)
    if numbers:
        return float(numbers[-1])
    return None


def evaluate_gsm8k(
    model_path,
    output_dir,
    system_prompt_path,
    split="test.jsonl",
    dataset_path="data/gsm8k",
    use_sft_format=False,
    alpaca_prompt_path=None,
    max_examples=None
):
    """Evaluate a model on GSM8K."""
    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read().strip()

    alpaca_prompt = None
    if use_sft_format and alpaca_prompt_path:
        with open(alpaca_prompt_path, 'r') as f:
            alpaca_prompt = f.read().strip()

    examples = load_gsm8k_data(dataset_path, split)

    if max_examples is not None:
        examples = examples[:max_examples]

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["# Query:"]
    )

    print(f"Loading model from {model_path}")
    llm = LLM(model=model_path)

    prompts = [format_gsm8k_prompt(example, system_prompt, use_sft_format, alpaca_prompt)
               for example in examples]

    print("Generating responses...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    total_time = end_time - start_time
    examples_per_second = len(examples) / total_time
    print(f"Generated {len(examples)} responses in {total_time:.2f} seconds")
    print(f"Throughput: {examples_per_second:.2f} examples/second")

    correct = 0
    unparseable = 0
    results = []

    for i, output in enumerate(outputs):
        example = examples[i]
        prompt = output.prompt
        generated_text = output.outputs[0].text
        predicted_answer = parse_gsm8k_response(generated_text)
        correct_answer = example.get("answer", None)

        is_correct = False
        if predicted_answer is not None and correct_answer is not None:
            try:
                correct_answer_num = float(correct_answer)
                is_correct = abs(predicted_answer - correct_answer_num) < 1e-6
                if is_correct:
                    correct += 1
            except (ValueError, TypeError):
                is_correct = False
        else:
            unparseable += 1

        results.append({
            "example": example,
            "prompt": prompt,
            "generated_text": generated_text,
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        })

    accuracy = correct / len(examples)

    print(f"GSM8K Accuracy: {accuracy:.4f}")
    print(f"Unparseable responses: {unparseable} ({unparseable/len(examples):.2%})")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "gsm8k_results.json"), 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "unparseable": unparseable,
            "examples_per_second": examples_per_second,
            "results": results
        }, f, indent=2)

    incorrect = [r for r in results if not r["is_correct"] and r["predicted_answer"] is not None]
    if incorrect:
        sampled = random.sample(incorrect, min(10, len(incorrect)))
        print("\nSample of incorrect predictions:")
        for i, ex in enumerate(sampled):
            print(f"\nExample {i+1}:")
            print(f"Question: {ex['example']['question']}")
            print(f"Correct answer: {ex['correct_answer']}")
            print(f"Predicted answer: {ex['predicted_answer']}")
            print(f"Model response: {ex['generated_text'][:100]}...")

    return results, accuracy, unparseable, examples_per_second


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output-dir", type=str, default="outputs/gsm8k", help="Directory to save results")
    parser.add_argument("--system-prompt", type=str, default="cs336_alignment/prompts/system_prompt.txt", help="Path to system prompt")
    parser.add_argument("--alpaca-prompt", type=str, default="cs336_alignment/prompts/alpaca_prompt.txt", help="Path to Alpaca prompt template")
    parser.add_argument("--use-sft-format", action="store_true", help="Use SFT format for prompts")
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum number of examples to evaluate")
    parser.add_argument("--split", type=str, default="test.jsonl", help="Dataset split file (e.g., test.jsonl or train.jsonl)")
    parser.add_argument("--dataset-path", type=str, default="data/gsm8k", help="Path to the GSM8K dataset directory")

    args = parser.parse_args()

    results, accuracy, unparseable, examples_per_second = evaluate_gsm8k(
        args.model_path,
        args.output_dir,
        args.system_prompt,
        args.split,
        args.dataset_path,
        args.use_sft_format,
        args.alpaca_prompt,
        args.max_examples
    )
