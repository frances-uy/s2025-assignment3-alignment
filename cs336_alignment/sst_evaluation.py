"""
SimpleSafetyTests evaluation script using Qwen with CSV input.
"""
import os
import csv
import json
import time
import argparse
from vllm import LLM, SamplingParams


def load_sst_data_from_csv(csv_path):
    """Load SimpleSafetyTests examples from a CSV file."""
    examples = []
    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            instruction = row.get("instruction", "").strip()
            if instruction:
                examples.append({"instruction": instruction})
    print(f"Loaded {len(examples)} examples from CSV.")
    return examples


def format_sst_prompt(example, system_prompt):
    """Format prompt with system instructions."""
    instruction = example.get("instruction", "")
    return f"{system_prompt}\n# Query:\n{instruction}\n# Answer:"


def generate_sst_outputs(
    model_path,
    csv_input_path,
    output_path,
    system_prompt_path,
    max_examples=None
):
    """Generate model outputs for SimpleSafetyTests from CSV."""
    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read().strip()

    examples = load_sst_data_from_csv(csv_input_path)

    if max_examples is not None:
        examples = examples[:max_examples]

    prompts = [format_sst_prompt(example, system_prompt) for example in examples]

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["# Query:"]
    )

    print(f"Loading model from {model_path}")
    llm = LLM(model=model_path)

    print("Generating responses...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    total_time = end_time - start_time
    examples_per_second = len(prompts) / total_time
    print(f"Generated {len(prompts)} responses in {total_time:.2f} seconds")
    print(f"Throughput: {examples_per_second:.2f} examples/second")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for i, output in enumerate(outputs):
            response = output.outputs[0].text.strip()
            f.write(json.dumps({
                "prompts_final": examples[i]["instruction"],
                "output": response
            }) + "\n")

    print(f"Saved outputs to {output_path}")
    return examples_per_second


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen on SimpleSafetyTests")
    parser.add_argument("--model-path", type=str, required=True, help="Path to Qwen model")
    parser.add_argument("--csv-input", type=str, required=True, help="Path to CSV input file")
    parser.add_argument("--output-path", type=str, default="outputs/sst/model_predictions.jsonl")
    parser.add_argument("--system-prompt", type=str, default="cs336_alignment/prompts/system_prompt.txt")
    parser.add_argument("--max-examples", type=int, default=None)

    args = parser.parse_args()

    eps = generate_sst_outputs(
        args.model_path,
        args.csv_input,
        args.output_path,
        args.system_prompt,
        args.max_examples
    )

    print("\nTo evaluate safety with LLaMA 3 70B Instruct, run:")
    print(f"python scripts/evaluate_safety.py \\")
    print(f"  --input-path {args.output_path} \\")
    print("  --model-name-or-path /home/shared/Meta-Llama-3-70B-Instruct \\")
    print("  --num-gpus 2 \\")
    print("  --output-path outputs/sst/evaluation_results.jsonl")
