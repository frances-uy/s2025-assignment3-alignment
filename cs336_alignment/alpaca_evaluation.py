"""
AlpacaEval evaluation script with debug statements.
"""
import os
import json
import time
import argparse
from vllm import LLM, SamplingParams


def load_alpaca_eval_data(dataset_path="data/alpaca_eval"):
    """Load AlpacaEval evaluation data."""
    examples = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    examples.extend(data)
    print(f"[INFO] Loaded {len(examples)} AlpacaEval examples")
    return examples


def format_alpaca_prompt(example, system_prompt, use_sft_format=False, alpaca_prompt=None):
    """Format an AlpacaEval example as a prompt for the model."""
    instruction = example.get("instruction", "")
    if use_sft_format and alpaca_prompt:
        prompt = alpaca_prompt.replace("{prompt}", instruction).replace("{response}", "")
        full_prompt = f"# Query:\n{prompt}\n# Answer:"
    else:
        full_prompt = f"# Query:\n{instruction}\n# Answer:"
    return system_prompt + "\n" + full_prompt


def generate_alpaca_eval_outputs(
    model_path,
    output_path,
    system_prompt_path,
    model_name="llama-3-8b-base",
    use_sft_format=False,
    alpaca_prompt_path=None,
    max_examples=None
):
    # Load system prompt
    if not os.path.exists(system_prompt_path):
        print(f"[ERROR] System prompt file not found: {system_prompt_path}")
        return
    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read().strip()
    print(f"[INFO] Loaded system prompt: {repr(system_prompt[:100])}...")

    # Load Alpaca SFT format prompt if needed
    alpaca_prompt = None
    if use_sft_format and alpaca_prompt_path:
        if os.path.exists(alpaca_prompt_path):
            with open(alpaca_prompt_path, 'r') as f:
                alpaca_prompt = f.read().strip()
            print(f"[INFO] Loaded Alpaca prompt template.")
        else:
            print(f"[WARNING] Alpaca prompt template not found: {alpaca_prompt_path}")

    # Load evaluation data
    examples = load_alpaca_eval_data()
    if max_examples is not None:
        examples = examples[:max_examples]

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["# Query:"]
    )

    # Initialize model
    print(f"[INFO] Loading model from {model_path}")
    llm = LLM(model=model_path)

    prompts = [format_alpaca_prompt(e, system_prompt, use_sft_format, alpaca_prompt) for e in examples]
    print(f"[DEBUG] First prompt:\n{prompts[0]}\n{'='*40}")

    print("[INFO] Generating responses...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    total_time = end_time - start_time
    examples_per_second = len(examples) / total_time
    print(f"[INFO] Generated {len(outputs)} responses in {total_time:.2f} seconds")
    print(f"[INFO] Throughput: {examples_per_second:.2f} examples/second")

    alpaca_results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        if i < 2:  # Print first 2 for debugging
            print(f"[DEBUG] Output {i + 1}:\n{repr(generated_text)}\n{'-'*40}")

        alpaca_results.append({
            "instruction": examples[i].get("instruction", ""),
            "output": generated_text,
            "generator": model_name,
            "dataset": examples[i].get("dataset", "")
        })

    if not alpaca_results:
        print("[WARNING] No outputs generated to save.")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(alpaca_results, f, indent=2)
        print(f"[INFO] Saved AlpacaEval results to {output_path}")

    return examples_per_second


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate outputs for AlpacaEval")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output-path", type=str, default="outputs/alpaca_eval/model_predictions.json", help="Path to save results")
    parser.add_argument("--system-prompt", type=str, default="cs336_alignment/prompts/system_prompt.txt", help="Path to system prompt")
    parser.add_argument("--alpaca-prompt", type=str, default="cs336_alignment/prompts/alpaca_prompt.txt", help="Path to Alpaca prompt template")
    parser.add_argument("--model-name", type=str, default="llama-3-8b-base", help="Name identifier for the model")
    parser.add_argument("--use-sft-format", action="store_true", help="Use SFT format for prompts")
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum number of examples to evaluate")

    args = parser.parse_args()

    examples_per_second = generate_alpaca_eval_outputs(
        args.model_path,
        args.output_path,
        args.system_prompt,
        args.model_name,
        args.use_sft_format,
        args.alpaca_prompt,
        args.max_examples
    )

    print("\nTo evaluate the results with Llama 3 70B Instruct as annotator, run:")
    print(f"alpaca_eval --model_outputs {args.output_path} \\")
    print("  --annotators_config 'scripts/alpaca_eval_vllm_llama3_70b_fn' \\")
    print("  --base-dir '.'")
