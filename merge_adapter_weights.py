from dataclasses import dataclass, field
from typing import Optional
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, HfArgumentParser

"""
Merge Adapter Weights Script

This script merges the adapter weights of a PEFT (Parameter-Efficient Fine-Tuning) model
with the base model and saves the merged model. It also provides an option to save the
tokenizer and push the merged model to the Hugging Face Hub.

Usage:
    python scripts/merge_adapter_weights.py --peft_model_id <model_id> --output_dir <output_path>

Arguments:
    --peft_model_id: The model ID of the PEFT model to merge (default: "philschmid/instruct-igel-001").
    --output_dir: The directory where the merged model should be saved (default: "merged-weights").
    --save_tokenizer: Whether to save the tokenizer along with the merged model (default: True).
    --push_to_hub: Whether to push the merged model to the Hugging Face Hub (default: False).
    --repository_id: The repository ID for pushing the model to the Hugging Face Hub (required if --push_to_hub is True).

Example:
    python scripts/merge_adapter_weights.py --peft_model_id philschmid/instruct-igel-001 --output_dir merged-weights --save_tokenizer True
"""

@dataclass
class ScriptArguments:
    peft_model_id: Optional[str] = field(default="philschmid/instruct-igel-001", metadata={"help": "The model ID of the PEFT model to merge."})
    output_dir: Optional[str] = field(default="merged-weights", metadata={"help": "The directory where the merged model should be saved."})
    save_tokenizer: Optional[bool] = field(default=True, metadata={"help": "Whether to save the tokenizer along with the merged model."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Whether to push the merged model to the Hugging Face Hub."})
    repository_id: Optional[str] = field(default=None, metadata={"help": "The repository ID for pushing the model to the Hugging Face Hub."})

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    # Merge adapter weights and base model
    model = model.merge_and_unload()

    # Save the merged model
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="4GB")

    if args.save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.peft_model_id)
        tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        if args.repository_id is None:
            raise ValueError("You must specify a repository ID to push the model to the Hugging Face Hub.")
        
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.repository_id,
            repo_type="model",
        )

if __name__ == "__main__":
    main()