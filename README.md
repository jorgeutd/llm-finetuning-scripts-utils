# llm-finetuning-scripts-utils
Collection of training scripts for supervised fine-tuning and alignment of transformer language models. The scripts are designed to be run on Amazon SageMaker for efficient training of large language models.

## Overview
The scripts in this repository utilize techniques such as LoRA (Low-Rank Adaptation) and QLoRA (Quantized Low-Rank Adaptation) to efficiently fine-tune large language models. These techniques offer several benefits:

LoRA
Reduces memory footprint: LoRA achieves this by applying a low-rank approximation to the weight update matrix (ΔW), representing ΔW as the product of two smaller matrices, significantly reducing the number of parameters needed to store ΔW.
Fast fine-tuning: LoRA offers fast training times compared to traditional fine-tuning methods due to its reduced parameter footprint.
Maintains performance: LoRA has been shown to maintain performance close to traditional fine-tuning methods in several tasks.

QLoRA
Enhances parameter efficiency: QLoRA takes LoRA a step further by also quantizing the weights of the LoRA adapters (smaller matrices) to lower precision (e.g., 4-bit instead of 8-bit), further reducing the memory footprint and storage requirements.
More memory efficient: QLoRA is even more memory efficient than LoRA, making it ideal for resource-constrained environments.
Similar effectiveness: QLoRA has been shown to maintain similar effectiveness to LoRA in terms of performance, while offering significant memory advantages.
Getting Started
To get started with fine-tuning large language models using the scripts in this repository, follow these steps:

### Clone the repository:

git clone https://github.com/your-username/llm-finetuning-scripts-utils.git

### Install the required dependencies:
pip install -r requirements.txt

Prepare your dataset and configure the desired hyperparameters in the script of your choice.
Run the selected script on Amazon SageMaker or your preferred environment.
Scripts
The repository includes the following scripts:

merge_adapter_weights.py: Merges the adapter weights with the base model.
run_qlora.py: Performs fine-tuning using QLoRA.
trl/run_sft.py: Performs supervised fine-tuning using the Transformer Reinforcement Learning (TRL) library.
utils/pack_dataset.py: Packs and preprocesses the dataset for fine-tuning.
Examples
Examples and tutorials demonstrating how to use the scripts can be found in the examples/ directory.

License
This project is licensed under the MIT License.

Acknowledgements
The scripts in this repository are built upon the work of various researchers and open-source projects. We would like to acknowledge their contributions and the following resources:

LoRA: Low-Rank Adaptation of Large Language Models
QLoRA: Efficient Finetuning of Quantized LLMs
Hugging Face Transformers
bitsandbytes


### License

This project is licensed under the terms of the MIT license.