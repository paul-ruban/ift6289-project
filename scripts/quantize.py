# Usage: python quantize.py --c ../config/quant.json
# Debug: python -m debugpy --listen 5678 quantize.py ../config/quant.json

import os
import sys
# Adds project to path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)

import json
import argparse
import shutil

import torch
from transformers import AutoModelForQuestionAnswering
from src.quantization import get_dtype


def read_config_quant(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def check_input_dir(input_dir):
    required_files = [
        "config.json",
        "vocab.txt",
        # "tokenizer.json",
        # "tokenizer_config.json",
        # "special_tokens_map.json",
        "pytorch_model.bin"
    ]
    if not os.path.exists(input_dir):
        raise ValueError("Input directory does not exist.")
    
    files_in_dir = os.listdir(input_dir)

    if set(required_files) - set(files_in_dir):
        raise ValueError("Input directory does not contain all required files.")    


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Config file.", required=True)
    args = parser.parse_args()

    # Load config
    config_quant = read_config_quant(args.config)

    input_dir = config_quant["input_dir"]
    dtype = get_dtype(config_quant["dtype"])
    output_dir = config_quant["output_dir"]

    # Check input directory
    check_input_dir(input_dir)

    assert (input_dir != output_dir), "Input and output directories cannot be the same."

    # Load model
    model = AutoModelForQuestionAnswering.from_pretrained(input_dir)

    # Quantize model
    quantized_model = torch.quantization.quantize_dynamic(model, dtype=dtype)

    # Save quantized model
    os.makedirs(output_dir, exist_ok=True)
    quantized_model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(quantized_model.state_dict(), quantized_model_path)

    # Copy config file
    shutil.copy(args.config, output_dir)

    # Copy other files
    shutil.copy(os.path.join(input_dir, "config.json"), output_dir)
    shutil.copy(os.path.join(input_dir, "vocab.txt"), output_dir)
    # shutil.copy(os.path.join(input_dir, "tokenizer.json"), output_dir)
    # shutil.copy(os.path.join(input_dir, "tokenizer_config.json"), output_dir)
    # shutil.copy(os.path.join(input_dir, "special_tokens_map.json"), output_dir)

    print("Quantized model saved to {}".format(output_dir))

# Run main
if __name__ == "__main__":
    main()