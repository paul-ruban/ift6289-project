# Usage: python run.py -c ../config/config.json
# Debug: python -m debugpy --listen 5678 run.py -c ../config/config.json

import os
import sys
import json
# Adds project to path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)

import torch
import argparse
import shutil
from functools import partial
from src.config import TrainConfig
from src.trainer import SQUADTrainer # noqa: E501
from src.data import (
    preprocess_train_dataset, 
    preprocess_eval_dataset, 
    post_process_function
)
from src.quantization import is_quantized, infer_quantization_dtype, get_dtype
from datasets import load_dataset, load_metric
from transformers import (
    TrainingArguments, 
    AutoModelForQuestionAnswering,
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback
)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Config file.", required=True)
    args = parser.parse_args()

    # Load config
    train_config = TrainConfig.from_json(args.config)

    # copy config to log dir
    os.makedirs(train_config.output_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(train_config.output_dir, "trainer_config.json"))

    tokenizer = AutoTokenizer.from_pretrained(train_config.model)
    
    quantized_flag = False
 
    if not os.path.isdir(train_config.model):
        model = AutoModelForQuestionAnswering.from_pretrained(train_config.model)
    else:
        model_config = AutoConfig.from_pretrained(train_config.model)
        model = AutoModelForQuestionAnswering.from_config(model_config)

        # Load state dict
        state_dict = torch.load(os.path.join(train_config.model, "pytorch_model.bin"))
        if is_quantized(state_dict):
            quantized_flag = True
            quantized_dtype = infer_quantization_dtype(state_dict)
            model = torch.quantization.quantize_dynamic(model, dtype=quantized_dtype)
        
        load_result = model.load_state_dict(state_dict)
        print(load_result)


    teacher_model = None
    if train_config.teacher_model is not None:
        teacher_model = AutoModelForQuestionAnswering.from_pretrained(train_config.teacher_model)
    
    dataset_name = train_config.dataset_name
    dataset = load_dataset(dataset_name)

    # Preprocess dataset
    processed_dataset = dataset.map(
        partial(preprocess_train_dataset, tokenizer=tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        evaluation_strategy=train_config.evaluation_strategy,
        eval_steps=train_config.eval_steps,
        save_strategy=train_config.save_strategy,
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        learning_rate=train_config.learning_rate,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        num_train_epochs=train_config.num_train_epochs,
        weight_decay=train_config.weight_decay,
        optim=train_config.optim,
        disable_tqdm=train_config.disable_tqdm,
        metric_for_best_model=train_config.metric_for_best_model,
        load_best_model_at_end=True
    )

    # reprocess dataset for evaluation
    eval_features = dataset["validation"].map(
        partial(preprocess_eval_dataset, tokenizer=tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names
    )

    # Create trainer
    trainer = SQUADTrainer(
        model=model,
        teacher_model=teacher_model,
        distillation_method=train_config.distillation_method,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        eval_dataset_raw=dataset["validation"],
        eval_features=eval_features,
        tokenizer=tokenizer,
        compute_metrics=load_metric(dataset_name).compute if train_config.compute_metrics else None,
        post_process_function=post_process_function,
        dataset_name=train_config.dataset_name,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
        pruning_config=train_config.pruning_config
    )

    # Train model
    if train_config.do_train:
        trainer.train()

    if train_config.quantize:
        trainer.model = torch.quantization.quantize_dynamic(
            trainer.model, 
            dtype=get_dtype(train_config.quantize)
        )
    
    # Evaluate model
    eval_results = trainer.evaluate()
    
    # Save model
    if train_config.do_train:
      trainer.save_model()
    else:
      with open(os.path.join(train_config.output_dir, "distilbert_eval_results.txt"), "w") as f:
        json.dump(eval_results, f, indent=6)


# Run main
if __name__ == "__main__":
    main()