# Usage: python run.py -c ../config/config.json
# Debug: python -m debugpy --listen 5678 run.py -c ../config/config.json

import os
import sys
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
from datasets import load_dataset, load_metric
from transformers import (
    TrainingArguments, 
    AutoModelForQuestionAnswering,
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
    model = AutoModelForQuestionAnswering.from_pretrained(train_config.model)
    
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
    trainer.evaluate()


    # Save model
    trainer.save_model()


def get_dtype(dtype):
    if dtype == "qint8":
        return torch.qint8
    elif dtype == "qint32":
        return torch.qint32
    elif dtype == "qint64":
        return torch.qint64
    elif dtype == "float16":
        return torch.float16
    else:
        raise ValueError("Invalid dtype.")

        
# Run main
if __name__ == "__main__":
    main()