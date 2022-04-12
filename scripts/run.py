# Usage: python run.py -c ../config/test.json
# Debug: python -m debugpy --listen 5678 run.py -c ../config/test.json

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Config file.", required=True)
    args = parser.parse_args()

    train_config = TrainConfig.from_json(args.config)

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
        compute_metrics=load_metric(dataset_name) if train_config.compute_metrics else None,
        post_process_function=post_process_function,
        dataset_name=train_config.dataset_name,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Train model
    trainer.train()

    # Evaluate model
    trainer.evaluate()

    # Save model
    trainer.save_model()


if __name__ == "__main__":
    main()