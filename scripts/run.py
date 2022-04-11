# Usage: python run.py -c ../config/test.json
# Debug: python -m debugpy --listen 5678 run.py -c ../config/test.json

import argparse
from functools import partial
from src.config import TrainConfig
from src.trainer import SQUADTrainer # noqa: E501
from src.data import preprocess_function, post_process_function
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
    
    dataset = load_dataset(train_config.dataset_name)

    remove_columns = dataset["train"].column_names + ["offset_mapping"]

    # Preprocess dataset
    processed_dataset = dataset.map(
        partial(preprocess_function, tokenizer=tokenizer),
        batched=True
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
        load_best_model_at_end=True,
    )

    if train_config.compute_metrics:
        metric = load_metric(train_config.dataset_name)

    def compute_metrics(x):
        results = metric.compute(
            predictions=x.predictions, references=x.label_ids)
        # accomodate squad and squadv2 metrics
        return {"exact_match": results["exact"] if results.get("exact") else results["exact_match"], 
                "f1": results["f1"]}

    dataset_for_training = processed_dataset.remove_columns(remove_columns)

    trainer = SQUADTrainer(
        model=model,
        teacher_model=teacher_model,
        distillation_method=train_config.distillation_method,
        args=training_args,
        train_dataset=dataset_for_training["train"],
        eval_dataset=dataset_for_training["validation"],
        eval_dataset_reference=processed_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if train_config.compute_metrics else None,
        post_process_function=post_process_function,
        dataset_name=train_config.dataset_name,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Train model
    # trainer.train()

    # Evaluate model
    trainer.evaluate(
        eval_dataset=dataset_for_training["validation"],
        eval_dataset_reference=processed_dataset["validation"])

    # Save model
    trainer.save_model()


if __name__ == "__main__":
    main()