# Usage: python run.py -c ../config/config.json

import argparse
from functools import partial
from src.config import TrainConfig
from src.trainer import SQUADTrainer # noqa: E501
from src.data import preprocess_function
from datasets import load_dataset
from transformers import (
    TrainingArguments, 
    AutoModelForQuestionAnswering,
    AutoTokenizer
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
    
    dataset = load_dataset(train_config.data_path)

    # Preprocess dataset
    dataset = dataset.map(
        partial(preprocess_function, tokenizer=tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        evaluation_strategy=train_config.evaluation_strategy,
        logging_steps=train_config.logging_steps,
        learning_rate=train_config.learning_rate,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        num_train_epochs=train_config.num_train_epochs,
        weight_decay=train_config.weight_decay,
        optim=train_config.optim,
        disable_tqdm=train_config.disable_tqdm
    )

    trainer = SQUADTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer
    )

    # Train model
    trainer.train(resume_from_checkpoint=True)

    # Evaluate model
    trainer.evaluate(eval_dataset=dataset["validation"])

    # Save model
    trainer.save_model()


if __name__ == "__main__":
    main()