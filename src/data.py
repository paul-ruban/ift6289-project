# Inspired by https://github.com/facebookresearch/SpanBERT/blob/main/code/run_squad.py

import json
import string
import logging

import torch
from torch.utils.data import Dataset
import transformers



def preprocess_function(examples, tokenizer, max_length=256): 
    """ Function used for mapping SQUAD datapoints to model inputs. 

    Args:
        examples (Union[Dict, Dict[List]]): a datapoint or a batch of datapoints
        tokenizer (PreTrainedTokenizer): tokenizer
        max_length (int): max sequence length

    Returns:
        Union[Dict, Dict[List]]: a preprocessed datapoint or a batch of datapoints
    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        text=questions,
        text_pair=examples["context"],
        max_length=max_length,
        truncation="only_second",
        return_offsets_mapping=True
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0] if answer["answer_start"] else 0
        end_char = answer["answer_start"][0] + len(answer["text"][0]) if answer["answer_start"] else 0
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 2

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs