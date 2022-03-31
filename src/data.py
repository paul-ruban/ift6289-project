# Inspired by https://github.com/facebookresearch/SpanBERT/blob/main/code/run_squad.py

import collections
from transformers import EvalPrediction
import numpy as np


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

    offset_mapping = inputs["offset_mapping"]
    answers = examples["answers"]
    answer_texts = []
    start_positions = []
    end_positions = []
    tokenized_inputs = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0] if answer["answer_start"] else 0
        end_char = answer["answer_start"][0] + len(answer["text"][0]) if answer["answer_start"] else 0
        answer_text = answer["text"][0] if answer["text"] else ''
        answer_texts.append(answer_text)
        tokenized_input = tokenizer.convert_ids_to_tokens(inputs["input_ids"][i])
        tokenized_inputs.append(tokenized_input)
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
    inputs["answer_texts"] = answer_texts
    inputs["tokenized_inputs"] = tokenized_inputs
    return inputs


def post_process_function(dataset, dataset_reference, predictions, tokenizer, dataset_type="squad"):
    """ Function used for mapping model predictions to SQUAD datapoints."""
    
    start_logits, end_logits = predictions
    assert (len(dataset) == len(start_logits)), "Dataset and predictions must have the same length"
    assert (dataset_type in ["squad", "squad_v2"]), f"Invalid dataset_type: {dataset_type}"

    # The dictionaries we have to fill.
    all_predictions = []

    for i, ref in enumerate(dataset_reference):
        candidate_predictions = []
        offset_mapping = ref["offset_mapping"]
        # no answer prediction
        start_ids = np.argsort(start_logits[i])[-1:-21:-1].tolist()
        end_ids = np.argsort(end_logits[i])[-1:-21:-1].tolist()
        for s_id in start_ids:
            for e_id in end_ids:
                if s_id >= len(offset_mapping) or e_id >= len(offset_mapping):
                    continue
                if e_id < s_id or e_id - s_id + 1 > 30:
                    continue
                pred_txt = tokenizer.convert_tokens_to_string(ref["tokenized_inputs"][s_id:e_id+1])
                score = start_logits[i][s_id] + end_logits[i][e_id]
                candidate_predictions.append(
                    dict(prediction_text=pred_txt, score=score)
                )
        
        if dataset_type == "squad_v2" or not candidate_predictions:
            no_answer_score = start_logits[i][0] + end_logits[i][0]
            candidate_predictions.append(
                dict(prediction_text="", score=no_answer_score)
            )
        
        best_prediction = sorted(candidate_predictions, key=lambda x: x["score"], reverse=True)[0]

        all_predictions.append(
            dict(id=dataset_reference["id"][i], prediction_text=best_prediction["prediction_text"])
        )

    # if dataset_type == "squad":
    #     predictions = [
    #         dict(id=k, prediction_text=v, no_answer_probability=0.0)
    #         for k, v in predictions.items() 
    #     ]
    # else:
    #     predictions = [
    #         dict(id=k, prediction_text=v) for k, v in predictions.items() 
    #     ]
    
    label_ids = [dict(id=ref["id"], answers=ref["answers"]) for ref in dataset_reference]
    return EvalPrediction(predictions=all_predictions, label_ids=label_ids)