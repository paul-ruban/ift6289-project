# Inspired by https://github.com/facebookresearch/SpanBERT/blob/main/code/run_squad.py and 
# https://colab.research.google.com/drive/1Xqx3Mp7EESpWC-laFJWGuC5hrLLeJEOg#scrollTo=jCQoq2MVkR0K

import collections
import numpy as np
from tqdm import tqdm


MAX_LENGTH = 384
STRIDE = 128


def preprocess_train_dataset(examples, tokenizer, max_length=MAX_LENGTH, stride=STRIDE): 
    """ Function used for mapping SQUAD datapoints to model inputs. 

    Args:
        examples (Union[Dict, Dict[List]]): a datapoint or a batch of datapoints
        tokenizer (PreTrainedTokenizer): tokenizer
        max_length (int): max sequence length
        stride (int): overlap between two part of the context when splitting it is needed

    Returns:
        Union[Dict, Dict[List]]: a preprocessed datapoint or a batch of datapoints
    """

    pad_on_right = tokenizer.padding_side == "right"

    examples["question"] = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        text=examples["question" if pad_on_right else "context"],
        text_pair=examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=stride,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        padding="max_length"
    )

    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")
    
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = inputs.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if not answers["answer_start"]:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)
                # Note: we could go after the last offset if the answer is the last word.
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


def preprocess_eval_dataset(examples, tokenizer, max_length=MAX_LENGTH, stride=STRIDE): 
    """ Function used for mapping SQUAD datapoints to model inputs. 

    Args:
        examples (Union[Dict, Dict[List]]): a datapoint or a batch of datapoints
        tokenizer (PreTrainedTokenizer): tokenizer
        max_length (int): max sequence length
        stride (int): overlap between two part of the context when splitting it is needed

    Returns:
        Union[Dict, Dict[List]]: a preprocessed datapoint or a batch of datapoints
    """

    pad_on_right = tokenizer.padding_side == "right"

    examples["question"] = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        text=examples["question" if pad_on_right else "context"],
        text_pair=examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=stride,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        padding="max_length"
    )

    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    input_ids = inputs["input_ids"]
    example_id = []

    for i in range(len(input_ids)):
        # find where the context and question are
        sequence_ids = inputs.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # index of the example containing the span of text.
        sample_index = sample_mapping[i]
        example_id.append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        inputs["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(inputs["offset_mapping"][i])
        ]

    inputs["example_id"] = example_id

    return inputs


def post_process_function(
    examples,
    features, 
    raw_predictions, 
    tokenizer, 
    dataset_type, 
    n_best=20, 
    max_answer_length=30
):
    """ Function used for mapping model predictions to SQUAD datapoints.
    
    Args:
        examples (Union[Dict, Dict[List]]): a datapoint or a batch of datapoints
        features (Union[Dict, Dict[List]]): a datapoint or a batch of datapoints
        raw_predictions (Union[Dict, Dict[List]]): a datapoint or a batch of datapoints with the model predictions
        tokenizer (PreTrainedTokenizer): tokenizer
        dataset_type (str): type of the dataset (squad, squad_v2)
        n_best (int): number of best predictions to keep
        max_answer_length (int): max length of the answer
        
    Returns:
        Dict: a datapoint or a batch of datapoints
    """
    
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            # TODO : check if we can use the start_logits and end_logits directly.
            if feature_index >= len(all_start_logits):
                feature_index = len(all_start_logits) - 1
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -(n_best+1) : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -(n_best+1) : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping) or
                        end_index >= len(offset_mapping) or
                        not offset_mapping[start_index] or
                        not offset_mapping[end_index]
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if dataset_type == "squad":
            predictions[example["id"]] = best_answer["text"]
        elif dataset_type == "squad_v2":
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ''
            predictions[example["id"]] = answer
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    return predictions