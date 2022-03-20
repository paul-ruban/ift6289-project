# Inspired by https://github.com/facebookresearch/SpanBERT/blob/main/code/run_squad.py

import json
import string
import logging

import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class SQUADExample:
    """
    A datapoint for SQUAD datasets.
    """

    def __init__(self,
                 id,
                 context,
                 question,
                 answer=None,
                 answer_start=None,
                 answer_end=None):
        self.id = id
        self.context = context
        self.question = question
        self.answer = answer
        self.answer_start = answer_start
        self.answer_end = answer_end

    def __str__(self):
        s = f"""id: {self.id},
        question: {self.question},
        context: {self.context},
        answer: {self.answer},
        answer_start: {self.answer_start},
        answer_end: {self.answer_end}
        """
        return s


class InputFeatures:
    """Features extracted from the datapoints."""

    def __init__(self,
                 id,
                 tokens,
                 input_ids,
                 segment_ids,
                 start_position=None,
                 end_position=None):
        self.id = id
        self.tokens = tokens
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position


def read_squad_examples(input_file, split):
    """Read a SQuAD json file into a list of SQUADSample."""
    assert (split in ["train", "dev"]), "split must be one of: 'train', 'dev'"

    with open(input_file, "r", encoding='utf-8') as f:
        input_data = json.load(f)["data"]

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True

            # normalize whitespaces
            for c in context:
                if c in string.whitespace:
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                id = qa["id"]
                question = qa["question"]
                answer = None
                answer_start = None
                answer_end = None
                if split == "train":
                    if qa.get("is_impossible"):
                        answer = ''
                        answer_start = -1
                        answer_end = -1
                    else:
                        assert (len(qa["answers"]) == 1), "For training, each question should have exactly 1 answer."
                        current_answer = qa["answers"][0]
                        answer = answer["text"]
                        answer_offset = current_answer["answer_start"]
                        answer_start = char_to_word_offset[answer_offset]
                        answer_end = char_to_word_offset[answer_offset + len(answer) - 1]
                        original_answer = ' '.join(doc_tokens[answer_start:answer_end+1])
                        clean_answer = ' '.join(answer.split())
                        if clean_answer not in original_answer:
                            logger.warning(f"Could not find answer: {original_answer} vs. {clean_answer}")
                            continue

                examples.append(SQUADExample(id, context, question, answer, answer_start, answer_end))

    return examples


def extract_features_from_examples(examples, tokenizer, max_seq_len):
    """Extracts features from datapoints."""

    features = []

    for i, example in enumerate(examples):
        input_features = tokenizer(
            text=example.question,
            text_pair=example.context,
            max_length=max_seq_len,
            truncation="only_second",
            return_offsets_mapping=True,
            return_token_type_ids=True
        )
        
        input_ids = input_features["input_ids"]
        offset_mapping = input_features["offset_mapping"]
        tokens = input_features.tokens()
        segment_ids = input_features["token_type_ids"]

        # Find the start and end of the context
        context_start = segment_ids.index(1)
        context_end = len(segment_ids) - 1


        # If the answer is not fully inside the context, label it (0, 0)
        if (offset_mapping[context_start][0] > example.answer_end or 
            offset_mapping[context_end][1] < example.answer_start):
            start_position = 0
            end_position = 0
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset_mapping[idx][0] <= example.answer_start:
                idx += 1
            start_position = idx - 1

            idx = context_end
            while idx >= context_start and offset_mapping[idx][1] >= example.answer_end:
                idx -= 1
            end_position = idx + 1

        features.append(
            InputFeatures(i, tokens, input_ids, segment_ids, start_position, end_position)
        )

    return features

