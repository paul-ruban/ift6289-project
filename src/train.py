import torch
from torch import nn
from transformers import Trainer


class SQUADTrainer(Trainer):
    """ Overrides the Trainer class to perform regular Fine-tuning and distillation 
    """
    def __init__(self, teacher_model=None, **kwargs):
        super().__init__(**kwargs)
        if teacher_model is not None:
            if self.place_model_on_device:
                self._move_model_to_device(teacher_model, self.args.device)
            self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = 0

        outputs = model(**inputs)
        loss_squad = outputs["loss"]
        loss += loss_squad

        if self.teacher_model is not None:
            log_softmax_fn = torch.nn.LogSoftmax(dim=-1)
            loss_distil_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)

            start_log_inputs = log_softmax_fn(outputs["start_logits"])
            end_log_inputs = log_softmax_fn(outputs["end_logits"])

            start_log_targets = log_softmax_fn(outputs["start_logits"])
            end_log_targets = log_softmax_fn(outputs["end_logits"])

            loss_distil = loss_distil_fn(start_log_inputs, start_log_targets) + \
                          loss_distil_fn(end_log_inputs, end_log_targets)

            loss += loss_distil

        return (loss, outputs) if return_outputs else loss