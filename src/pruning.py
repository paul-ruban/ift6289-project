import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

class Pruner:
  def process_modules(self,get_biases=0):
    modules_to_prune = []
    for name, param in self.model.named_parameters():
      m = self.model.retrieve_modules_from_names([name])[0]
      modules_to_prune.append((m,"weight"))
      if get_biases:
        modules_to_prune.append((m,"bias"))
    return modules_to_prune

  def __init__(self, **args):
    self.model = args["model"]
    self.max_sparsity = args["max_sparsity"]
    self.pruning_config = args["pruning_config"]

    self.modules_to_prune = self.process_modules()

  def model_sparsity(self):
    sparsity = 0
    n_elements = 0
    for module in self.modules_to_prune:
      module = module[0]
      sparsity += torch.sum(module.weight == 0)
      n_elements += module.weight.nelement()

    return 100*float(sparsity)/(float(n_elements))

  def is_prunable(self):
    return self.model_sparsity() < self.max_sparsity

  def prune(self):
    if self.is_prunable():
      if self.pruning_config["magnitude"]["active"]:
        prune.global_unstructured(
            self.modules_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.pruning_config["magnitude"]["number_pruned"],
        )
      if self.pruning_config["random"]["active"]:
        prune.global_unstructured(
          self.modules_to_prune,
          pruning_method=prune.RandomUnstructured,
          amount=self.pruning_config["random"]["rate"],
      )

    return self.model


  def save_serialized_model(self):
    torch.save(self.model.state_dict(), ".")


def L0_regularization_term(self):
  n_zeros = 0
  for module in self.modules_to_prune:
    module = module[0]
    n_zeros += torch.sum(module.weight == 0)
  return n_zeros


