import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

class Pruner:
  def process_modules(self, model, get_biases=0):
    modules_to_prune = []
    for name, param in model.named_parameters():
      m = model.retrieve_modules_from_names([name])[0]
      modules_to_prune.append((m,"weight"))
      if get_biases:
        modules_to_prune.append((m,"bias"))
    return modules_to_prune

  def __init__(self, **args):
    self.pruning_config = args["pruning_config"]
    
  def current_model_sparsity(self, modules_to_prune):
    sparsity = 0
    n_elements = 0
    for module in modules_to_prune:
      module = module[0]
      sparsity += torch.sum(module.weight == 0)
      n_elements += module.weight.nelement()
    
   
    return float(sparsity)/(float(n_elements))

  def is_prunable(self, modules_to_prune):
    sparsity = self.current_model_sparsity(modules_to_prune)
    print(f"\n The current model sparsity is {sparsity}.")

    if self.pruning_config["random"]["active"]:
      n_elements = 0
      for module in modules_to_prune:
        n_elements += module[0].weight.nelement()

      number_zeros = sparsity*n_elements

      if sparsity == 0:
        futur_sparsity = float(self.pruning_config["random"]["rate"])
      else:
        futur_number_zeros = number_zeros*(1+self.pruning_config["random"]["rate"])
        futur_sparsity = float(futur_number_zeros)/float(n_elements)

      print(f"After pruning the sparsity will be {futur_sparsity}.")

      if futur_sparsity < self.pruning_config["max_sparsity"]:
        print(f"Futur sparsity ({futur_sparsity}) does not exceed\
                the max sparsity ({self.pruning_config['max_sparsity']})\
                : the model is prunable.)") 
      else:
        print(f"futur sparsity ({futur_sparsity}) exceeds\
              the max sparsity ({self.pruning_config['max_sparsity']})\
              : the model is not prunable.") 
            
      return futur_sparsity < self.pruning_config["max_sparsity"]
    
    else:
      if sparsity < self.pruning_config["max_sparsity"]:
        print(f"The model is prunable. Current sparsity of {sparsity}")
      else:
        print(f"The model is not prunable. Current sparsity of {sparsity}")

      return sparsity < self.pruning_config["max_sparsity"]
    

  def prune(self, model):
    with torch.no_grad():
      modules_to_prune = self.process_modules(model)

      if self.is_prunable(modules_to_prune):
        print("Pruning model...")
        if self.pruning_config["magnitude"]["active"]:
          prune.global_unstructured(
              modules_to_prune,
              pruning_method=prune.L1Unstructured,
              amount=self.pruning_config["magnitude"]["number_pruned"],
          )
        if self.pruning_config["random"]["active"]:
          prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=self.pruning_config["random"]["rate"],
        )
    
    print(f"The model has been pruned!")
    return model


def L0_regularization_term(self):
  n_zeros = 0
  for module in self.modules_to_prune:
    module = module[0]
    n_zeros += torch.sum(module.weight == 0)
  return n_zeros


