import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention
import torch
from typing import List, Dict, Tuple, Optional


import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))


class _L0Norm(nn.Module):

    def __init__(self, origin, loc_mean=0, loc_sdev=0.01, beta=2 / 3, gamma=-0.1,
                 zeta=1.1, fix_temp=True):
        """
        Base class of layers using L0 Norm
        :param origin: original layer such as nn.Linear(..), nn.Conv2d(..)
        :param loc_mean: mean of the normal distribution which generates initial location parameters
        :param loc_sdev: standard deviation of the normal distribution which generates initial location parameters
        :param beta: initial temperature parameter
        :param gamma: lower bound of "stretched" s
        :param zeta: upper bound of "stretched" s
        :param fix_temp: True if temperature is fixed
        """
        super(_L0Norm, self).__init__()
        self._origin = origin
        self._size = self._origin.weight.size()
        self.loc = nn.Parameter(torch.zeros(self._size).normal_(loc_mean, loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros(self._size))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)

    def _get_mask(self):
        if self.training:
            self.uniform.uniform_()
            u = Variable(self.uniform)
            s = F.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = F.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio).sum()
        else:
            s = F.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
            penalty = 0
        return hard_sigmoid(s), penalty


class L0Linear(_L0Norm):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(L0Linear, self).__init__(nn.Linear(in_features, out_features, bias=bias), **kwargs)

    def forward(self, input):
        mask, penalty = self._get_mask()
        return F.linear(input, self._origin.weight * mask, self._origin.bias), penalty


class MultiHeadSelfAttentionGated(MultiHeadSelfAttention):
    def __init__(self, config):
        super().__init__(config=config)
        # self.g = torch.nn.Linear(torch.zeros(self.n_heads, config.dim, config.dim))
        self.g = torch.nn.Linear(in_features=config.dim, out_features=config.dim)

        # g = torch.Tensor(16, 12, 384, 64)
        g = torch.randn(16, 12, 384, 64)
        # g = torch.bernoulli(g)
        self.g = torch.nn.Parameter(g)
        self.gates = L0Linear(64, 64)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)
        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores = scores.masked_fill(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)

        context = context.reshape(16*12*384, 64)

        #Apply the gates (L0Linear returns tensor and L0 penalty)
        context, L0penalty = L0Linear(context)[0]
        context = context.reshape(16,12,384,64)

        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
          return (context, L0penalty)
        else:
            return (context,)

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

def L0_regularization_term(model, get_biases=1):
  non_zeros = 0
  modules_to_prune = []
  for name, param in model.named_parameters():
    m = model.retrieve_modules_from_names([name])[0]
    modules_to_prune.append((m,"weight"))
    if get_biases:
      modules_to_prune.append((m,"bias"))
  for module in modules_to_prune:
    module = module[0]
    non_zeros += torch.sum(module.weight != 0)
  return non_zeros

def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)
        
        if isinstance(module, MultiHeadSelfAttention):
            setattr(model, n, new)


def gate_model(model):
  replace_layers(model, MultiHeadSelfAttention, MultiHeadSelfAttentionGated(config=model.config))
  return model

