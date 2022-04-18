import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertSelfAttention
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


class MultiHeadSelfAttentionGated(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config=config)

        dims = config.hidden_size // self.num_attention_heads
        self.gates = L0Linear(in_features=dims, out_features=dims)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        # apply gates
        context_layer, l0_penalty = self.gates(context_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)


        # hack to push l0 penalty with model outputs
        outputs = (context_layer, l0_penalty) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs



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
        # make pruning permanent
        prune.remove(modules_to_prune[0], modules_to_prune[1])
    
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
        
        if isinstance(module, old):
            setattr(model, n, new.to(next(module.parameters()).device))


def gate_model(model):
  replace_layers(
    model, 
    BertSelfAttention, 
    MultiHeadSelfAttentionGated(config=model.config)
  )
  return model

