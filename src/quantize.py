import numpy as np
import copy
import torch
from datasets import load_dataset
from transformers import DistilBertForQuestionAnswering, AutoTokenizer

max_float32_val = torch.tensor(3.4028235e38)
min_float32_val = torch.tensor(1.175494351e-38)

def quantization(x, scale, zeropoint, alpha_q, beta_q):
    x_q = np.round(1 / scale * x + zeropoint, decimals=0)
    x_q = np.clip(x_q, a_min=alpha_q, a_max=beta_q)
    return x_q

def quantization_float32_int8(x, scale, zeropoint):
    if not (torch.is_tensor(x) and x.dtype == torch.float32):
      raise ValueError('Input x has to be float32 tensor')
    x_q = quantization(x, scale, zeropoint, alpha_q=-128, beta_q=127)
    x_q = x_q.to(torch.int8)
    return x_q

def dequantization_int8_float32(x_q, scale, zeropoint):
    if not (torch.is_tensor(x_q) and x_q.dtype == torch.int8):
      raise ValueError('Input x_q has to be int8 tensor')
    x = scale * (x_q - zeropoint)
    x = x.to(torch.float32)
    return x

def generate_quantization_constants(alpha, beta, alpha_q, beta_q):
    # Affine quantization mapping
    scale = (beta - alpha) / (beta_q - alpha_q)
    zeropoint = int((beta * alpha_q - alpha * beta_q) / (beta - alpha))
    return scale, zeropoint

def generate_quantization_int8_constants(alpha, beta):
    scale, zeropoint = generate_quantization_constants(alpha=alpha,
                                           beta=beta,
                                           alpha_q=-128,
                                           beta_q=127)
    return scale, zeropoint

def min_max(state_dict):
  min_max = {}
  for key in state_dict:
    min_max[key] = (torch.min(state_dict[key]), torch.max(state_dict[key]))
  return min_max

def prepare_qa_inputs(question, text, tokenizer, device=None):
    inputs = tokenizer(question, text, return_tensors="pt")
    if device is not None:
        inputs_cuda = dict()
        for input_name in inputs.keys():
            inputs_cuda[input_name] = inputs[input_name].to(device)
        inputs = inputs_cuda
    return inputs

def move_inputs_to_device(inputs, device=None):
    inputs_cuda = dict()
    for input_name in inputs.keys():
        inputs_cuda[input_name] = inputs[input_name].to(device)
    return inputs_cuda

def run_qa(model, tokenizer, question, text, device=None):
    inputs = prepare_qa_inputs(question=question, text=text, tokenizer=tokenizer)
    all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0])
    if device is not None:
        inputs = move_inputs_to_device(inputs, device=device)
        model = model.to(device)
    outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    answer_start_idx = torch.argmax(start_scores, 1)[0]
    answer_end_idx = torch.argmax(end_scores, 1)[0] + 1
    answer = " ".join(all_tokens[answer_start_idx : answer_end_idx])
    return answer


def main():
  model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
  tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
  squad = load_dataset("squad")

  quantized_model = copy.deepcopy(model)
  theta = quantized_model.state_dict()
  theta_min_max = min_max(theta)

  # quantize state_dict
  for name in theta:
    theta[name] = quantization_float32_int8(theta[name], theta_min_max[name][0], theta_min_max[name][1])

  # dequantize state_dict
  for name in theta:
    theta[name] = dequantization_int8_float32(theta[name], theta_min_max[name][0], theta_min_max[name][1])

  quantized_model.load_state_dict(theta)

  # testing
  question = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"

  text = "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend Venite Ad Me Omnes. Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."

  inputs = prepare_qa_inputs(question=question, text=text, tokenizer=tokenizer)
  answer = run_qa(model=model, tokenizer=tokenizer, question=question, text=text)
  answer2 = run_qa(model=quantized_model, tokenizer=tokenizer, question=question, text=text)

  print(answer)
  print(answer2)

if __name__ == "__main__":
    main()

