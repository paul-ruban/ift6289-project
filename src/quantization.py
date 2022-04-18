import torch

    
def get_dtype(dtype):
    if dtype == "qint8":
        return torch.qint8
    elif dtype == "qint32":
        return torch.qint32
    elif dtype == "qint64":
        return torch.qint64
    elif dtype == "float16":
        return torch.float16
    else:
        raise ValueError("Invalid dtype.")


def is_quantized(state_dict):
    """ Returns True if the model is quantized. 
    
    Args:
        state_dict: A dictionary containing the model's state.
        
    Returns:
        True if the model is quantized, False otherwise.
    """   
    for k, v in state_dict.items():
        if isinstance(v, tuple):
            for t in v:
                if isinstance(t, torch.Tensor):
                    if t.qscheme() == torch.per_tensor_affine:
                        return True
    return False


def infer_quantization_dtype(state_dict):
    """ Infer the quantization dtype of the model.
    
    Args:
        state_dict: A dictionary containing the model's state.
    
    Returns:
        The quantization dtype of the model.
    """
    for k, v in state_dict.items():
        if isinstance(v, tuple):
            for t in v:
                if isinstance(t, torch.Tensor):
                    if t.qscheme() == torch.per_tensor_affine:
                        return t.dtype