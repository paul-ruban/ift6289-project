import os


def compute_model_size(model_path):
    """ Size of a file in megabytes """
    return os.path.getsize(model_path) / (1024 * 1024)


def compute_num_parameters(model):
    """ Number of trainable parameters in a model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)