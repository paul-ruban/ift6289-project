import math
import torch
import torch.nn.functional as F


def get_distillation(distillation_method):
    assert distillation_method in ["soft_target", "factor_transfer", "correlation_congruence"]

    if distillation_method == "soft_target":
        return SoftTarget()
    elif distillation_method == "factor_transfer":
        return FactorTransfer()
    elif distillation_method == "correlation_congruence":
        return CorrelationCongruence()


class SoftTarget(torch.nn.Module):
	'''
	Sort Target Distilling with KLDiv Loss
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def forward(self, input, target):
		""" Compute the loss for the SoftTarget distillation method.

		Args:
			input (torch.Tensor): The input to the model.
			target (torch.Tensor): The target to be used for the loss.

		Returns:
			torch.Tensor: The loss for the SoftTarget distillation method.
		"""
		loss = F.kl_div(F.log_softmax(input, dim=1), F.softmax(target, dim=1), reduction='batchmean')
		return loss


class FactorTransfer(torch.nn.Module):
	'''
	Paraphrasing Complex Network: Network Compression via Factor Transfer
	http://papers.nips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer.pdf
	'''
	def forward(self, input, target):
		""" Compute the loss for the SoftTarget distillation method.

		Args:
			input (torch.Tensor): The input to the model.
			target (torch.Tensor): The target to be used for the loss.

		Returns:
			torch.Tensor: The loss for the SoftTarget distillation method.
		"""
		loss = F.l1_loss(self.normalize(input), self.normalize(target))
		loss = loss.sum() / input.size(0)
		return loss

	def normalize(self, input):
		""" Normalize the input.
		
		Args:
			input (torch.Tensor): The input to be normalized.
		
		Returns:
			torch.Tensor: The normalized input.
		"""
		input_norm = F.normalize(input.view(input.size(0),-1))
		return input_norm


class CorrelationCongruence(torch.nn.Module):
	'''
	Correlation Congruence from: 
    http://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf
	'''
	def __init__(self, gamma=0.4, p_order=2):
		""" Initialize the CorrelationCongruence object.

		Args:
			gamma (float): The gamma value for the correlation.
			p_order (int): The order of the polynomial.
		"""
		super().__init__()
		self.gamma = gamma
		self.p_order = p_order

	def forward(self, input, target):
		""" Compute the loss for the SoftTarget distillation method.

		Args:
			input (torch.Tensor): The input to the model.
			target (torch.Tensor): The target to be used for the loss.

		Returns:
			torch.Tensor: The loss for the SoftTarget distillation method.
		"""
		corr_matrix_input = self.correlation_matrix(input)
		corr_matrix_target = self.correlation_matrix(target)

		loss = F.mse_loss(corr_matrix_input, corr_matrix_target)
		loss = loss.sum() / input.size(0)
		return loss

	def correlation_matrix(self, x):
		""" Compute the correlation matrix.

		Args:
			x (torch.Tensor): The input to compute the correlation matrix.
		
		Returns:
			torch.Tensor: The correlation matrix.
		"""
		x = F.normalize(x, p=2, dim=-1)
		similarity  = torch.matmul(x, x.t())
		corr_matrix = torch.zeros_like(similarity)

		for p in range(self.p_order + 1):
			corr_matrix += math.exp(-2 * self.gamma) * (2 * self.gamma)**p / \
						math.factorial(p) * torch.pow(similarity, p)

		return corr_matrix