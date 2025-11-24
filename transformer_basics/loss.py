import torch
import ipdb
from einops import reduce
import torch.nn.functional as F

def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.
    """
    
    # Subtract the max for numerical stability
    max_logits = inputs.max(dim=-1, keepdim=True).values
    inputs = inputs - max_logits

    log_sum_exp = torch.logsumexp(inputs, dim=-1)

    # one hot encoding the target logits
    # ipdb.set_trace()
    one_hot = (torch.arange(inputs.shape[-1], device=targets.device)[None, :] == targets[:, None]).to(inputs.dtype)
    one_hots = reduce(inputs * one_hot, 'b v -> b', 'sum')
    loss = log_sum_exp - one_hots

    return loss.mean()

