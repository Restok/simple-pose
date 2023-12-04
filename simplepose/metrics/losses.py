import torch
def mpjpe(predicted, target):
    """
    Mean Per Joint Position Error
    :param predicted: (batch, 21, 3) predicted joint positions, ndarray
    :param target: (batch, 21, 3) target joint positions, ndarray
    """
    #First align the poses
    predicted = predicted - predicted[:, 0:1, :]
    target = target - target[:, 0:1, :]
    error = torch.sqrt(torch.sum((predicted - target) ** 2, dim=-1))
    error = torch.mean(error, dim=-1).mean()
    return error