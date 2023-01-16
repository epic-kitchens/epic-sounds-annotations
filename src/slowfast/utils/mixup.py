import torch
import numpy as np

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    # Get random permutation for batch
    batch_size = x[0].size(0)
    index = torch.randperm(batch_size).cuda()

    # Mixup data
    mixed_x = [lam * x[0] + (1 - lam) * x[0][index, :]]

    # Create label/mixup label pairs
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, weights=None):
    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    if weights is not None:
        loss_a = (loss_a * weights).sum(1).mean()
        loss_b = (loss_b * weights).sum(1).mean()
    return lam * loss_a + (1 - lam) * loss_b

    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target_a.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = lam * pred.eq(target_a.view(1, -1).expand_as(pred)) \
        + (1 - lam) * pred.eq(target_b.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).to(torch.float32).sum(0)
        res.append(float(correct_k.mul_(100.0 / batch_size)))
    return tuple(res)