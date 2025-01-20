import torch

# def ContrastiveLoss(x1, x2, label, margin: float = 1.0):
#     """
#     Computes Contrastive Loss
#     """

#     dist = torch.nn.functional.pairwise_distance(x1, x2)

#     loss = (1 - label) * torch.pow(dist, 2) \
#         + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
#     loss = torch.mean(loss)

#     return loss

def EuclidianDistance(x1, x2):
    return (x1-x2).pow(2).sum(1).sqrt()

def ContrastiveLoss(x1, x2, target, margin: float = 1.0):
    eps = 1e-8
    distances = EuclidianDistance(x1, x2)
    losses = target.float() * distances.pow(2) + (1 + -1 * target).float() * torch.nn.functional.relu(margin - (distances + eps)).pow(2)
    return losses.mean()