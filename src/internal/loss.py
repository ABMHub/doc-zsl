import torch

def EuclidianDistance(x1, x2):
    return (x1-x2).pow(2).sum(1).sqrt()

def CosineDistance(x1, x2):
    return 1 - torch.nn.functional.cosine_similarity(x1, x2, dim=1)

class ContrastiveLoss:
    def __init__(self, margin: float = 1.0, cosine_distance: bool = False):
        self.margin = margin
        self.distance_f = CosineDistance if cosine_distance else EuclidianDistance
    
    def __call__(self, x1, x2, target):
        eps = 1e-8

        distances = self.distance_f(x1, x2)
        pos = target.float() * distances.pow(2)
        neg = (1 + -1 * target).float() * torch.nn.functional.relu(self.margin - (distances + eps)).pow(2)
        losses = pos + neg
        return losses.mean()