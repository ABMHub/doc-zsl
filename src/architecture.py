import torch
import torchvision
from vit_pytorch.cct import cct_2

class Vit(torch.nn.Module):  # modelo poderoso e grande, aprende com muitos dados
  def __init__(self, out_dim: int = 64):
    super(Vit, self).__init__()
    self.model = torchvision.models.vit_b_16(torchvision.models.ViT_B_16_Weights)
    self.linear = torch.nn.Linear(1000, out_dim)

  def forward(self, x):
    out_imagenet = self.model(x)
    out = self.linear(out_imagenet)

    return out 
  
  @property
  def name(self):
    return "VIT_B_16"

class CCT(torch.nn.Module):  # modelo economico, aprende com menos dados
  def __init__(self, out_dim: int = 64, img_shape = (224, 224)):
    super(CCT, self).__init__()
    self.model = cct_2(  # parametros padroes
      img_size=img_shape,
      n_conv_layers = 3,
      kernel_size = 7,
      stride = 2,
      padding = 3,
      pooling_kernel_size = 3,
      pooling_stride = 2,
      pooling_padding = 1,
      num_classes = out_dim,
      positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
      n_input_channels=1,
    )

  def forward(self, x):
    return self.model(x)
  
  @property
  def name(self):
    return "CCT_2"

class SiameseModel(torch.nn.Module):
  def __init__(self, model: torch.nn.Module):
    super(SiameseModel, self).__init__()
    self.model = model

  def forward(self, x1, x2):
    return self.model(x1), self.model(x2)
  
  @property
  def name(self):
    return self.model.name
