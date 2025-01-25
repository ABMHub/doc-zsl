import torch
import torchvision
from vit_pytorch import cct

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
  def __init__(self, out_dim: int = 64, img_shape = (224, 224), model_version = 2, n_input_channels = 3):
    super(CCT, self).__init__()
    ccts = {
      2: cct.cct_2,
      4: cct.cct_4,
      6: cct.cct_6,
      7: cct.cct_7,
      8: cct.cct_8,
      14: cct.cct_14,
      16: cct.cct_16,
    }

    cct_obj = ccts[model_version]

    self.model = cct_obj(  # parametros padroes
      img_size=img_shape,
      num_classes = out_dim,
      n_input_channels=n_input_channels,
      n_conv_layers = 3,
      kernel_size = 7,
      stride = 2,
      padding = 3,
      pooling_kernel_size = 3,
      pooling_stride = 2,
      pooling_padding = 1,
      positional_embedding = 'learnable'
    )

  def forward(self, x):
    return self.model(x)
  
  @property
  def name(self):
    return "CCT_2"

class EfficientNet(torch.nn.Module):
  def __init__(self, out_dim: int = 64, model_version = 0):
    super(EfficientNet, self).__init__()
    ens = {
      0: (torchvision.models.efficientnet_b0, torchvision.models.EfficientNet_B0_Weights.DEFAULT),
      1: (torchvision.models.efficientnet_b1, torchvision.models.EfficientNet_B1_Weights.DEFAULT),
      2: (torchvision.models.efficientnet_b2, torchvision.models.EfficientNet_B2_Weights.DEFAULT),
      3: (torchvision.models.efficientnet_b3, torchvision.models.EfficientNet_B3_Weights.DEFAULT),
      4: (torchvision.models.efficientnet_b4, torchvision.models.EfficientNet_B0_Weights.DEFAULT),
      5: (torchvision.models.efficientnet_b5, torchvision.models.EfficientNet_B5_Weights.DEFAULT),
      6: (torchvision.models.efficientnet_b6, torchvision.models.EfficientNet_B6_Weights.DEFAULT),
      7: (torchvision.models.efficientnet_b7, torchvision.models.EfficientNet_B7_Weights.DEFAULT),
    }

    model, w = ens[model_version]

    self.model = model(weights=w)
    c = torch.nn.Sequential(
      torch.nn.Dropout(0.2, True),
      torch.nn.Linear(1280, out_dim)
      # torch.nn.Linear(1408, out_dim)
    )
    self.model.classifier = c

  def forward(self, x):
    return self.model(x)
  
  @property
  def name(self):
    return "efficient_net_b0"

class SiameseModel(torch.nn.Module):
  def __init__(self, model: torch.nn.Module):
    super(SiameseModel, self).__init__()
    self.model = model

  def forward(self, x1, x2):
    return self.model(x1), self.model(x2)
  
  @property
  def name(self):
    return self.model.name
