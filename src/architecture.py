import torch
import torchvision
from vit_pytorch import cct

class Vit(torch.nn.Module):  # modelo poderoso e grande, aprende com muitos dados
  def __init__(self, out_dim: int = 64, model_version = "b", pretrained: bool = True):
    super(Vit, self).__init__()
    ens = {
      "b": (torchvision.models.vit_b_32, torchvision.models.ViT_B_16_Weights.DEFAULT),
      "l": (torchvision.models.vit_l_32, torchvision.models.ViT_L_16_Weights.DEFAULT),
    }

    self.model_version = model_version

    model, w = ens[model_version]

    if not pretrained:
      w = None

    self.model = model(weights=w)

    self.model.heads.head = torch.nn.Linear(in_features=self.model.hidden_dim, out_features=out_dim)

  def forward(self, x):
    return self.model(x)
  
  @property
  def name(self):
    return f"VIT_{self.model_version}_32"

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
  def __init__(self, out_dim: int = 64, model_version = 0, pretrained = True):
    super(EfficientNet, self).__init__()
    ens = {
      0: (torchvision.models.efficientnet_b0, torchvision.models.EfficientNet_B0_Weights.DEFAULT),
      1: (torchvision.models.efficientnet_b1, torchvision.models.EfficientNet_B1_Weights.DEFAULT),
      2: (torchvision.models.efficientnet_b2, torchvision.models.EfficientNet_B2_Weights.DEFAULT),
      3: (torchvision.models.efficientnet_b3, torchvision.models.EfficientNet_B3_Weights.DEFAULT),
      4: (torchvision.models.efficientnet_b4, torchvision.models.EfficientNet_B4_Weights.DEFAULT),
      5: (torchvision.models.efficientnet_b5, torchvision.models.EfficientNet_B5_Weights.DEFAULT),
      6: (torchvision.models.efficientnet_b6, torchvision.models.EfficientNet_B6_Weights.DEFAULT),
      7: (torchvision.models.efficientnet_b7, torchvision.models.EfficientNet_B7_Weights.DEFAULT),
    }

    self.model_version = model_version

    model, w = ens[model_version]

    if not pretrained:
      w = None

    self.model = model(weights=w)
    in_features = self.model.classifier[1].in_features

    lin = torch.nn.Linear(in_features, out_dim)

    self.model.classifier[1] = lin

  def forward(self, x):
    return self.model(x)
  
  @property
  def name(self):
    return f"efficient_net_b{self.model_version}"

class ResNet(torch.nn.Module):
  def __init__(self, out_dim: int = 64, model_version = 18, pretrained = True):
    super(ResNet, self).__init__()
    ens = {
      18: (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.DEFAULT),
      34: (torchvision.models.resnet34, torchvision.models.ResNet34_Weights.DEFAULT),
      50: (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.DEFAULT),
      101: (torchvision.models.resnet101, torchvision.models.ResNet101_Weights.DEFAULT),
      152: (torchvision.models.resnet152, torchvision.models.ResNet152_Weights.DEFAULT),
    }

    self.model_version = model_version

    model, w = ens[model_version]

    if not pretrained:
      w = None

    self.model = model(weights=w)
    in_features = self.model.fc.in_features

    lin = torch.nn.Linear(in_features, out_dim)

    self.model.fc = lin

  def forward(self, x):
    return self.model(x)
  
  @property
  def name(self):
    return f"resnet{self.model_version}"

class SiameseModel(torch.nn.Module):
  def __init__(self, model: torch.nn.Module):
    super(SiameseModel, self).__init__()
    self.model = model

  def forward(self, x1, x2):
    return self.model(x1), self.model(x2)
  
  @property
  def name(self):
    return self.model.name
