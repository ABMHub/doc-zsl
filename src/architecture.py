import torch
import torch.optim.sgd
import torchvision
from vit_pytorch import cct

class SiameseModelUnit(torch.nn.Module):
  def __init__(self):
    super(SiameseModelUnit, self).__init__()
    self.im_shape = 224
    self.learning_rate = 1e-2
    self.momentum = 0.9
    self.weight_decay = 1e-4
    self.scheduler = torch.optim.lr_scheduler.StepLR
    self.lr_gamma = 0.1

  @property
  def name(self):
    return "model"

  def forward(self, x):
    return self.model(x)
  
  @property
  def optimizer(self):
    return torch.optim.SGD(
      self.model.parameters(), 
      self.learning_rate, 
      self.momentum, 
      weight_decay=self.weight_decay
    )

class AlexNet(SiameseModelUnit):
  def __init__(self, out_dim: int = 64, pretrained: bool = True, **kwargs):
    super(AlexNet, self).__init__()
    w = None if not pretrained else torchvision.models.AlexNet_Weights.DEFAULT
    self.model = torchvision.models.alexnet(w)
    self.model.classifier[-1] = torch.nn.Linear(4096, out_dim)
    self.learning_rate = 1e-3

  @property
  def name(self):
    return f"AlexNet"

class DenseNet(SiameseModelUnit):
  def __init__(self, out_dim: int = 64, model_version = 121, pretrained: bool = True):
    super(DenseNet, self).__init__()
    ens = {
      121: (torchvision.models.densenet121, torchvision.models.DenseNet121_Weights.DEFAULT),
      161: (torchvision.models.densenet161, torchvision.models.DenseNet161_Weights.DEFAULT),
      169: (torchvision.models.densenet169, torchvision.models.DenseNet169_Weights.DEFAULT),
      201: (torchvision.models.densenet201, torchvision.models.DenseNet201_Weights.DEFAULT),
    }

    model, w = ens[model_version]
    self.model = model(weights = w if pretrained else None)
    ln: torch.nn.Linear = self.model.classifier
    self.model.classifier = torch.nn.Linear(ln.in_features, out_dim)

class VGG(SiameseModelUnit):
  def __init__(self, out_dim: int = 64, model_version = 11, pretrained: bool = True):
    super(VGG, self).__init__()
    ens = {
      11: (torchvision.models.vgg11, torchvision.models.VGG11_Weights.DEFAULT),
      13: (torchvision.models.vgg13, torchvision.models.VGG13_Weights.DEFAULT),
      16: (torchvision.models.vgg16, torchvision.models.VGG16_Weights.DEFAULT),
      19: (torchvision.models.vgg19, torchvision.models.VGG19_Weights.DEFAULT),
    }

    model, w = ens[model_version]
    self.model = model(weights = w if pretrained else None)
    ln: torch.nn.Linear = self.model.classifier[-1]
    self.model.classifier[-1] = torch.nn.Linear(ln.in_features, out_dim)

class Vit(SiameseModelUnit):  # modelo poderoso e grande, aprende com muitos dados
  def __init__(self, out_dim: int = 64, model_version = "b", pretrained: bool = True):
    super(Vit, self).__init__()
    ens = {
      "b": (torchvision.models.vit_b_32, torchvision.models.ViT_B_32_Weights.DEFAULT),
      "l": (torchvision.models.vit_l_32, torchvision.models.ViT_L_32_Weights.DEFAULT),
    }

    self.model_version = model_version

    model, w = ens[model_version]

    if not pretrained:
      w = None

    self.model = model(weights=w)

    self.model.heads.head = torch.nn.Linear(in_features=self.model.hidden_dim, out_features=out_dim)
    self.learning_rate = 1e-3
    self.weight_decay = 1e-5
    self.lr_gamma = 0
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

  @property
  def optimizer(self):
    return torch.optim.AdamW(
      self.model.parameters(), 
      self.learning_rate, 
      weight_decay=self.weight_decay
    )

  @property
  def name(self):
    return f"VIT_{self.model_version}_32"

class CCT(SiameseModelUnit):  # modelo economico, aprende com menos dados
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

  @property
  def name(self):
    return "CCT_2"

class EfficientNet(SiameseModelUnit):
  def __init__(self, out_dim: int = 64, model_version = 0, pretrained = True):
    super(EfficientNet, self).__init__()
    ens = {
      0: (torchvision.models.efficientnet_b0, torchvision.models.EfficientNet_B0_Weights.DEFAULT, 224),
      1: (torchvision.models.efficientnet_b1, torchvision.models.EfficientNet_B1_Weights.DEFAULT, 240),
      2: (torchvision.models.efficientnet_b2, torchvision.models.EfficientNet_B2_Weights.DEFAULT, 288),
      3: (torchvision.models.efficientnet_b3, torchvision.models.EfficientNet_B3_Weights.DEFAULT, 300),
      4: (torchvision.models.efficientnet_b4, torchvision.models.EfficientNet_B4_Weights.DEFAULT, 380),
      5: (torchvision.models.efficientnet_b5, torchvision.models.EfficientNet_B5_Weights.DEFAULT, 456),
      6: (torchvision.models.efficientnet_b6, torchvision.models.EfficientNet_B6_Weights.DEFAULT, 528),
      7: (torchvision.models.efficientnet_b7, torchvision.models.EfficientNet_B7_Weights.DEFAULT, 600),
    }

    self.model_version = model_version

    model, w, self.im_shape = ens[model_version]

    if not pretrained:
      w = None

    self.model = model(weights=w)
    in_features = self.model.classifier[1].in_features

    lin = torch.nn.Linear(in_features, out_dim)

    self.model.classifier[1] = lin

  @property
  def name(self):
    return f"efficient_net_b{self.model_version}"

class EfficientNetV2(SiameseModelUnit):
  def __init__(self, out_dim: int = 64, model_version = 's', pretrained = True):
    super(EfficientNetV2, self).__init__()
    ens = {
      's': (torchvision.models.efficientnet_v2_s, torchvision.models.EfficientNet_V2_S_Weights.DEFAULT, 384),
      'm': (torchvision.models.efficientnet_v2_m, torchvision.models.EfficientNet_V2_M_Weights.DEFAULT, 480),
      'l': (torchvision.models.efficientnet_v2_l, torchvision.models.EfficientNet_V2_L_Weights.DEFAULT, 480),
    }

    self.model_version = model_version
    model, w, self.im_shape = ens[model_version]

    if not pretrained:
      w = None

    self.model = model(weights=w)
    in_features = self.model.classifier[1].in_features

    lin = torch.nn.Linear(in_features, out_dim)
    self.model.classifier[1] = lin

    self.weight_decay = 0.00002
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    self.lr_gamma = 0

  @property
  def name(self):
    return f"efficient_net_b{self.model_version}"

class MobileNetV3(SiameseModelUnit):
  def __init__(self, out_dim: int = 64, model_version = 0, pretrained = True):
    super(MobileNetV3, self).__init__()
    ens = {
      'small': (torchvision.models.mobilenet_v3_small, torchvision.models.MobileNet_V3_Small_Weights.DEFAULT, 224),
      'large': (torchvision.models.mobilenet_v3_large, torchvision.models.MobileNet_V3_Large_Weights.DEFAULT, 224),
    }

    self.model_version = model_version
    model, w, self.im_shape = ens[model_version]

    self.model = model(weights = w if pretrained else None)
    ln: torch.nn.Linear = self.model.classifier[-1]
    self.model.classifier[-1] = torch.nn.Linear(ln.in_features, out_dim)

    self.learning_rate = 0.01
    self.weight_decay = 1e-5
    self.lr_gamma = 0.973

  @property
  def optimizer(self):
    return torch.optim.RMSprop(
      self.model.parameters(), 
      self.learning_rate, 
      weight_decay=self.weight_decay
    )

  @property
  def name(self):
    return f"efficient_net_b{self.model_version}"
  
class ResNet(SiameseModelUnit):
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

    self.model.fc = torch.nn.Linear(in_features, out_dim)

  @property
  def name(self):
    return f"resnet{self.model_version}"

class ConvNext(SiameseModelUnit):
  def __init__(self, out_dim: int = 64, model_version = 18, pretrained = True):
    super(ConvNext, self).__init__()
    ens = {
      "tiny": (torchvision.models.convnext_tiny, torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT),
      "small": (torchvision.models.convnext_small, torchvision.models.ConvNeXt_Small_Weights.DEFAULT),
      "base": (torchvision.models.convnext_base, torchvision.models.ConvNeXt_Base_Weights.DEFAULT),
      "large": (torchvision.models.convnext_large, torchvision.models.ConvNeXt_Large_Weights.DEFAULT),
    }

    model, w = ens[model_version]
    self.model = model(weights = w if pretrained else None)
    ln: torch.nn.Linear = self.model.classifier[-1]
    self.model.classifier[-1] = torch.nn.Linear(ln.in_features, out_dim)

    self.learning_rate = 1e-3
    self.weight_decay = 1e-5
    self.lr_gamma = 0
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

  @property
  def optimizer(self):
    return torch.optim.AdamW(
      self.model.parameters(), 
      self.learning_rate, 
      weight_decay=self.weight_decay
    )

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
