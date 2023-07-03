import torch
import logging
import torchvision
from torch import nn
from torch.autograd import Function
from typing import Tuple
from google_drive_downloader import GoogleDriveDownloader as gdd

from cosplace_model.layers import Flatten, L2Norm, GeM

"""
Using this class we implement a Gradient Reversal Function for domain adaptation to promote domain-invariant 
feature learning. It reverses gradients during training by multiplying the negative gradient output with an 
'alpha' value. This technique enhances the model's ability to generalize across different domains, 
resulting in improved performance on the target domain.
"""
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class RevGrad(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
    
# Pretrained models on Google Landmarks v2 and Places 365
PRETRAINED_MODELS = {
    'resnet18_places'  : '1DnEQXhmPxtBUrRc81nAvT8z17bk-GBj5',
    'vgg16_places'     : '1UWl1uz6rZ6Nqmp1K5z3GHAIZJmDh4bDu'
}



# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
    "vit_b_16": 768,
    "vit_b_32": 768,
    "vit_l_16": 1024,
    "vit_l_32": 1024,
    "vit_h_14": 1280,
    "efficientnet_v2_s": 1280,
}


class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone: str, fc_output_dim: int, places: bool, alpha: float = None, domain_adapt: str = None):
        """Return a model for GeoLocalization.
        
        Args:
            backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
        """
        super().__init__()
        self.alpha = alpha
        self.domain_adapt = domain_adapt
        assert backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        self.backbone, features_dim = get_backbone(backbone, places)
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(features_dim, fc_output_dim),
            L2Norm()
        )
        try:
            self.domain_adapt_aggregation = nn.Sequential(
                RevGrad(alpha=self.alpha),
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),
                L2Norm()
            )
        except Exception as e:
            logging.info(f"Errore durante la gestione dell'adattamento del dominio in " + str(e))

    def forward(self, x, alpha=None):
        x = self.backbone(x)
        if self.domain_adapt is not None and self.alpha is not None:  # applichiamo l'adattamento del dominio
            # Dominio, applichiamo l'inversione del gradiente
            revgrad = RevGrad(self.alpha)
            x_rev = revgrad(x)
            domain_adaptation_out = self.domain_adapt_aggregation(x_rev)
            return domain_adaptation_out
        else:
            x = self.aggregation(x)  # etichette UTM
            return x
        
def get_pretrained_places_torchvision_model(backbone_name: str) -> torch.nn.Module:    
    if backbone_name.startswith("resnet18"):
        file_path = ("data/pretrained_on_places/resnet18_places.pth")
        model = torchvision.models.resnet18(num_classes=365)
        gdd.download_file_from_google_drive(file_id=PRETRAINED_MODELS["resnet18_places"], dest_path=file_path)
    elif backbone_name == "VGG16":
        file_path = ("data/pretrained_on_places/vgg16_places.pth")
        model = torchvision.models.vgg16(num_classes = 365)
        gdd.download_file_from_google_drive(file_id=PRETRAINED_MODELS["vgg16_places"])
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model  


def get_pretrained_torchvision_model(backbone_name: str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model


def get_backbone(backbone_name : str, places: bool) -> Tuple[torch.nn.Module, int]:
    if places:
        backbone = get_pretrained_places_torchvision_model(backbone_name)
    else:
        backbone = get_pretrained_torchvision_model(backbone_name)
    if backbone_name.startswith("ResNet"):
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
        
    elif backbone_name.startswith("efficientnet"):
        layers = list(backbone.children())[:-2] # Remove avg pooling and FC layer
        for layer in layers[:-2]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug(f"Train last layers of the efficientnet, freeze the previous ones")
    
    elif backbone_name == "VGG16":
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")
    
    #UPGRADE: new models below
    # ViT architectures models, vit_b_16 or VIT_H_14 or vit_b_16
    elif backbone_name.startswith("vit"):
        layers = list(backbone.children())[:-2]
        for layer in layers:
            for params in layer.parameters():
                params.requires_grad = False
        logging.debug(f"Train only last layer of the {backbone_name}, freezing the previous ones")

    backbone = torch.nn.Sequential(*layers)

    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]

    return backbone, features_dim
