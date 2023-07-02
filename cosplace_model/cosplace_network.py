import torch
import logging
import torchvision
import os
from torch import nn
from torch.autograd import Function
from typing import Tuple
from transformers import ViTModel
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


# Urls for pretrained models backbones on places 365 and Google Landmarks v2
MODEL_URLS = {
    'resnet18_places': '1DnEQXhmPxtBUrRc81nAvT8z17bk-GBj5',
    'resnet50_places': '1zsY4mN4jJ-AsmV3h4hjbT72CBfJsgSGC',
    'resnet101_places': '1E1ibXQcg7qkmmmyYgmwMTh7Xf1cDNQXa',
    'vgg16_places': '1UWl1uz6rZ6Nqmp1K5z3GHAIZJmDh4bDu',
    'resnet18_gldv2': '1wkUeUXFXuPHuEvGTXVpuP5BMB-JJ1xke',
    'resnet50_gldv2': '1UDUv6mszlXNC1lv6McLdeBNMq9-kaA70',
    'resnet101_gldv2': '1apiRxMJpDlV0XmKlC5Na_Drg2jtGL-uE',
    'vgg16_gldv2': '10Ov9JdO7gbyz6mB5x0v_VSAUMj91Ta4o'
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
    'resnet18_places': 365,
    'resnet50_places': 365,
    'resnet101_places': 365,
    'vgg16_places': 365,
    'resnet18_gldv2': 512,
    'resnet50_gldv2': 512,
    'vgg16_gldv2': 512
}


class GeoLocalizationNet(nn.Module):
    def __init__(self, args):
        """Return a model for GeoLocalization.
        
        Args:
            backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
        """
        super().__init__()
        self.alpha = args.alpha
        assert args.backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        if self.pretrain:
            self.backbone = get_backbone_pretrained(args.backbone)
            if self.fc_output_dim is not None:
                # Concatenate fully connected layer to the aggregation layer
                self.aggregation = nn.Sequential(self.aggregation,
                                                 nn.Linear(self.features_dim, self.fc_output_dim),
                                                 L2Norm())
                features_dim = self.fc_output_dim
        else:
            self.backbone, features_dim = get_backbone(args.backbone)
            self.aggregation = nn.Sequential(
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(features_dim, args.fc_output_dim),
                L2Norm()
            )
        try:
            self.domain_adapt_aggregation = nn.Sequential(
                RevGrad(alpha=self.alpha),
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(features_dim, args.fc_output_dim),
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


def get_pretrained_torchvision_model(backbone_name: str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]),
                                 f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model

# from deep-visual geolocalization benchmark
def get_pretrained_model(args):
    if args.pretrain == 'places':
        num_classes = 365
    elif args.pretrain == 'gldv2':
        num_classes = 512

    if args.backbone.startswith("resnet18"):
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif args.backbone.startswith("resnet50"):
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif args.backbone.startswith("resnet101"):
        model = torchvision.models.resnet101(num_classes=num_classes)
    elif args.backbone.startswith("vgg16"):
        model = torchvision.models.vgg16(num_classes=num_classes)

    if args.backbone.startswith('resnet'):
        model_name = args.backbone.split('conv')[0] + "_" + args.pretrain
    else:
        model_name = args.backbone + "_" + args.pretrain
    file_path = os.path.join("data", "pretrained_nets", model_name + ".pth")

    if not os.path.exists(file_path):
        gdd.download_file_from_google_drive(file_id=MODEL_URLS[model_name],
                                            dest_path=file_path)
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model


def get_backbone_pretrained(args):
    # The aggregation layer works differently based on the type of architecture
    args.work_with_tokens = args.backbone.startswith('cct') or args.backbone.startswith('vit')
    if args.backbone.startswith("resnet"):
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        elif args.backbone.startswith("resnet18"):
            backbone = torchvision.models.resnet18(pretrained=True)
        elif args.backbone.startswith("resnet50"):
            backbone = torchvision.models.resnet50(pretrained=True)
        elif args.backbone.startswith("resnet101"):
            backbone = torchvision.models.resnet101(pretrained=True)
        for name, child in backbone.named_children():
            # Freeze layers before conv_3
            if name == "layer3":
                break
            for params in child.parameters():
                params.requires_grad = False
        if args.backbone.endswith("conv4"):
            logging.debug(
                f"Train only conv4_x of the resnet{args.backbone.split('conv')[0]} (remove conv5_x), freeze the previous ones")
            layers = list(backbone.children())[:-3]
        elif args.backbone.endswith("conv5"):
            logging.debug(
                f"Train only conv4_x and conv5_x of the resnet{args.backbone.split('conv')[0]}, freeze the previous ones")
            layers = list(backbone.children())[:-2]
    elif args.backbone == "vgg16":
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        else:
            backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the vgg16, freeze the previous ones")
    elif args.backbone == "alexnet":
        backbone = torchvision.models.alexnet(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the alexnet, freeze the previous ones")
    elif args.backbone.startswith("vit"):
        assert args.resize[0] in [224, 384], f'Image size for ViT must be either 224 or 384, but it\'s {args.resize[0]}'
        if args.resize[0] == 224:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        elif args.resize[0] == 384:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-384')

        if args.trunc_te:
            logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te + 1}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.encoder.layer.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        backbone = VitWrapper(backbone, args.aggregation)

        args.features_dim = 768
        return backbone

    backbone = torch.nn.Sequential(*layers)
    args.features_dim = get_output_channels_dim(backbone)  # Dinamically obtain number of channels in output
    return backbone


class VitWrapper(nn.Module):
    def __init__(self, vit_model, aggregation):
        super().__init__()
        self.vit_model = vit_model
        self.aggregation = aggregation

    def forward(self, x):
        if self.aggregation in ["netvlad", "gem"]:
            return self.vit_model(x).last_hidden_state[:, 1:, :]
        else:
            return self.vit_model(x).last_hidden_state[:, 0, :]


def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]


def get_backbone(backbone_name: str) -> Tuple[torch.nn.Module, int]:
    backbone = get_pretrained_torchvision_model(backbone_name)
    if backbone_name.startswith("ResNet"):
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer

    elif backbone_name == "VGG16":
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")

    # UPGRADE: new models below
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
