# GeoWarp
# GEO WARP -> Code from https://github.com/gmberton/geo_warp
##### GEOWARP network
from torch import nn
import torch
from cosplace_model.cosplace_network import get_backbone
from cosplace_model.layers import L2Norm, GeM, Flatten, feature_L2_norm
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, backbone, fc_output_dim):
        super().__init__()
        self.backbone, features_dim = get_backbone(backbone)
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(features_dim, fc_output_dim),
            L2Norm()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((15, 15))
        self.l2norm = L2Norm()

    def forward(self, x, f_type="local"):
        x = self.backbone(x)
        if f_type == "local":
            x = self.avgpool(x)
            x = self.l2norm(x)
            return x
        elif f_type == "global":
            x = self.aggregation(x)
            return x
        else:
            raise ValueError(f"Invalid features type: {f_type}")


##### MODULE FOR GEOWARP
class HomographyRegression(nn.Module):
    def __init__(self, output_dim=16, kernel_sizes=[7, 5], channels=[225, 128, 64], padding=0):
        super().__init__()
        assert len(kernel_sizes) == len(channels) - 1, \
            f"In HomographyRegression the number of kernel_sizes must be less than channels, but you said {kernel_sizes} and {channels}"
        nn_modules = []
        for in_channels, out_channels, kernel_size in zip(channels[:-1], channels[1:], kernel_sizes):
            nn_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            nn_modules.append(nn.BatchNorm2d(out_channels))
            nn_modules.append(nn.ReLU())
        self.conv = nn.Sequential(*nn_modules)
        # Find out output size of last conv, aka the input of the fully connected
        shape = self.conv(torch.ones([2, 225, 15, 15])).shape
        output_dim_last_conv = shape[1] * shape[2] * shape[3]
        self.linear = nn.Linear(output_dim_last_conv, output_dim)
        # Initialize the weights/bias with identity transformation
        init_points = torch.tensor([-1, -1, 1, -1, 1, 1, -1, 1]).type(torch.float)
        init_points = torch.cat((init_points, init_points))
        self.linear.bias.data = init_points
        self.linear.weight.data = torch.zeros_like((self.linear.weight.data))

    def forward(self, x):
        B = x.shape[0]
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)
        x = x.reshape(B, 8, 2)
        return x.reshape(B, 8, 2)


##### MODULE GEOWARP
class GeoWarp(nn.Module):
    """
    Overview of the network:
    name                 input                                       output
    FeaturesExtractor:   (2B x 3 x H x W)                            (2B x 256 x 15 x 15)
    compute_similarity:  (B x 256 x 15 x 15), (B x 256 x 15 x 15)    (B x 225 x 15 x 15)
    HomographyRegression:(B x 225 x 15 x 15)                         (B x 16)
    """

    def __init__(self, features_extractor, homography_regression):
        super().__init__()
        self.features_extractor = features_extractor
        self.homography_regression = homography_regression

    def forward(self, operation, args):
        """Compute a forward pass, which can be of different types.
        This "ugly" step of passing the operation as a string has been adapted
        to allow calling different methods through the Network.forward().
        This is because only Network.forward() works on multiple GPUs when using torch.nn.DataParallel().

        Parameters
        ----------
        operation : str, defines the type of forward pass.
        args : contains the tensor(s) on which to apply the operation.

        """
        assert operation in ["features_extractor", "similarity", "regression", "similarity_and_regression"]
        if operation == "features_extractor":  # Encoder
            if len(args) == 2:
                tensor_images, features_type = args
                return self.features_extractor(tensor_images, features_type)
            else:
                tensor_images = args
                return self.features_extractor(tensor_images, "local")

        elif operation == "similarity":
            tensor_img_1, tensor_img_2 = args
            return self.similarity(tensor_img_1, tensor_img_2)

        elif operation == "regression":
            similarity_matrix = args
            return self.regression(similarity_matrix)

        elif operation == "similarity_and_regression":
            tensor_img_1, tensor_img_2 = args
            similarity_matrix_1to2, similarity_matrix_2to1 = self.similarity(tensor_img_1, tensor_img_2)
            return self.regression(similarity_matrix_1to2), self.regression(similarity_matrix_2to1)

    def similarity(self, tensor_img_1, tensor_img_2):
        features_1 = self.features_extractor(tensor_img_1.cuda())
        features_2 = self.features_extractor(tensor_img_2.cuda())
        similarity_matrix_1to2 = compute_similarity(features_1, features_2)
        similarity_matrix_2to1 = compute_similarity(features_2, features_1)
        return similarity_matrix_1to2, similarity_matrix_2to1

    def regression(self, similarity_matrix):
        return self.homography_regression(similarity_matrix)


def compute_similarity(features_a, features_b):
    b, c, h, w = features_a.shape
    features_a = features_a.transpose(2, 3).contiguous().view(b, c, h * w)
    features_b = features_b.view(b, c, h * w).transpose(1, 2)
    features_mul = torch.bmm(features_b, features_a)
    correlation_tensor = features_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
    correlation_tensor = feature_L2_norm(F.relu(correlation_tensor))
    return correlation_tensor