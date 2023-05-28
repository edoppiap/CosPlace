
import torch
from typing import Tuple, Union
import torchvision.transforms as T

AUTOAUGMENT_POLICY = {
    "IMAGENET": T.AutoAugmentPolicy.IMAGENET,
    "CIFAR10": T.AutoAugmentPolicy.CIFAR10,
    "SVHN": T.AutoAugmentPolicy.SVHN
}


class DeviceAgnosticColorJitter(T.ColorJitter):
    def __init__(self, brightness: float = 0., contrast: float = 0., saturation: float = 0., hue: float = 0.):
        """This is the same as T.ColorJitter but it only accepts batches of images and works on GPU"""
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        color_jitter = super(DeviceAgnosticColorJitter, self).forward
        augmented_images = [color_jitter(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images


class DeviceAgnosticRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size: Union[int, Tuple[int, int]], scale: float):
        """This is the same as T.RandomResizedCrop but it only accepts batches of images and works on GPU"""
        super().__init__(size=size, scale=scale)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        random_resized_crop = super(DeviceAgnosticRandomResizedCrop, self).forward
        augmented_images = [random_resized_crop(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        return augmented_images
    
class DeviceAgnosticAutoAugment(T.AutoAugment):
    def __init__(self, policy_name: str, interpolation: T.InterpolationMode):
        assert policy_name in AUTOAUGMENT_POLICY, f"Policy must be one of {list(AUTOAUGMENT_POLICY.keys())}"
        policy = AUTOAUGMENT_POLICY.get(policy_name)
        """This is the same as T.AutoAugment but it only accepts batches of images and works on GPU"""
        super(DeviceAgnosticAutoAugment, self).__init__(policy=policy, interpolation=interpolation)

    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape

        # Convert torch.Tensor to PIL images
        pil_images = [T.ToPILImage()(img.cuda()) for img in images]

        # Apply AutoAugment transformations
        autoaugment_transform = T.AutoAugment(policy=self.policy, interpolation=self.interpolation).to(images.device)
        augmented_images = [autoaugment_transform(pil_img) for pil_img in pil_images] 

        # Convert PIL images back to torch.Tensor
        augmented_images = torch.stack([T.ToTensor()(img).to(images.device) for img in augmented_images])

        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images

if __name__ == "__main__":
    """
    You can run this script to visualize the transformations, and verify that
    the augmentations are applied individually on each image of the batch.
    """
    from PIL import Image
    # Import skimage in here, so it is not necessary to install it unless you run this script
    from skimage import data
    
    # Initialize DeviceAgnosticRandomResizedCrop
    random_crop = DeviceAgnosticRandomResizedCrop(size=[256, 256], scale=[0.5, 1])
    # Initialize DeviceAgnosticAutoAugment
    autoaugment = DeviceAgnosticAutoAugment(policy = T.AutoAugmentPolicy.IMAGENET, 
                                            interpolation = T.InterpolationMode.NEAREST)
    
    # Create a batch with 2 astronaut images
    pil_image = Image.fromarray(data.astronaut())
    tensor_image = T.functional.to_tensor(pil_image).unsqueeze(0)
    images_batch = torch.cat([tensor_image, tensor_image])
    # Apply augmentation (individually on each of the 2 images)
    augmented_batch_0 = random_crop(images_batch)
    augmented_batch_1 = autoaugment(images_batch)
    # Convert to PIL images
    augmented_image_0 = T.functional.to_pil_image(augmented_batch_0[0])
    augmented_image_1 = T.functional.to_pil_image(augmented_batch_0[1])
    augmented_image_2 = T.functional.to_pil_image(augmented_batch_1[0])
    augmented_image_3 = T.functional.to_pil_image(augmented_batch_1[1])
    # Visualize the original image, as well as the two augmented ones
    pil_image.show()
    augmented_image_0.show()
    augmented_image_1.show()
    augmented_image_2.show()
    augmented_image_3.show()
