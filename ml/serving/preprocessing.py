from __future__ import annotations

from typing import Sequence

from PIL import Image
import torchvision.transforms as transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EFFICIENTNET_B0_EVAL_RESIZE = 256
EFFICIENTNET_B0_CROP_SIZE = 224


def build_preprocessing(
    image_size: int = 160,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Build the validation/inference transform used by the trained MobileNet model."""
    padded_size = int(round(image_size * 1.15))
    return transforms.Compose(
        [
            transforms.Resize((padded_size, padded_size)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_efficientnet_b0_preprocessing(
    image_size: int = EFFICIENTNET_B0_CROP_SIZE,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Exact EfficientNet-B0 v3 eval recipe: 256x256 resize, 224px center crop, ImageNet norm."""
    if image_size != EFFICIENTNET_B0_CROP_SIZE:
        raise ValueError(f"EfficientNet-B0 v3 inference requires {EFFICIENTNET_B0_CROP_SIZE}px crops, got {image_size}")
    return transforms.Compose(
        [
            transforms.Resize((EFFICIENTNET_B0_EVAL_RESIZE, EFFICIENTNET_B0_EVAL_RESIZE)),
            transforms.CenterCrop((EFFICIENTNET_B0_CROP_SIZE, EFFICIENTNET_B0_CROP_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_inference_transform(
    backbone: str,
    image_size: int,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    if backbone == "EfficientNet-B0":
        return build_efficientnet_b0_preprocessing(image_size, mean, std)
    return build_preprocessing(image_size, mean, std)


def prepare_image(image: Image.Image, transform: transforms.Compose):
    """Convert an image to RGB and apply the serving transform."""
    return transform(image.convert("RGB"))
