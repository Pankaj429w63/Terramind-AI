from __future__ import annotations

import unittest

import torch
from PIL import Image

from ml.serving.preprocessing import build_inference_transform
from mvpdr.train_efficientnet_b0 import build_eval_transform


class EfficientNetPreprocessingTests(unittest.TestCase):
    def test_serving_transform_matches_training_for_non_square_image(self):
        image = Image.new("RGB", (420, 210))
        pixels = image.load()
        for y in range(image.height):
            for x in range(image.width):
                pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
        trained = build_eval_transform(224)(image)
        served = build_inference_transform("EfficientNet-B0", 224)(image)
        self.assertEqual(tuple(trained.shape), (3, 224, 224))
        self.assertTrue(torch.equal(trained, served))


if __name__ == "__main__":
    unittest.main()
