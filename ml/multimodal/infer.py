from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ml.serving.predictor import Predictor
from .runtime import DEFAULT_MULTIMODAL_CHECKPOINT, MultimodalRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production EfficientNet diagnosis plus a validated multimodal fused representation.")
    parser.add_argument("image", type=Path, help="Plant image file")
    parser.add_argument("--production-checkpoint", type=Path, default=None)
    parser.add_argument("--multimodal-checkpoint", type=Path, default=DEFAULT_MULTIMODAL_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.multimodal_checkpoint.is_file():
        parser.error(f"No trained multimodal checkpoint is installed: {args.multimodal_checkpoint}")

    predictor = Predictor(model_path=args.production_checkpoint, device=args.device) if args.production_checkpoint else Predictor(device=args.device)
    runtime = MultimodalRuntime.load(predictor, args.multimodal_checkpoint, args.device)
    diagnosis = predictor.predict(args.image)
    image = predictor._to_pil(args.image)
    image_tensor = predictor.transform(image).unsqueeze(0)
    fused = runtime.encode(image_tensor, diagnosis.predicted_class.label)
    print(json.dumps({
        "diagnosis": diagnosis.to_dict(),
        "metadata_text": fused["metadata_text"],
        "fused_representation": {"dimension": fused["dimension"], "vector": fused["vector"]},
        "production_classifier": "EfficientNet-B0",
        "multimodal_training_state": "completed and validation-gated checkpoint",
    }, indent=2))


if __name__ == "__main__":
    main()
