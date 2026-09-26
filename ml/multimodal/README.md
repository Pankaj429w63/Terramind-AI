# TerraMind multimodal representation module

This experimental module is separate from the production EfficientNet-B0 classifier. It uses the current EfficientNet-B0 convolutional feature weights as its image encoder initialization, and uses exact PlantWild class labels from `data/plantwild/plantwild/classes.txt` as the text modality. PlantWild split rows provide the existing image-to-class-ID pairing. The text pipeline tokenizes those labels; it does not generate or assume natural-language captions.

## Implemented architecture

- EfficientNet-B0 image feature map (1280 channels) and a causal Transformer class-metadata encoder map to the same latent dimension `Z`.
- Bidirectional Transformer decoders use cross-attention: image queries attend to text keys/values, and text queries attend to image keys/values.
- The cross-attended image stream reconstructs normalized image pixels. The cross-attended text stream predicts the next class-metadata tokens.
- A symmetric multi-positive contrastive alignment objective treats repeated examples of the same class as positives rather than false negatives.
- Pooled cross-attended streams are projected to a fused latent vector. A future trained checkpoint can expose this vector alongside the unchanged production diagnosis.

The repository contains the PlantWild `train_split.txt` and `val_split.txt` metadata and their corresponding image files (13,045 train entries and 1,820 validation entries at implementation time). Preparation validates each split path and class ID and uses only the mapped class label. This is image/class-metadata supervision, not a dataset of paired natural-language descriptions.

## Training and inference

Training is explicit and does not run during backend startup. It reads the existing production checkpoint without changing it and writes a new artifact only to `models/multimodal/terramind_multimodal.pth` (or an explicitly selected new output path). Existing output paths are never overwritten. Training logs reconstruction/alignment losses only; it does not claim disease classification accuracy.

```powershell
.\.venv\Scripts\python.exe -m ml.multimodal.train
```

After a validated checkpoint is created, standalone inference uses the existing production classifier and emits its prediction with the fused vector:

```powershell
.\.venv\Scripts\python.exe -m ml.multimodal.infer path/to/plant.jpg
```

The backend checks `TERRAMIND_MULTIMODAL_CHECKPOINT`, defaulting to the path above. `/api/multimodal/status` reports the module and checkpoint state. `/api/multimodal/analyze` is unavailable until a checkpoint has completed training and validation; when available, it returns the unchanged EfficientNet diagnosis and a fused representation based on the predicted class label. The classifier remains responsible for all production disease predictions.

## Training status

Architecture, deterministic data preparation, loss functions, training code, runtime loading, API status/analysis routes, and focused tests are implemented. **Training has not been run. No multimodal checkpoint, accuracy, or experimental result is claimed.**
