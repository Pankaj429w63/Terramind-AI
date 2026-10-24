# TerraMind multimodal representation module

This experimental representation module is separate from the production EfficientNet-B0 classifier. It uses exact PlantWild class labels from `data/plantwild/plantwild/classes.txt` as its text modality, paired by split metadata. This is class-conditioned target information, not a valid leakage-free text input for disease classification. The dataset's GPT-generated prompts are also associated with each class and are not valid per-image observations. Do not use this module or those prompts to claim multimodal disease-classification performance. Valid paired text must be available at inference before such an experiment can proceed.

## Implemented architecture

- EfficientNet-B0 image feature map (1280 channels) and a causal Transformer class-metadata encoder map to the same latent dimension `Z`.
- Bidirectional Transformer decoders use cross-attention: image queries attend to text keys/values, and text queries attend to image keys/values.
- The cross-attended image stream reconstructs normalized image pixels. The cross-attended text stream predicts the next class-metadata tokens.
- A symmetric multi-positive contrastive alignment objective treats repeated examples of the same class as positives rather than false negatives.
- Pooled cross-attended streams are projected to a fused latent vector. A future trained checkpoint can expose this vector alongside the unchanged production diagnosis.

The repository contains PlantWild `train_split.txt` and `val_split.txt` metadata and their corresponding image files (13,045 train entries and 1,820 validation entries at implementation time). Preparation validates each split path and class ID and uses only the mapped class label. This is image/class-metadata representation learning, not a dataset of paired natural-language descriptions or a supervised fusion classifier.

## Training and inference

Training is explicit and does not run during backend startup. It reads the existing production checkpoint without changing it and writes a new artifact only to `models/multimodal/terramind_multimodal.pth` (or an explicitly selected new output path). Existing output paths are never overwritten. Training logs reconstruction/alignment losses only; it does not claim disease classification accuracy.

```powershell
.\.venv\Scripts\python.exe -m ml.multimodal.train
```

After a valid paired-text checkpoint is created, standalone inference requires image-specific observed text and emits it with the fused vector:

```powershell
.\.venv\Scripts\python.exe -m ml.multimodal.infer path/to/plant.jpg --paired-text "observed leaf symptoms"
```

The backend checks `TERRAMIND_MULTIMODAL_CHECKPOINT`, defaulting to the path above. `/api/multimodal/status` reports the module/checkpoint state and explicitly marks the early, intermediate, and late classification fusion methods unavailable. `/api/multimodal/analyze` remains unavailable until a checkpoint has completed training and validation. Any future disease-classification integration must accept valid user-available text and must not feed the target class or a class-derived description at inference.

## Training status

Architecture, deterministic data preparation, loss functions, training code, runtime loading, API status/analysis routes, and focused tests exist. **Training has not been run. No multimodal checkpoint, accuracy, or experimental result is claimed. The current metadata text is not eligible for disease-classifier evaluation due to target leakage.**
