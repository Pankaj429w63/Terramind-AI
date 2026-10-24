# TerraMind AI completion plan

## Repository findings

- Production inference is already served by `ml/serving/predictor.py` and `backend/app.py` using the unchanged EfficientNet-B0 v3 checkpoint. Its persisted official-split results report 89 classes, 60.0218% test accuracy, 56.6969% test macro-F1, and best validation macro-F1 56.5659% at epoch 20 (seed 42; 13,045/1,820/3,677 train/validation/test images).
- PlantWild has official image splits and class prompts, but no paired farmer observations/captions. `plantwild_prompts.json` is GPT-generated per class; the existing multimodal experiment pairs exact class labels with images. Both are class-conditioned and cannot support a leakage-free disease classifier at inference. The current cross-attention autoencoder is representation learning, not one of the three evaluated classifiers, and its README reports no training run.
- The knowledge base has 218 source JSON documents (58 diseases, 58 treatments, 34 fertilizer, 34 plant care, 34 agriculture). The persisted local TF-IDF/Qdrant-compatible fallback currently reports 251 points from 218 sources and a 4,096-dimensional vocabulary. Retrieval currently returns ranked cited passages; no response generator is needed to preserve evidence grounding.
- Existing FastAPI, frontend, agents, authentication, user-owned persistence, and low-confidence review flow are present. Keep them intact unless an evidence-backed defect is found.

## Execution plan

1. Preserve the production checkpoint and existing knowledge documents; record repository status.
2. Explicitly block multimodal disease-classification training/evaluation until valid paired, inference-available text exists. Do not treat class labels/prompts as valid text.
3. Evaluate current RAG retrieval on a small manually labeled set spanning all five categories, report index persistence/counts, and retain citations.
4. Generate structured result files and an editable academic report from stored baseline metrics and measured retrieval outputs; mark unavailable per-class metrics and fusion results clearly.
5. Run focused repository tests/build checks only where available and report blockers accurately.

## Current experiment gates

- Early, intermediate, and late fusion disease-classification results: blocked by invalid/absent paired text, not implemented as trained predictive models.
- Image-only baseline: existing verified results are available; the dataset and checkpoint exist. No new candidate training is justified before the requested weak-class analysis can be produced from saved per-class predictions.
- RAG retrieval: runnable against the persisted local index without external services.
