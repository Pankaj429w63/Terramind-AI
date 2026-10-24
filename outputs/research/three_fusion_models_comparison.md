# TerraMind AI — Three-Model Fusion Comparison

- Status: **EVALUATED**
- Evaluated kinds: early, intermediate, late
- Not evaluated: none
- Same eligible samples across evaluated runs: True
- Fixed test example count: 3531

| model | test_examples | test_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | top3_accuracy | parameter_count | model_size_bytes | inference_latency_ms_per_image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Early fusion | 3531 | 0.5491 | 0.5544 | 0.4994 | 0.4969 | 0.5393 | 0.7661 | 5856213 | 38561995 | 21.3825 |
| Intermediate fusion | 3531 | 0.5678 | 0.5596 | 0.5177 | 0.5127 | 0.5570 | 0.7811 | 6382037 | 25858660 | 21.6925 |
| Late fusion | 3531 | 0.5641 | 0.5569 | 0.4981 | 0.4965 | 0.5450 | 0.7791 | 5377070 | 21837264 | 21.9703 |
| Image-only EfficientNet-B0 baseline | 3531 | 0.6058 | 0.5765 | 0.5918 | 0.5728 | 0.6019 | 0.8035 | 4121557 | 16783865 | 21.0563 |
| Text-only baseline | 3531 | 0.1946 | 0.1700 | 0.1515 | 0.1412 | 0.1725 | 0.3350 | 666969 | 2691209 | 0.1654 |

- Only metrics present in real test_metrics.json files are shown.
- The image-only and text-only rows are the baselines recorded by the fusion evaluation.
- Fusion must not be claimed to improve on the baselines unless these rows show it.
