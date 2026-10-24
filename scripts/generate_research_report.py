"""Generate academic deliverables from persisted model/RAG evaluation artifacts only."""
from __future__ import annotations

import csv
import json
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "research"
BASE = ROOT / "outputs" / "experiments" / "efficientnet_b0_error_analysis"
CANDIDATE = ROOT / "outputs" / "experiments" / "efficientnet_b0_weighted_loss_v1_error_analysis"
COMPARISON = ROOT / "outputs" / "experiments" / "efficientnet_b0_comparison" / "comparison.json"
SERVED = ROOT / "models" / "archive" / "2026-09-09_efficientnet_b0_official_v3"
CANDIDATE_RUN = ROOT / "outputs" / "experiments" / "efficientnet_b0_weighted_loss_v1"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value, digits=2):
    return "Unavailable" if value is None else f"{value * 100:.{digits}f}%"


def metric_row(name, status, split, subset, n, metrics, checkpoint, training_seconds=None, training_basis=None):
    return {
        "model": name, "status": status, "split": split, "evaluation_subset": subset,
        "n": n, "accuracy": metrics.get("accuracy"),
        "macro_precision": metrics.get("macro_precision"), "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"), "weighted_f1": metrics.get("weighted_f1"),
        "top3_accuracy": metrics.get("top3_accuracy"), "nll": metrics.get("nll"),
        "ece_15_bins": metrics.get("ece_15_bins"), "checkpoint": checkpoint,
        "training_duration_seconds": training_seconds, "training_duration_basis": training_basis,
        "parameter_count": None, "checkpoint_size_bytes": None, "inference_mean_ms_per_image": None,
    }


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    OUT.mkdir(parents=True, exist_ok=True)
    comparison = load(COMPARISON)
    baseline_analysis = load(BASE / "metrics.json")
    candidate_analysis = load(CANDIDATE / "metrics.json")
    stored = load(SERVED / "results.json")
    baseline_config = load(SERVED / "config.json")
    candidate_result = load(CANDIDATE_RUN / "results.json")
    rag = load(OUT / "rag_retrieval_evaluation.json")
    verification_path = OUT / "verification_results.json"
    verification = load(verification_path) if verification_path.is_file() else {
        "status": "not-recorded", "note": "Run final verification before submission."
    }

    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    stored_metrics = {
        "accuracy": stored.get("test_acc"), "macro_f1": stored.get("test_macro_f1"),
        "n": None,
    }
    full_rows_metrics = baseline_analysis["test_raw_official_all_rows"]
    unique_metrics = baseline_analysis["test_raw_duplicate_free"]
    cand_metrics = candidate_analysis["test_raw_duplicate_free"]
    rows = [
        metric_row("Production EfficientNet-B0 (stored result artifact)", "stored; denominator/procedure not recorded in results.json",
                   "official test (stored artifact)", "stored run; sample count unavailable", None, stored_metrics,
                   "models/archive/2026-09-09_efficientnet_b0_official_v3/best_model.pth"),
        metric_row("Production EfficientNet-B0 (independent all-row evaluation)", "measured; independent evaluation",
                   "official test", "all official rows; duplicates retained", full_rows_metrics["n"], full_rows_metrics,
                   baseline_analysis["checkpoint"]),
        metric_row("Production EfficientNet-B0 (matched comparison baseline)", "measured; not promoted candidate",
                   "official test", baseline["evaluation_subset"], baseline["n"], baseline,
                   baseline_analysis["checkpoint"]),
        metric_row("One-epoch class-weighted EfficientNet-B0 candidate", "experimental; not promoted",
                   "official test", candidate["evaluation_subset"], candidate["n"], candidate,
                   str(CANDIDATE_RUN.relative_to(ROOT) / "best_model.pth"), candidate["training_and_validation_wall_seconds"],
                   candidate.get("training_duration_basis")),
    ]
    for row, analysis in ((rows[1], baseline_analysis), (rows[2], baseline_analysis),
                          (rows[3], candidate_analysis)):
        row["parameter_count"] = analysis.get("parameter_count")
        row["checkpoint_size_bytes"] = analysis.get("checkpoint_size_bytes")
        row["inference_mean_ms_per_image"] = analysis.get("inference_latency_ms_per_image", {}).get("mean")
    for fusion_name in ("Early fusion", "Intermediate fusion", "Late fusion"):
        rows.append(metric_row(fusion_name, "implemented; untrained; unavailable for inference",
                               "official test", "not evaluated: valid paired text unavailable", None, {}, "Unavailable"))

    payload = {
        "title": "TerraMind AI final measured comparison",
        "generated_from_artifacts": [
            str(COMPARISON.relative_to(ROOT)), str((BASE / "metrics.json").relative_to(ROOT)),
            str((CANDIDATE / "metrics.json").relative_to(ROOT)), str((SERVED / "results.json").relative_to(ROOT)),
            str((OUT / "rag_retrieval_evaluation.json").relative_to(ROOT)),
        ],
        "official_split": comparison["protocol"],
        "matched_comparison": {"baseline": baseline, "candidate": candidate,
                               "candidate_promoted": comparison.get("candidate_promoted", False)},
        "stored_production_result": stored_metrics,
        "independent_full_official_test": full_rows_metrics,
        "fusion_models": [{"model": row["model"], "status": row["status"], "metrics": None} for row in rows[-3:]],
        "rows": rows,
        "rag_evaluation": rag,
        "verification": verification,
        "limitations": [
            "The stored production results.json reports 60.0218% accuracy and 56.6969% macro-F1, but does not record its evaluation sample count or enough protocol detail to reproduce those values.",
            "Independent evaluation of the same named production checkpoint on all 3,677 official test rows reports different metrics; the duplicate-free matched comparison uses 3,545 images. These records are preserved separately and are not averaged.",
            "Fusion classifiers are implemented but untrained: no independent, per-image text paired with PlantWild images is present. Class labels and class prompts are target-derived and prohibited as evaluation/inference inputs.",
            "The retrieval evaluation has ten manually labeled queries; its low precision@5 is a material limitation.",
            "Local checks do not verify external hosting, Supabase reachability, or a live LLM provider.",
        ],
    }
    (OUT / "model_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / "model_comparison.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Preserve useful per-class and confusion artifacts alongside the report.
    for source, target in (
        (BASE / "per_class_metrics.csv", OUT / "per_class_metrics_baseline_duplicate_free.csv"),
        (CANDIDATE / "per_class_metrics.csv", OUT / "per_class_metrics_candidate_duplicate_free.csv"),
        (BASE / "confusion_matrix.csv", OUT / "confusion_matrix_baseline_duplicate_free.csv"),
        (CANDIDATE / "confusion_matrix.csv", OUT / "confusion_matrix_candidate_duplicate_free.csv"),
        (BASE / "confusion_matrix.png", OUT / "confusion_matrix_baseline_duplicate_free.png"),
        (CANDIDATE / "confusion_matrix.png", OUT / "confusion_matrix_candidate_duplicate_free.png"),
    ):
        if source.is_file():
            shutil.copyfile(source, target)

    # Class error analysis is derived from measured per-class CSV rows.
    with (BASE / "per_class_metrics.csv").open(newline="", encoding="utf-8") as source:
        per_class = list(csv.DictReader(source))
    weakest = sorted(per_class, key=lambda item: float(item["f1"]))[:10]
    confused = baseline_analysis.get("top_confusion_pairs", [])[:10]
    rag_metrics = rag["metrics"]
    old_full = baseline_analysis["test_raw_official_all_rows"]
    delta_acc = (candidate["accuracy"] - baseline["accuracy"]) * 100
    delta_f1 = (candidate["macro_f1"] - baseline["macro_f1"]) * 100
    train_split = comparison["protocol"]["official_split"]
    eligible = comparison["protocol"]["eligible"]
    cats = rag["index"]["by_category"]

    report = f'''# TerraMind AI: Multimodal Fusion and Agricultural Decision Support

**Final experimental status:** the three fusion architectures are implemented, but no valid paired text dataset or trained fusion checkpoints are available. Therefore, this report does not claim a three-way fusion result. The measured classifier comparison is between the image-only EfficientNet-B0 and one separate, one-epoch class-weighted candidate.

## Abstract

TerraMind AI is a decision-support prototype that combines a PlantWild image classifier with retrieval-augmented agricultural guidance. This report consolidates persisted measurements for the production EfficientNet-B0, a class-weighted experimental candidate, and a ten-query source-retrieval evaluation. The matched duplicate-free official test comparison measured {pct(baseline['accuracy'])} accuracy and {pct(baseline['macro_f1'])} macro-F1 for the baseline, versus {pct(candidate['accuracy'])} and {pct(candidate['macro_f1'])} for the candidate. The candidate remains experimental. Early, intermediate, and late fusion implementations are untrained because PlantWild class labels and class-level prompts are not independent per-image text and would leak the target. RAG retrieval returned 251 points sourced from 218 documents; on ten manually labeled questions, precision@5 was {rag_metrics['precision_at_5']:.2f}. Local integration tests passed, but hosted services were not verified. Outputs are guidance and model predictions, not confirmed diagnoses.

## 1. Problem statement and objectives

Plant disease recognition in uncontrolled imagery is difficult because of visual variation within a disease class and similarity between classes. The project objective is to provide image-based candidate predictions and cite relevant agricultural knowledge while preserving uncertainty. The research objective is to compare an image-only baseline with three fusion mechanisms without test leakage or fabricated paired observations. The fusion comparison remains blocked by the data requirement described below.

## 2. System architecture

The Next.js interface calls a FastAPI service. The backend lazily loads the production EfficientNet-B0 for image inference, exposes explicit model availability, and uses a source-backed retrieval pipeline with TF-IDF embeddings and a local persistent fallback (Qdrant is optional). A supervisor routes selected requests to existing specialist workflows and records which agents ran. Optional LLM generation is server-configured; when no provider is configured, the backend returns an evidence-template response with citations. Supabase handles user identity and durable records when reachable; local report/review files are a fallback, not cross-device persistence. The frontend displays model scores separately from test-set metrics and exposes uncertainty/review paths.

## 3. Dataset and split protocol

The repository’s PlantWild v1 material describes 18,542 images across 89 classes and supplies official split metadata. The project artifacts record {train_split['train']:,} training, {train_split['validation']:,} validation, and {train_split['test']:,} test rows. Content-duplicate filtering excluded {eligible['excluded_validation_duplicates']} validation and {eligible['excluded_test_duplicates']} test entries and yielded {eligible['validation']:,} validation and {eligible['test']:,} test eligible images for the duplicate-free comparison. Model selection uses validation macro-F1; the comparison artifact records that test data was not used for selection.

## 4. Image-only EfficientNet-B0 baseline

The unchanged production checkpoint is `models/archive/2026-09-09_efficientnet_b0_official_v3/best_model.pth`, with 89 outputs and 224-pixel input. Its companion training artifact records seed {baseline_config['seed']}, {baseline_config['epochs']} epochs, and best validation macro-F1 {pct(stored['best_val_macro_f1'], 4)} at epoch {stored['best_epoch']}. This is the serving checkpoint; its stored result reports accuracy {pct(stored['test_acc'], 4)} and macro-F1 {pct(stored['test_macro_f1'], 4)}.

### Reconciliation of production metrics

The stored result artifact does not record its test sample count or enough evaluation-protocol detail to reproduce the score. Independent evaluation of the named checkpoint on all {old_full['n']:,} official test rows measured {pct(old_full['accuracy'])} accuracy and {pct(old_full['macro_f1'])} macro-F1. A separate duplicate-free evaluation on {baseline['n']:,} rows measured {pct(baseline['accuracy'])} and {pct(baseline['macro_f1'])}. These values differ because the stored score and independent evaluation are not demonstrably the same evaluation record/protocol. We preserve all three and use only the duplicate-free baseline for the matched candidate comparison; we do not overwrite or average the records.

## 5. Early, intermediate, and late fusion methods

The repository implements reusable classifiers and training/evaluation interfaces:

- **Early fusion:** concatenate projected image and text features before the classification head.
- **Intermediate fusion:** combine projected encodings through a learned gate.
- **Late fusion:** combine image and text class probabilities, with modality weights selected using validation data.

All three are **implemented but untrained and unavailable for inference**. They have no test accuracy, F1, confusion matrix, parameter/inference measurements, or checkpoint to report. Their blank metrics in the generated CSV/JSON mean unavailable, not zero.

## 6. Text modality and leakage controls

PlantWild provides class names and GPT-generated descriptions per class, not independent farmer observations or captions paired to each image. The existing multimodal preparation that uses class labels is class-conditioned; using it as model text would reveal the target. The fusion pipeline requires provenance-bearing per-image text, rejects label/target columns and class-name content, checks paths and split alignment, and fits the vocabulary on training text only. These checks cannot authenticate a curator’s provenance assertion. Valid paired data must be collected or supplied and also be available at inference before fusion training can proceed.

## 7. Experimental setup and image-only candidate

The candidate is an EfficientNet-B0 initialized compatibly with the production model and trained for one epoch using class-weighted loss on the official training split. It was selected on validation macro-F1 ({pct(comparison['candidate_validation_macro_f1'], 4)} versus baseline {pct(comparison['baseline_validation_macro_f1'], 4)} on the duplicate-free validation subset). The comparison artifact records a {candidate['training_and_validation_wall_seconds']:.1f}-second filesystem interval from config creation to results write; this is not a trainer-only timer. It was evaluated on the same {candidate['n']:,} duplicate-free official test examples as the baseline. Test metrics were not used to select it. The experimental checkpoint remains separate and is not promoted.

## 8. Measured model comparison

| Model/checkpoint | Evaluation subset | N | Accuracy | Macro-P/R/F1 | Weighted-F1 | Top-3 | Parameters | Size (MB) | CPU ms/image |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Production EfficientNet-B0 v3 | Duplicate-free official test | {baseline['n']:,} | {pct(baseline['accuracy'])} | {pct(baseline['macro_precision'])}/{pct(baseline['macro_recall'])}/{pct(baseline['macro_f1'])} | {pct(baseline['weighted_f1'])} | {pct(baseline['top3_accuracy'])} | {baseline_analysis['parameter_count']:,} | {baseline_analysis['checkpoint_size_bytes'] / 1_000_000:.2f} | {baseline_analysis['inference_latency_ms_per_image']['mean']:.2f} |
| One-epoch class-weighted candidate (not promoted) | Same duplicate-free official test | {candidate['n']:,} | {pct(candidate['accuracy'])} | {pct(candidate['macro_precision'])}/{pct(candidate['macro_recall'])}/{pct(candidate['macro_f1'])} | {pct(candidate['weighted_f1'])} | {pct(candidate['top3_accuracy'])} | {candidate_analysis['parameter_count']:,} | {candidate_analysis['checkpoint_size_bytes'] / 1_000_000:.2f} | {candidate_analysis['inference_latency_ms_per_image']['mean']:.2f} |
| Early fusion | Not evaluated | — | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable |
| Intermediate fusion | Not evaluated | — | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable |
| Late fusion | Not evaluated | — | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable |

On this finite duplicate-free subset, the candidate’s measured delta was {delta_acc:+.3f} percentage points accuracy and {delta_f1:+.3f} points macro-F1. This is a single candidate run, not evidence of robust improvement or external-field performance, and promotion was not automatic. The candidate’s weighted-F1 was {pct(candidate['weighted_f1'])}, compared with baseline {pct(baseline['weighted_f1'])}; top-3 was {pct(candidate['top3_accuracy'])} versus {pct(baseline['top3_accuracy'])}. The comparison CSV/JSON identifies the checkpoint paths, subset, n, and metric sources.

![Measured image-only comparison; fusion metrics are unavailable](model_comparison.png)

## 9. Per-class error analysis

The baseline error analysis is calculated from actual duplicate-free test predictions. The ten lowest-F1 classes are:

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
{chr(10).join(f"| {row['class_name']} | {int(row['support'])} | {float(row['precision']):.3f} | {float(row['recall']):.3f} | {float(row['f1']):.3f} |" for row in weakest)}

Most frequent measured confusions include:

| Actual | Predicted | Count |
|---|---|---:|
{chr(10).join(f"| {row['actual']} | {row['predicted']} | {row['count']} |" for row in confused)}

The split’s class-count artifacts are included under the baseline and candidate error-analysis directories. The evaluation script identified no dedicated healthy-class IDs in the current mapping, so a separate healthy-versus-diseased score is unavailable. Per-class CSVs and 89×89 confusion matrices are included alongside this report. Temperature scaling was fitted on {baseline_analysis['calibration']['validation_samples']:,} duplicate-free validation examples; the baseline calibrated duplicate-free test ECE was {pct(baseline_analysis['calibration']['test']['ece_15_bins'])}, compared with raw ECE {pct(baseline['ece_15_bins'])}. Calibration changes confidence, not predicted model accuracy.

## 10. RAG architecture and retrieval evaluation

The pipeline preprocesses the question, retrieves and ranks chunks, then returns source IDs/titles and citations with either an optional configured provider or a transparent evidence template. Diagnosis class context is retrieval context only and low-confidence output remains explicitly unconfirmed. The current measured index reports **{rag['index']['by_source_count']} source documents and {rag['index']['total_points']} vector points** using `{rag['index']['backend']}`; category point counts are `{json.dumps(cats, ensure_ascii=False)}`. A source can produce more than one chunk.

On the existing manually labeled ten-query evaluation (two queries per category), measured hit rate/recall@5 was **{rag_metrics['hit_rate_at_5']:.2f}**, precision@5 **{rag_metrics['precision_at_5']:.2f}**, MRR **{rag_metrics['mean_reciprocal_rank']:.3f}**, and citation-presence **{rag_metrics['citation_presence_rate']:.2f}**. Precision is weak: only 18% of the top-five retrieved passages were judged relevant under this small label set. This is a smoke evaluation, not a broad benchmark. Full cases and labels are retained in `rag_retrieval_evaluation.json`/`.csv`.

## 11. Supervisor and specialist agents

The supervisor selects existing specialists for request context. The diagnosis workflow records Vision, Diagnosis, Research/RAG, Treatment, Fertilizer, Care, and Report execution events when applicable; chat can route an agricultural question to a relevant specialist. API tests inspected actual returned event records, and unit tests covered category routing, citations, no-evidence responses, provider failure fallback, and local vector-store persistence. Agent guidance is not a model diagnosis, and citations are retained from retrieval evidence.

## 12. Application integration

FastAPI serves single/batch image inference, model registry/status, analytics artifacts, RAG/chat, agent workflows, and ownership-checked history/review routes. The Next.js production build completed locally and reads backend responses through configured rewrites. The candidate is explicitly unpromoted; all three fusion choices remain unavailable. The service uses confidence calibration when a checkpoint-matched validation sidecar is available and preserves raw scores for display.

## 13. Verification and deployment

Verification record: `{verification.get('status', 'not recorded')}`. Local tests/builds are not hosted-service checks. The local Windows API process measured approximately {verification.get('backend_startup_rss_mib', 'Unavailable')} MiB RSS at startup after the configured single-thread limit. Render Free advertises 512 MB memory and an ephemeral filesystem; this local process leaves little headroom, and no Render/Linux fit or live endpoint was verified. Supabase connectivity and a live LLM provider were not tested. The local API reported LLM unconfigured and uses evidence-template generation.

## 14. Limitations and ethical considerations

PlantWild evaluation does not establish performance on a particular farm, cultivar, camera, or region. The class prediction is a candidate label; it is not a confirmed agricultural diagnosis. Users should inspect source-backed guidance and seek qualified local expertise, particularly for uncertain images or pesticide decisions. The ten-query retrieval set is too small to establish category-wide retrieval quality. Fusion comparisons await genuinely paired text. The one-epoch candidate has limited training and must not replace the production checkpoint without a documented, protocol-matched evaluation and review.

## 15. Conclusion

This release delivers an operational image-only decision-support path and a measured, cited RAG prototype. The best matched test result in the current saved comparison is the separate candidate at {pct(candidate['accuracy'])} accuracy and {pct(candidate['macro_f1'])} macro-F1, but it remains experimental. The three fusion models are architectures only: their results are unavailable until non-label-derived text paired to each image is supplied. The stored production result discrepancy is retained and documented rather than hidden. External deployment, persistent hosted storage, and provider behavior remain unverified.

## References and repository evidence

1. Wei, T., Chen, Z., Huang, Z., & Yu, X. (2024). *Benchmarking In-the-wild Multimodal Disease Recognition and A Versatile Baseline*. arXiv:2408.03120. https://arxiv.org/abs/2408.03120
2. Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML, PMLR 97, 6105–6114. https://proceedings.mlr.press/v97/tan19a.html
3. Singh, D. et al. (2020). *PlantDoc: A Dataset for Visual Plant Disease Detection*. ACM India Joint International Conference on Data Science and Management of Data. https://arxiv.org/abs/1911.10317. PlantDoc is contextual literature, not paired text used in these experiments.
4. Dataset documentation and official split files: `data/plantwild/plantwild/readme.txt`, `train_split.txt`, `val_split.txt`, `test_split.txt`, `classes.txt`.
5. Source-backed RAG examples (the original JSON records, with provenance retained): `data/knowledge_base/diseases/apple/apple_black_rot.diseases.json`; `data/knowledge_base/treatment/tomato/tomato_late_blight.treatment.json`; `data/knowledge_base/fertilizer/tomato/tomato.fertilizer.json`; `data/knowledge_base/plant_care/cucumber/cucumber.plant_care.json`; `data/knowledge_base/agriculture/corn/corn.agriculture.json`.
6. Measured artifacts: `outputs/experiments/efficientnet_b0_comparison/comparison.json`; `outputs/experiments/efficientnet_b0_error_analysis/metrics.json`; `outputs/experiments/efficientnet_b0_weighted_loss_v1_error_analysis/metrics.json`; `outputs/research/rag_retrieval_evaluation.json`.
'''
    (OUT / "final_report.md").write_text(report, encoding="utf-8")

    fig, axes = plt.subplots(figsize=(10, 5.2))
    names = ["Production baseline", "Weighted-loss candidate"]
    series = [(baseline["accuracy"], candidate["accuracy"]),
              (baseline["macro_f1"], candidate["macro_f1"]),
              (baseline["weighted_f1"], candidate["weighted_f1"]),
              (baseline["top3_accuracy"], candidate["top3_accuracy"])]
    labels = ["Accuracy", "Macro-F1", "Weighted-F1", "Top-3 accuracy"]
    width = 0.18
    for index, (label, values) in enumerate(zip(labels, series)):
        xs = [i + (index - 1.5) * width for i in range(2)]
        bars = axes.bar(xs, [value * 100 for value in values], width, label=label)
        axes.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    axes.set_xticks(range(2), names)
    axes.set_ylim(0, 100)
    axes.set_ylabel("Score (%)")
    axes.set_title("Duplicate-free official PlantWild test subset (n=3,545 per model)")
    axes.legend(ncols=2)
    axes.grid(axis="y", alpha=0.2)
    fig.text(0.5, 0.015, "Early, intermediate, and late fusion are implemented but untrained; no test metrics available.",
             ha="center", fontsize=9, color="#8a4e16")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    chart_path = OUT / "model_comparison.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)

    with PdfPages(OUT / "final_report.pdf") as pdf:
        lines = []
        for line in report.splitlines():
            lines.extend(textwrap.wrap(line, width=106, break_long_words=True, break_on_hyphens=False) or [""])
        for start in range(0, len(lines), 50):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.075, 0.965, "TerraMind AI — Final Research Report", fontsize=13, weight="bold", va="top")
            fig.text(0.075, 0.925, "\n".join(lines[start:start + 50]), fontsize=7.4, va="top", family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.imshow(plt.imread(chart_path))
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        for matrix_path in (OUT / "confusion_matrix_baseline_duplicate_free.png",
                            OUT / "confusion_matrix_candidate_duplicate_free.png"):
            if matrix_path.is_file():
                fig, ax = plt.subplots(figsize=(9, 8))
                ax.imshow(plt.imread(matrix_path))
                ax.axis("off")
                ax.set_title(matrix_path.stem.replace("_", " "))
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    print(json.dumps({"markdown": str(OUT / "final_report.md"), "pdf": str(OUT / "final_report.pdf"),
                      "csv": str(OUT / "model_comparison.csv"), "json": str(OUT / "model_comparison.json"),
                      "chart": str(chart_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
