# Next Session

## Start Here

Read:

1. PROJECT_MEMORY.md
2. SESSION_LOG.md
3. outputs/annotation/pilot/ANNOTATION_QC_CHANGE_REPORT.md
4. outputs/annotation/pilot/PROPOSED_CORRECTIONS.md
5. outputs/annotation/pilot/QUALITY_REVIEW_REPORT.md

## Current Phase

Phase 8 — Multimodal AI (Pilot Annotation Quality Control & Review Gate)

## Exact Next Task

1. Collect the fresh human re-annotation for `garlic rust/google_0003.jpg` (`obs-ed667e8194fd`).
   - Staging workbook: `outputs/annotation/pilot/annotator_workbook_reannotation.csv`
   - Command: `python -m scripts.review_annotations --workbook outputs/annotation/pilot/annotator_workbook_reannotation.csv --alias-map outputs/annotation/pilot/sealed/alias_map.json --port 8045`
   - Web interface: `http://127.0.0.1:8045`
2. Confirm the 8 curated protocol/typographical edits from `ANNOTATION_QC_CHANGE_REPORT.md`.
3. Perform independent review: An independent reviewer (distinct from the row's observer ID) marks `approved` or `rejected` with their reviewer name and date in `outputs/annotation/pilot/review_ledger.csv`.
4. Validate annotations and emit the paired dataset:
   `python -m scripts.validate_annotations --workbook <final_workbook.csv> --alias-map outputs/annotation/pilot/sealed/alias_map.json --review-ledger outputs/annotation/pilot/review_ledger.csv --text-origin user_observation --instrument-version "pilot-v1" --provenance "Blinded human pilot annotation via review_tool; instrument pilot-v1" --source-reference "outputs/annotation/pilot/annotator_workbook.csv" --emit-observations "data/annotation/paired_observations.csv"`
5. Run the fusion data audit:
   `python -m scripts.audit_fusion_data --paired-csv data/annotation/paired_observations.csv`

## Do Not Do

- Do not copy AI-generated descriptions into the human observation dataset.
- Do not overwrite `outputs/annotation/pilot/annotator_workbook.csv` or `annotator_workbook_backup_178.csv`.
- Do not mark any review ledger entry approved without genuine independent human review.
- Do not modify production image-only checkpoints or existing experimental checkpoints.
- Do not run fusion training until the independent review ledger is approved and the data audit passes.

## Definition of Done

The human re-annotation is recorded, independent review is completed and approved in the review ledger, the paired dataset is emitted, and the fusion data audit passes.