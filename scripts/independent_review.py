"""Launch the dedicated independent plant-observation review server and UI.

Examples (Windows PowerShell / CMD):
    python -m scripts.independent_review --check
    python -m scripts.independent_review --port 8050
    python -m scripts.independent_review --workbook outputs/annotation/pilot/annotator_workbook_reannotation.csv --alias-map outputs/annotation/pilot/sealed/alias_map.json --review-ledger outputs/annotation/pilot/review_ledger.csv

The tool displays the actual plant image alongside its human observation,
enforces anti-self-review rules, records genuine reviewer decisions
(approved, rejected, disputed) with real calendar dates, and atomically updates
the review ledger.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.annotation.independent_review_tool import main  # noqa: E402

if __name__ == "__main__":
    main()
