"""Launch the local, blinded plant-image review tool.

Examples (Windows CMD):
    python -m scripts.review_annotations --check
    python -m scripts.review_annotations
    python -m scripts.review_annotations --port 9000

Run ``--help`` for all options. The tool only reads the sealed alias map and the
official dataset and only writes observations into the annotator workbook.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.annotation.review_tool import main  # noqa: E402

if __name__ == "__main__":
    main()
