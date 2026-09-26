"""Upload the unchanged EfficientNet checkpoint to the private model-artifacts bucket."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from backend.supabase_client import get_supabase
from ml.models.model_loader import DEFAULT_MODEL_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, default=Path(os.getenv("TERRAMIND_MODEL_PATH", DEFAULT_MODEL_PATH)))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint file does not exist: {args.checkpoint}")

    bucket = os.getenv("TERRAMIND_MODEL_BUCKET", "model-artifacts")
    object_path = os.getenv("TERRAMIND_MODEL_OBJECT", "efficientnet-b0/best_model.pth")
    client = get_supabase()
    if client is None:
        parser.error("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the backend environment first.")

    content = args.checkpoint.read_bytes()
    checksum = hashlib.sha256(content).hexdigest()
    client.storage.from_(bucket).upload(
        object_path,
        content,
        {"content-type": "application/octet-stream", "upsert": "true"},
    )
    print(f"Uploaded {len(content)} bytes to private bucket {bucket}/{object_path}")
    print(f"Set TERRAMIND_MODEL_SHA256={checksum} in Render's server environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
