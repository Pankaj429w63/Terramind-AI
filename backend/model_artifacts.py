from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from ml.models.model_loader import DEFAULT_MODEL_PATH


def resolve_model_checkpoint() -> Path:
    """Use the local checkpoint, or restore it from private Supabase Storage.

    Render Free has an ephemeral filesystem, so a configured private artifact
    object is downloaded and checksum-verified at process startup when needed.
    The checkpoint bytes are never changed.
    """
    checkpoint = Path(os.getenv("TERRAMIND_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if checkpoint.is_file():
        return checkpoint

    bucket = os.getenv("TERRAMIND_MODEL_BUCKET", "").strip()
    object_path = os.getenv("TERRAMIND_MODEL_OBJECT", "").strip()
    expected_sha = os.getenv("TERRAMIND_MODEL_SHA256", "").strip().lower()
    if not bucket or not object_path:
        return checkpoint
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("TERRAMIND_MODEL_SHA256 must contain the uploaded checkpoint's 64-character SHA-256.")

    from backend.supabase_client import get_supabase

    client = get_supabase()
    if client is None:
        raise RuntimeError("Cannot restore the model checkpoint: Supabase server credentials are unavailable.")
    content = client.storage.from_(bucket).download(object_path)
    if not isinstance(content, bytes) or not content:
        raise ValueError("Supabase Storage returned an empty or invalid model checkpoint.")
    if len(content) > 512 * 1024 * 1024:
        raise ValueError("Downloaded model checkpoint exceeds the 512 MiB safety limit.")
    actual_sha = hashlib.sha256(content).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("Downloaded model checkpoint SHA-256 does not match TERRAMIND_MODEL_SHA256.")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint.with_suffix(checkpoint.suffix + ".download")
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(checkpoint)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return checkpoint
