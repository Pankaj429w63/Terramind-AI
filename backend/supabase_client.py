from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # Optional for callers that inject environment variables directly.
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env.local", override=False)

try:
    from supabase import Client, create_client
except ImportError:  # Optional until Supabase is configured.
    Client = Any  # type: ignore[misc,assignment]
    create_client = None


class SupabaseConfig:
    def __init__(self) -> None:
        self.url = os.getenv("SUPABASE_URL", "").strip()
        # Database writes run server-side and require the service-role key.
        # The public anon key belongs only in the browser Supabase client.
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.key and create_client is not None)

    @property
    def missing_variables(self) -> list[str]:
        missing = []
        if not self.url:
            missing.append("SUPABASE_URL")
        if not os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
        if create_client is None:
            missing.append("Python package: supabase")
        return missing


CONFIG = SupabaseConfig()
_CLIENT: Optional[Client] = None


def get_supabase() -> Optional[Client]:
    global _CLIENT
    if not CONFIG.enabled:
        return None
    if _CLIENT is None:
        _CLIENT = create_client(CONFIG.url, CONFIG.key)
    return _CLIENT
