#!/usr/bin/env python3
"""One-off verification sweep: fetch every agriculture source URL and report status.

Not part of the KB pipeline; used only to confirm no fabricated/broken URLs are
written into the knowledge base.
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _populate_agriculture import (
    _agriculture_records,
    _agriculture_records_2,
    _agriculture_records_3,
    _agriculture_records_4,
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def main() -> int:
    records: dict[str, object] = {}
    for part in (_agriculture_records, _agriculture_records_2, _agriculture_records_3, _agriculture_records_4):
        records.update(part())

    bad: list[tuple[str, str, str]] = []
    for key, rec in sorted(records.items()):
        url = rec.url  # type: ignore[attr-defined]
        req = urllib.request.Request(url, headers=HEADERS, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                print(f"OK   {resp.status}  {key:<14} {url}")
        except urllib.error.HTTPError as exc:
            print(f"FAIL {exc.code}  {key:<14} {url}")
            bad.append((key, str(exc.code), url))
        except Exception as exc:  # noqa: BLE001
            print(f"ERR  ----  {key:<14} {url}  ({exc})")
            bad.append((key, type(exc).__name__, url))

    print()
    print(f"checked={len(records)} bad={len(bad)}")
    for key, code, url in bad:
        print(f"  BAD {key}: {code} {url}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
