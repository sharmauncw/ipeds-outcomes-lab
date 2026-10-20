"""01 — Download IPEDS bulk files from the NCES Data Center.

Source: https://nces.ed.gov/ipeds/datacenter/data/{STEM}.zip
This is the canonical bulk distribution of the Integrated Postsecondary
Education Data System (IPEDS). The Urban Institute Education Data Portal API
was attempted first on the build date and returned HTTP 500 on every IPEDS
endpoint, so the pipeline uses NCES bulk files directly (see RUN_LOG.md).

Files that do not exist yet for a given year (NCES posts final files on a
lag) are logged in outputs/download_manifest.json and skipped — downstream
steps treat the corresponding measures as missing for that year rather than
failing.

Usage: .venv/bin/python src/01_download.py
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import NCES_BASE, OM_YEARS, PANEL_YEARS, RAW_DIR, file_stems_for_year


def fetch(stem: str) -> dict:
    dest = RAW_DIR / f"{stem}.zip"
    url = f"{NCES_BASE}/{stem}.zip"
    if dest.exists() and dest.stat().st_size > 10_000:
        return {"stem": stem, "status": "cached", "bytes": dest.stat().st_size}
    req = urllib.request.Request(url, headers={"User-Agent": "ipeds-outcomes-lab/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return {"stem": stem, "status": "downloaded", "bytes": len(data)}
    except Exception as exc:  # 404 for not-yet-posted years, network, etc.
        return {"stem": stem, "status": f"unavailable: {exc.__class__.__name__}", "bytes": 0}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stems: list[str] = []
    for year in PANEL_YEARS:
        stems.extend(file_stems_for_year(year))
    stems.extend(f"OM{y}" for y in OM_YEARS)
    stems = sorted(set(stems))
    print(f"Fetching {len(stems)} IPEDS files ...")
    results = []
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        for res in pool.map(fetch, stems):
            results.append(res)
            print(f"  {res['stem']:<14} {res['status']:<28} {res['bytes']:>10,} B")
    manifest = {
        "source": "NCES IPEDS Data Center bulk downloads",
        "base_url": NCES_BASE,
        "files": results,
        "downloaded_or_cached": sum(r["status"] in ("downloaded", "cached") for r in results),
        "unavailable": [r["stem"] for r in results if r["status"] not in ("downloaded", "cached")],
    }
    out = Path(__file__).resolve().parent.parent / "outputs" / "download_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"\nOK: {manifest['downloaded_or_cached']}/{len(results)} files present")
    print(f"Unavailable: {manifest['unavailable']}")


if __name__ == "__main__":
    main()
