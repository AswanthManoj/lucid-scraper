"""
Lucide icon scraper — groups SVGs by category using Lucide's own
GitHub metadata (icons/<name>.json has a required "categories" field).

Usage:
    uv run main.py                # scrape everything
    uv run main.py --limit 50     # test run on first 50 icons
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

GITHUB_API = "https://api.github.com/repos/lucide-icons/lucide/contents/icons"
RAW_BASE = "https://raw.githubusercontent.com/lucide-icons/lucide/main/icons"
OUTPUT_DIR = Path("output")
UNCATEGORIZED = "uncategorized"

# Be polite to the GitHub API; unauthenticated rate limit is 60 req/hour
# for the REST "contents" endpoint, but raw.githubusercontent.com fetches
# are unauthenticated file downloads and not subject to that limit.
CONCURRENCY = 10


def safe_name(name: str) -> str:
    """Sanitize category/icon names for filesystem use."""
    return re.sub(r"[^a-zA-Z0-9_\-]+", "-", name).strip("-").lower()


@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(5))
async def fetch_json(client: httpx.AsyncClient, url: str) -> httpx.Response:
    resp = await client.get(url)
    resp.raise_for_status()
    return resp


async def list_icon_names(client: httpx.AsyncClient, token: str | None) -> list[str]:
    """
    List all icon base names via the GitHub Trees API (single call,
    avoids pagination limits of the Contents API on large directories).
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Get the tree SHA for the icons/ directory via the recursive tree API
    url = "https://api.github.com/repos/lucide-icons/lucide/git/trees/main?recursive=1"
    resp = await client.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    names: set[str] = set()
    for entry in data["tree"]:
        path = entry["path"]
        if path.startswith("icons/") and path.endswith(".svg"):
            names.add(Path(path).stem)
    return sorted(names)


async def process_icon(
    client: httpx.AsyncClient,
    name: str,
    sem: asyncio.Semaphore,
    manifest: list[dict],
) -> None:
    async with sem:
        try:
            json_resp = await fetch_json(client, f"{RAW_BASE}/{name}.json")
            meta = json_resp.json()
        except Exception as e:
            print(f"  [warn] metadata fetch failed for {name}: {e}")
            meta = {}

        categories = meta.get("categories") or [UNCATEGORIZED]
        tags = meta.get("tags") or []

        try:
            svg_resp = await fetch_json(client, f"{RAW_BASE}/{name}.svg")
            svg_bytes = svg_resp.content
        except Exception as e:
            print(f"  [error] svg fetch failed for {name}: {e}")
            return

        for category in categories:
            cat_dir = OUTPUT_DIR / safe_name(category)
            cat_dir.mkdir(parents=True, exist_ok=True)
            (cat_dir / f"{name}.svg").write_bytes(svg_bytes)

        manifest.append(
            {
                "name": name,
                "categories": categories,
                "tags": tags,
            }
        )


async def run(limit: int | None, token: str | None) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)

    async with httpx.AsyncClient(timeout=30, limits=limits) as client:
        print("Listing icons from GitHub...")
        names = await list_icon_names(client, token)
        if limit:
            names = names[:limit]
        print(f"Found {len(names)} icons. Downloading...")

        sem = asyncio.Semaphore(CONCURRENCY)
        manifest: list[dict] = []

        tasks = [process_icon(client, name, sem, manifest) for name in names]
        await tqdm_asyncio.gather(*tasks)

    manifest.sort(key=lambda m: m["name"])
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Summary
    by_category: dict[str, int] = {}
    for entry in manifest:
        for c in entry["categories"]:
            by_category[c] = by_category.get(c, 0) + 1

    print("\nDone. Icons per category:")
    for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"  {cat:30s} {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Lucide icons grouped by category")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of icons (for testing)")
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional GitHub token to raise API rate limits (not required, only used for the tree listing call)",
    )
    args = parser.parse_args()
    asyncio.run(run(args.limit, args.token))


if __name__ == "__main__":
    main()
