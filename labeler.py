"""
Render each icon SVG to a black-on-white PNG, caption it with a VLM using
a reasoning schema, and save the schema fields directly (no separate
internal/response split — see design discussion).

Resumable via output/progress.jsonl, same pattern as before.

Usage:
    uv run caption.py                 # process everything not yet done
    uv run caption.py --limit 50      # test run
    uv run caption.py --retry-failed  # only retry previous failures
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
from pathlib import Path
from datetime import datetime, timezone

import resvg_py
from PIL import Image
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

OUTPUT_DIR = Path("output")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
PROGRESS_PATH = OUTPUT_DIR / "progress.jsonl"
CAPTIONS_PATH = OUTPUT_DIR / "captions.jsonl"

CONCURRENCY = 8
MODEL = "gpt-4o-mini"
PNG_SIZE = 256  # render resolution, px


# ---------------------------------------------------------------------------
# Reasoning schema — fields ARE the saved output, phrased as literal questions
# ---------------------------------------------------------------------------

class IconCaption(BaseModel):
    what_shapes_are_visible: str = Field(
        ...,
        description="List the literal geometric shapes and their arrangement "
        "in the icon. No interpretation of what it represents or means. "
        "Under 40 words.",
    )
    what_real_world_object_does_this_depict: str = Field(
        ...,
        description="Name the real-world object or symbol these shapes most "
        "resemble, based on the shapes above. One sentence, under 20 words.",
    )
    what_ui_actions_or_concepts_does_this_represent: str = Field(
        ...,
        description="Given the object identified above, list the UI actions, "
        "states, or concepts it's conventionally used for. You may include "
        "common associations even if not strictly derived from the shape. "
        "3-6 short phrases, comma separated.",
    )
    what_search_terms_would_find_this_icon: list[str] = Field(
        ...,
        description="Words or phrases someone might type to search for this "
        "icon in an icon picker. Include the object name, the UI concepts "
        "above, and close synonyms. 4-8 items.",
    )


PROMPT = """You are labeling a UI icon for a semantic search index used by an \
AI website builder. The image is a single icon rendered in black on a white \
background. Answer each field in order, building on the previous answer."""


# ---------------------------------------------------------------------------
# SVG -> PNG rendering
# ---------------------------------------------------------------------------

def svg_to_png_bytes(svg_bytes: bytes, size: int = PNG_SIZE) -> bytes:
    """Render SVG to a black-on-white PNG at a fixed resolution."""
    svg_text = svg_bytes.decode("utf-8")
    png_bytes = resvg_py.svg_to_bytes(
        svg_string=svg_text,
        width=size,
        height=size,
    )
    # Composite onto white background (resvg renders with transparency)
    img = Image.open(io.BytesIO(bytes(png_bytes))).convert("RGBA")
    bg = Image.new("RGBA", img.size, "white")
    flattened = Image.alpha_composite(bg, img).convert("RGB")
    buf = io.BytesIO()
    flattened.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Progress tracking (same resumable pattern as the scraper)
# ---------------------------------------------------------------------------

def load_manifest() -> dict[str, dict]:
    data = json.loads(MANIFEST_PATH.read_text())
    return {entry["name"]: entry for entry in data}


def load_progress() -> dict[str, dict]:
    if not PROGRESS_PATH.exists():
        return {}
    progress: dict[str, dict] = {}
    with PROGRESS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            progress[rec["name"]] = rec  # last write wins
    return progress


def append_progress(record: dict) -> None:
    with PROGRESS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def append_caption(record: dict) -> None:
    with CAPTIONS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "-", name).strip("-").lower()


def find_svg_path(name: str, categories: list[str]) -> Path | None:
    for cat in categories:
        p = OUTPUT_DIR / safe_name(cat) / f"{name}.svg"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Captioning
# ---------------------------------------------------------------------------

@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(4))
async def caption_one(client: AsyncOpenAI, png_bytes: bytes, name: str, tags: list[str]) -> IconCaption:
    b64 = base64.b64encode(png_bytes).decode()
    data_url = f"data:image/png;base64,{b64}"

    completion = await client.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{PROMPT}\n\nIcon file name: {name}\nKnown tags: {', '.join(tags)}"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        response_format=IconCaption,
        max_tokens=500,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        refusal = completion.choices[0].message.refusal
        raise ValueError(f"model returned no parsed output (refusal: {refusal})")
    return parsed


async def process_one(
    client: AsyncOpenAI,
    name: str,
    entry: dict,
    sem: asyncio.Semaphore,
) -> None:
    async with sem:
        svg_path = find_svg_path(name, entry["categories"])
        if svg_path is None:
            append_progress({
                "name": name,
                "status": "failed",
                "error": "svg file not found on disk",
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            return

        try:
            svg_bytes = svg_path.read_bytes()
            png_bytes = svg_to_png_bytes(svg_bytes)
            caption = await caption_one(client, png_bytes, name, entry.get("tags", []))

            append_caption({
                "name": name,
                "categories": entry["categories"],
                "tags": entry.get("tags", []),
                **caption.model_dump(),
            })
            append_progress({
                "name": name,
                "status": "done",
                "ts": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            append_progress({
                "name": name,
                "status": "failed",
                "error": str(e),
                "ts": datetime.now(timezone.utc).isoformat(),
            })


async def run(limit: int | None, retry_failed_only: bool, skip_failed: bool) -> None:
    manifest = load_manifest()
    progress = load_progress()

    if retry_failed_only:
        todo = [name for name, rec in progress.items() if rec["status"] == "failed"]
    else:
        done_names = {n for n, r in progress.items() if r["status"] == "done"}
        failed_names = {n for n, r in progress.items() if r["status"] == "failed"}
        todo = [
            name for name in manifest
            if name not in done_names and not (skip_failed and name in failed_names)
        ]

    if limit:
        todo = todo[:limit]

    if not todo:
        print("Nothing to do — all icons already captioned (or use --retry-failed).")
        return

    print(f"Captioning {len(todo)} icons (skipping {len(manifest) - len(todo)} already done)...")

    client = AsyncOpenAI()  # reads OPENAI_API_KEY from env
    sem = asyncio.Semaphore(CONCURRENCY)

    tasks = [process_one(client, name, manifest[name], sem) for name in todo]
    await tqdm_asyncio.gather(*tasks)

    final_progress = load_progress()
    n_done = sum(1 for r in final_progress.values() if r["status"] == "done")
    n_failed = sum(1 for r in final_progress.values() if r["status"] == "failed")
    print(f"\nDone: {n_done} succeeded, {n_failed} failed. Re-run to retry failures automatically.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.limit, args.retry_failed, args.skip_failed))


if __name__ == "__main__":
    main()
