from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

CAPTIONS_PATH = Path("output/captions.jsonl")

EMBED_DIR = Path("embeddings")
VECTORS_PATH = EMBED_DIR / "vectors.npy"
METADATA_PATH = EMBED_DIR / "metadata.jsonl"
PROGRESS_PATH = EMBED_DIR / "progress.jsonl"
MANIFEST_PATH = EMBED_DIR / "manifest.json"

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536

EMBED_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------


def load_progress() -> dict[str, dict]:
    if not PROGRESS_PATH.exists():
        return {}

    progress = {}

    with PROGRESS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            progress[rec["name"]] = rec

    return progress


def append_progress(record: dict):
    with PROGRESS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------


def load_captions() -> list[dict]:
    captions = []

    with CAPTIONS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            captions.append(json.loads(line))

    return captions


def build_text(record: dict) -> str:
    return f"""
Visible:
{record["what_shapes_are_visible"]}

Object:
{record["what_real_world_object_does_this_depict"]}

Meaning:
{record["what_ui_actions_or_concepts_does_this_represent"]}

Keywords:
{", ".join(record["what_search_terms_would_find_this_icon"])}

Categories:
{", ".join(record["categories"])}

Tags:
{", ".join(record["tags"])}
""".strip()


# ---------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
)
async def embed_text(client: AsyncOpenAI, text: str) -> list[float]:
    response = await client.embeddings.create(
        model=MODEL,
        input=text,
    )

    return response.data[0].embedding


async def process_one(client: AsyncOpenAI, record: dict):
    vector = await embed_text(client, build_text(record))

    metadata = {
        "name": record["name"],
        "categories": record["categories"],
        "tags": record["tags"],
    }

    return vector, metadata


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


async def run(limit: int | None, rebuild: bool):
    if rebuild and EMBED_DIR.exists():
        shutil.rmtree(EMBED_DIR)

    EMBED_DIR.mkdir(exist_ok=True)

    captions = load_captions()
    progress = load_progress()

    done = {
        name
        for name, rec in progress.items()
        if rec["status"] == "done"
    }

    todo = [
        c
        for c in captions
        if c["name"] not in done
    ]

    if limit:
        todo = todo[:limit]

    if not todo:
        print("Nothing to embed.")
        return

    print(f"Embedding {len(todo)} icons...")

    client = AsyncOpenAI()

    vectors = []

    if VECTORS_PATH.exists():
        vectors.extend(np.load(VECTORS_PATH).tolist())

    tasks = [
        process_one(client, record)
        for record in todo
    ]

    results = await tqdm_asyncio.gather(*tasks)

    with METADATA_PATH.open("a") as meta_file:

        for (vector, metadata), record in zip(results, todo):

            vectors.append(vector)

            meta_file.write(json.dumps(metadata) + "\n")

            append_progress(
                {
                    "name": record["name"],
                    "status": "done",
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )

    vectors = np.asarray(vectors, dtype=np.float32)

    np.save(VECTORS_PATH, vectors)

    manifest = {
        "model": MODEL,
        "dimensions": vectors.shape[1],
        "count": len(vectors),
        "created": datetime.now(timezone.utc).isoformat(),
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2)
    )

    print(f"\nSaved {len(vectors)} embeddings.")


# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
    )

    args = parser.parse_args()

    asyncio.run(
        run(
            limit=args.limit,
            rebuild=args.rebuild,
        )
    )


if __name__ == "__main__":
    main()
