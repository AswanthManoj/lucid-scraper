from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

EMBED_DIR = Path("embeddings")
VECTORS_PATH = EMBED_DIR / "vectors.npy"
METADATA_PATH = EMBED_DIR / "metadata.jsonl"

MODEL = "text-embedding-3-small"
TOP_K = 10


def load_index():
    vectors = np.load(VECTORS_PATH).astype(np.float32)

    # Normalize once for cosine similarity
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    metadata = []

    with METADATA_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))

    if len(metadata) != len(vectors):
        raise RuntimeError(
            f"Metadata ({len(metadata)}) and vectors ({len(vectors)}) differ."
        )

    return vectors, metadata


async def embed_query(client: AsyncOpenAI, query: str) -> np.ndarray:
    response = await client.embeddings.create(
        model=MODEL,
        input=query,
    )

    vector = np.asarray(
        response.data[0].embedding,
        dtype=np.float32,
    )

    vector /= np.linalg.norm(vector)

    return vector


async def main():
    print("Loading index...")

    vectors, metadata = load_index()

    client = AsyncOpenAI()

    print(f"Loaded {len(metadata)} icons.\n")

    while True:
        try:
            query = input("Search> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not query:
            continue

        query_vec = await embed_query(client, query)

        scores = vectors @ query_vec

        order = np.argsort(scores)[::-1][:TOP_K]

        print()

        for rank, idx in enumerate(order, start=1):
            item = metadata[idx]

            print(
                f"{rank:2d}. "
                f"{item['name']}.svg"
                f"  {scores[idx]:.4f}"
                f"  [{', '.join(item['categories'])}]"
            )

        print()


if __name__ == "__main__":
    asyncio.run(main())
