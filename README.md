# Lucide Icon Scraper & Labeler

Downloads all Lucide icons grouped by category, then captions each icon
with a VLM to produce rich, searchable text for embedding-based icon search.

## Setup

```bash
uv sync
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

## Usage

### 1. Scrape icons (done)

```bash
uv run scraper.py                # full run
uv run scraper.py --limit 20     # test run
```

Pulls every icon's SVG + metadata from the Lucide GitHub repo and
sorts them into `output/<category>/<icon-name>.svg`. Icons in multiple
categories are copied into each. Produces `output/manifest.json` — the
source of truth mapping every icon name to its categories and tags.

### 2. Caption icons

```bash
uv run labeler.py                 # process everything not yet done
uv run labeler.py --limit 20      # test run
uv run labeler.py --retry-failed  # only reprocess previous failures
uv run labeler.py --skip-failed   # skip failures instead of auto-retrying
```

Renders each SVG to a black-on-white PNG and asks `gpt-4o-mini` to answer
four fixed questions about it (a reasoning schema — see below), enforced
via structured outputs so every field is always present and correctly typed.

## Resumability

Both scripts are safe to stop and re-run at any time.

- **Scraper**: re-run skips icons already present on disk.
- **Labeler**: tracks progress in `output/progress.jsonl`, one JSON line
  per icon (`{"name": ..., "status": "done"|"failed", ...}`). On each run:
  - icons marked `done` are skipped
  - icons marked `failed` are retried automatically (unless `--skip-failed`)
  - `--retry-failed` restricts the run to *only* previous failures

## Output files

| File | Contents |
|---|---|
| `output/<category>/<name>.svg` | Icon SVGs grouped by category |
| `output/manifest.json` | icon name → categories, tags |
| `output/progress.jsonl` | Per-icon caption run status (done/failed) |
| `output/captions.jsonl` | Final captions, one JSON object per icon |

Each line in `captions.jsonl`:

```json
{
  "name": "house",
  "categories": ["buildings"],
  "tags": ["home", "house", "shelter"],
  "what_shapes_are_visible": "...",
  "what_real_world_object_does_this_depict": "...",
  "what_ui_actions_or_concepts_does_this_represent": "...",
  "what_search_terms_would_find_this_icon": ["...", "..."]
}
```

## Why four fields instead of one caption

The labeler uses a **reasoning schema**: each field is a literal question,
answered in order, where later fields build on earlier ones (shapes seen →
object identified → UI meaning → search terms). This forces the model to
ground its interpretation in what's actually drawn before guessing at
intent, instead of free-associating a caption. All four fields are saved
as-is — no separate internal/output split — since for vector search the
intermediate reasoning is itself useful searchable text.

## Next step (not yet built)

Concatenate the four caption fields per icon into one text blob, embed
with `text-embedding-3-small`, and store alongside `name` + `categories`
in a vector DB for icon search.
