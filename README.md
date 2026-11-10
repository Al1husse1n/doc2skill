# doc2skill

doc2skill converts documentation (local markdown or a docs URL) into an agent skill — a compact `SKILL.md` that another agent can load to actually use the tool being documented.

## Installation

`doc2skill` is not published on PyPI yet. Install it as an editable package from source:

```bash
git clone https://github.com/Al1husse1n/doc2skill.git
cd doc2skill
pip install -e .
```

## Environment

Configure Gemini access via environment variables (no `.env` files are loaded automatically):

- `GEMINI_API_KEY` — Google Gemini API key (required for frontmatter synthesis and corpus condensation)
- `DOC2SKILL_LLM_MODEL` — Gemini model name (optional; defaults to `gemini-2.5-flash`)

### Token costs

A typical docs site build costs $0.01–$0.10 on `gemini-2.5-flash` depending on docs size. Built-in budget guards enforce `--max-tokens` (default `50000`) before invoking the LLM.

## Usage

Build a skill from a local markdown directory:

```bash
doc2skill build ./docs --output ./output
```

Build a skill from a documentation URL (discovers `sitemap.xml` when present, up to `--max-pages` 100 by default):

```bash
doc2skill build https://docs.example.com/ --output ./output
```

Launch the local web UI (`http://127.0.0.1:8000`) with live Server-Sent Events progress streaming and `SKILL.md` download:

```bash
doc2skill serve
```

Run `doc2skill build --help` or `doc2skill serve --help` for flags (`--max-tokens`, `--max-pages`, `--max-context-tokens`, `--no-cache`, `--no-condense`, `--no-llm`).

## What it does / What it doesn't do

### What it does
- **Single-pass LLM condensation**: turns ingested pages into structured `SKILL.md` sections (`API surface`, `Common workflows`, `Gotchas`, `Reference`, `Source`).
- **Context truncation + budget guard**: drops deeply-nested tail pages first when a corpus exceeds `--max-context-tokens`, then checks estimated token cost against `--max-tokens` before calling Gemini.
- **Disk cache**: caches built skills locally to avoid repeated LLM charges for identical inputs.
- **SSE web UI**: streams ingest, truncation, budget, LLM progress, and final skill download in a local-only web page.

### What it doesn't do (yet)
- **Hierarchical condensation**: corpora larger than `--max-context-tokens` are truncated, not map-reduced across multiple LLM passes.
- **Hallucination validation**: condensation includes refusal/stub checks and conservative prompting, not automated fact-checking against source APIs.
- **Auth or rate limiting**: `doc2skill serve` is a local developer UI without authentication, rate limiting, persistence, or multi-user isolation.

## Tests

```bash
pytest -q
```

Live network/LLM tests skip automatically unless `RUN_LIVE_TESTS=1` (and `GEMINI_API_KEY`) are set.
