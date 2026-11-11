import asyncio
import os
import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore", message=".*allowed_objects.*")

import typer

from doc2skill.budget.guard import DEFAULT_MAX_TOKENS, estimate_tokens
from doc2skill.cache import disk as cache
from doc2skill.emit.skill import derive_skill_name, write_skill_file
from doc2skill.ingest.coalesce import AsyncSingleFlight
from doc2skill.ingest.url import load_url
from doc2skill.pipeline.graph import build_pipeline_graph
from doc2skill.pipeline.state import create_initial_state

_single_flight = AsyncSingleFlight()


class DefaultGroup(typer.core.TyperGroup):
    """Custom TyperGroup that defaults to the 'build' command when no command is specified."""

    def parse_args(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["build"] + args
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=DefaultGroup,
    name="doc2skill",
    help="Convert documentation into agent skills.",
    add_completion=False,
)


@app.command(name="build")
def main(
    source_path: Annotated[
        str,
        typer.Argument(
            help="Path to documentation directory or URL to ingest.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "--output-dir",
            "-o",
            help="Directory where output skills will be written.",
        ),
    ] = Path("./output"),
    max_tokens: Annotated[
        int,
        typer.Option(
            "--max-tokens",
            help="Maximum allowed token budget.",
        ),
    ] = DEFAULT_MAX_TOKENS,
    max_pages: Annotated[
        int,
        typer.Option(
            "--max-pages",
            help="Maximum pages to ingest from URL/sitemap.",
        ),
    ] = 100,
    no_sitemap: Annotated[
        bool,
        typer.Option(
            "--no-sitemap",
            help="Disable sitemap discovery for URLs.",
        ),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Bypass disk cache and re-run pipeline.",
        ),
    ] = False,
    no_llm: Annotated[
        bool,
        typer.Option(
            "--no-llm",
            help="Force fallback frontmatter and raw body without making LLM calls.",
        ),
    ] = False,
    no_condense: Annotated[
        bool,
        typer.Option(
            "--no-condense",
            help="Skip single-call corpus condensation and emit concatenated raw documents.",
        ),
    ] = False,
    max_output_tokens: Annotated[
        int,
        typer.Option(
            "--max-output-tokens",
            help="Maximum output tokens for condensed skill body.",
        ),
    ] = 15_000,
    max_context_tokens: Annotated[
        int,
        typer.Option(
            "--max-context-tokens",
            help="Maximum input context tokens before dropping deeply-nested documents.",
        ),
    ] = 500_000,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Gemini model name for frontmatter synthesis and condensation.",
        ),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Fail with exit code 1 if condensed output contains unverified identifiers.",
        ),
    ] = False,
) -> None:
    """Ingest markdown documentation or URL and emit a SKILL.md agent skill."""
    selected_model = model or os.getenv("DOC2SKILL_LLM_MODEL", "gemini-2.5-flash")
    is_url = source_path.startswith(("http://", "https://"))
    source_type = "url" if is_url else "markdown_dir"

    if not is_url:
        local_dir = Path(source_path)
        if not local_dir.exists() or not local_dir.is_dir():
            raise typer.BadParameter(f"Directory '{source_path}' does not exist.")

    skill_name = derive_skill_name(source_path)
    key = cache.cache_key(
        str(source_path),
        max_tokens=max_tokens,
        model=selected_model,
        source_type=source_type,
    )

    # Check cache unless --no-cache, --no-condense, or --strict is specified
    bypass_cache = no_cache or no_condense or strict
    cached_content = None if bypass_cache else cache.get(key)
    if cached_content is not None and not bypass_cache:
        typer.echo("Cache hit: writing from cache")
        out_file = write_skill_file(cached_content, skill_name, output_dir=output)
        est_tokens = estimate_tokens(cached_content)

        typer.echo("Summary:")
        typer.echo(f"  Source: {source_path}")
        typer.echo("  Documents used: cached")
        typer.echo(f"  Estimated input tokens: {est_tokens}")
        typer.echo(f"  Output: {out_file}")
        return

    # Cache miss or bypass: load content and run LangGraph pipeline
    loaded_documents = None
    if is_url:
        async def fetch_url():
            return await load_url(
                source_path,
                max_pages=max_pages,
                follow_sitemap=not no_sitemap,
            )

        try:
            loaded_documents = asyncio.run(_single_flight.do(source_path, fetch_url))
        except Exception as exc:
            typer.secho(
                f"Error fetching URL '{source_path}': {exc}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        if not loaded_documents:
            typer.secho(
                f"Error: No documents could be extracted from {source_path}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

    graph = build_pipeline_graph()
    initial_state = create_initial_state(
        source_path=str(source_path),
        max_tokens=max_tokens,
        output_dir=str(output),
        no_llm=no_llm,
        model=selected_model,
        documents=loaded_documents,
        max_pages=max_pages,
        max_output_tokens=max_output_tokens,
        max_context_tokens=max_context_tokens,
        no_condense=no_condense,
        strict=strict,
    )

    result = graph.invoke(initial_state)

    # Check if budget guard skipped the LLM or if LLM call failed
    has_budget_error = any("Budget exceeded" in err for err in result.get("errors", []))
    has_llm_error = any(
        "LLM synthesis failed" in err for err in result.get("errors", [])
    )
    fatal_errors = [
        err
        for err in result.get("errors", [])
        if "Budget exceeded" not in err and "LLM synthesis failed" not in err
    ]

    if fatal_errors:
        typer.secho(
            "Errors encountered during pipeline execution:",
            fg=typer.colors.RED,
            err=True,
        )
        for error in fatal_errors:
            typer.secho(f"  - {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if has_budget_error:
        typer.echo("LLM skipped (budget). Used fallback frontmatter.")
    elif has_llm_error:
        typer.echo("LLM call failed. Used fallback frontmatter.")
    elif no_llm:
        typer.echo("LLM skipped (--no-llm). Used fallback frontmatter.")

    skill_md = result.get("skill_md", "")
    output_path = result.get("output_path") or str(
        write_skill_file(skill_md, skill_name, output_dir=output)
    )

    # Store result in cache (unless --no-cache or --no-condense)
    if not no_cache and not no_condense:
        cache.set(key, skill_md)

    dropped_docs = result.get("dropped_documents", [])
    docs_loaded = len(result.get("documents", [])) + len(dropped_docs)
    docs_used = len(result.get("distilled_sources", [])) or len(
        result.get("documents", [])
    )
    truncated = bool(result.get("truncated", False))
    estimated_input_tokens = result.get("estimated_tokens", 0)
    actual_tokens = result.get("actual_tokens", 0)
    in_tokens = result.get("llm_input_tokens", 0)
    out_tokens = result.get("llm_output_tokens", 0)
    cond_in = result.get("condense_input_tokens", 0)
    cond_out = result.get("condense_output_tokens", 0)

    if is_url:
        sitemap_mode = (
            f"sitemap, {docs_loaded} pages"
            if (not no_sitemap and docs_loaded > 1)
            or source_path.endswith(".xml")
            or "sitemap" in source_path.lower()
            else "single page"
        )
        source_desc = f"{source_path} ({sitemap_mode})"
    else:
        source_desc = str(source_path)

    typer.echo("Summary:")
    typer.echo(f"  Source: {source_desc}")
    if truncated and dropped_docs:
        typer.echo(f"  Dropped: {len(dropped_docs)} docs (context limit)")
    typer.echo(f"  Documents used: {docs_used}")
    typer.echo(f"  Estimated input tokens: {estimated_input_tokens}")
    if actual_tokens > 0:
        typer.echo(
            f"  Actual tokens used: {actual_tokens} (in: {in_tokens}, out: {out_tokens})"
        )
    if cond_in + cond_out > 0:
        typer.echo(
            f"  Condensed tokens: in {cond_in}, out {cond_out} (total {cond_in + cond_out})"
        )
    validation_report = result.get("validation_report")
    if validation_report is not None:
        v_count = len(validation_report.get("verified", []))
        k_count = len(validation_report.get("known_external", []))
        sd_count = len(validation_report.get("source_divergent", []))
        u_count = len(validation_report.get("unverified", []))
        total_ids = v_count + k_count + sd_count + u_count
        precision = float(validation_report.get("precision", 1.0))
        inference_rate = float(validation_report.get("inference_rate", 0.0))
        typer.echo(
            f"  Validation: {v_count} verified, {k_count} known-external, "
            f"{sd_count} source-divergent ({inference_rate:.1%}), "
            f"{u_count} unverified ({precision:.1%} precision)"
        )
        if sd_count > 0:
            typer.secho(
                f"Warning: {sd_count} identifiers appear to be correct inferences from API signatures "
                "but are not in source docs. Review if you need source-faithful output.",
                fg=typer.colors.BLUE,
                err=True,
            )
        if u_count > 0:
            if total_ids > 0 and u_count > 0.2 * total_ids:
                typer.secho(
                    f"Warning: {u_count} unverified identifiers — the LLM may be hallucinating. Check the output carefully.",
                    fg=typer.colors.YELLOW,
                    err=True,
                )
            else:
                typer.secho(
                    f"Warning: {u_count} unverified identifier(s) not found in source docs.",
                    fg=typer.colors.YELLOW,
                    err=True,
                )
    typer.echo(f"  Output: {output_path}")


@app.command(name="serve")
def serve(
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="Host address to bind (default 127.0.0.1 for local-only).",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port to listen on.",
        ),
    ] = 8000,
    reload: Annotated[
        bool,
        typer.Option(
            "--reload",
            help="Enable auto-reload for development.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt when binding to 0.0.0.0.",
        ),
    ] = False,
) -> None:
    """Launch the local doc2skill web application."""
    if host == "0.0.0.0" and not yes:
        confirmed = typer.confirm(
            "This will expose the app to your network. Continue?",
            default=False,
        )
        if not confirmed:
            typer.echo("Aborted.")
            raise typer.Exit(code=1)

    if not os.environ.get("GEMINI_API_KEY"):
        typer.secho(
            "Warning: GEMINI_API_KEY is not set. LLM synthesis will be unavailable unless configured (use Skip LLM in UI).",
            fg=typer.colors.YELLOW,
            err=True,
        )

    typer.echo("doc2skill server starting")
    typer.echo(f"→ http://{host}:{port}")
    typer.echo("Press Ctrl+C to stop")

    import uvicorn

    if reload:
        uvicorn.run(
            "doc2skill.web.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
        )
    else:
        from doc2skill.web.app import create_app

        uvicorn.run(
            create_app(),
            host=host,
            port=port,
        )


@app.command(name="version", hidden=True)
def version() -> None:
    """Print package version."""
    import doc2skill

    typer.echo(f"doc2skill v{doc2skill.__version__}")


if __name__ == "__main__":
    app()
