import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore", message=".*allowed_objects.*")

import typer

from doc2skill.budget.guard import DEFAULT_MAX_TOKENS, estimate_tokens
from doc2skill.cache import disk as cache
from doc2skill.emit.skill import derive_skill_name, write_skill_file
from doc2skill.pipeline.graph import build_pipeline_graph
from doc2skill.pipeline.state import create_initial_state


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
        Path,
        typer.Argument(
            help="Path to the documentation directory to ingest.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
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
            help="Force fallback frontmatter without making LLM calls.",
        ),
    ] = False,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Gemini model name for frontmatter synthesis.",
        ),
    ] = "gemini-2.5-flash",
) -> None:
    """Ingest markdown documentation and emit a SKILL.md agent skill."""
    skill_name = derive_skill_name(source_path)
    key = cache.cache_key(str(source_path), max_tokens=max_tokens, model=model)

    # Check cache unless --no-cache is specified
    cached_content = None if no_cache else cache.get(key)
    if cached_content is not None and not no_cache:
        typer.echo("Cache hit: writing from cache")
        out_file = write_skill_file(cached_content, skill_name, output_dir=output)
        est_tokens = estimate_tokens(cached_content)

        typer.echo("Summary:")
        typer.echo("  Documents loaded: cached")
        typer.echo(f"  Estimated input tokens: {est_tokens}")
        typer.echo(f"  Output: {out_file}")
        return

    # Cache miss or bypass: run LangGraph pipeline
    graph = build_pipeline_graph()
    initial_state = create_initial_state(
        source_path=str(source_path),
        max_tokens=max_tokens,
        output_dir=str(output),
        no_llm=no_llm,
        model=model,
    )

    result = graph.invoke(initial_state)

    # Check if budget guard skipped the LLM or if LLM call failed
    has_budget_error = any("Budget exceeded" in err for err in result.get("errors", []))
    has_llm_error = any(
        "LLM synthesis failed" in err for err in result.get("errors", [])
    )
    has_llm_error = any("LLM synthesis failed" in err for err in result.get("errors", []))
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

    # Store result in cache
    cache.set(key, skill_md)

    docs_loaded = len(result.get("documents", []))
    estimated_input_tokens = result.get("estimated_tokens", 0)
    actual_tokens = result.get("actual_tokens", 0)
    in_tokens = result.get("llm_input_tokens", 0)
    out_tokens = result.get("llm_output_tokens", 0)

    typer.echo("Summary:")
    typer.echo(f"  Documents loaded: {docs_loaded}")
    typer.echo(f"  Estimated input tokens: {estimated_input_tokens}")
    if actual_tokens > 0:
        typer.echo(
            f"  Actual tokens used: {actual_tokens} (in: {in_tokens}, out: {out_tokens})"
        )
    typer.echo(f"  Output: {output_path}")


@app.command(name="version", hidden=True)
def version() -> None:
    """Print package version."""
    import doc2skill

    typer.echo(f"doc2skill v{doc2skill.__version__}")


if __name__ == "__main__":
    app()
