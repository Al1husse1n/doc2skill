import warnings

warnings.filterwarnings("ignore", message=".*allowed_objects.*")

from pathlib import Path
from typing import Annotated

import typer

from doc2skill.budget.guard import DEFAULT_MAX_TOKENS, estimate_tokens
from doc2skill.cache import disk as cache
from doc2skill.emit.skill import derive_skill_name, write_skill_file
from doc2skill.pipeline.graph import build_pipeline_graph
from doc2skill.pipeline.state import create_initial_state

app = typer.Typer(
    name="doc2skill",
    help="Convert documentation into agent skills.",
    add_completion=False,
)


@app.command()
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
) -> None:
    """Ingest markdown documentation and emit a SKILL.md agent skill."""
    skill_name = derive_skill_name(source_path)
    config_hash = f"max_tokens={max_tokens}"
    key = cache.cache_key(str(source_path), config_hash)

    # Check cache unless --no-cache is specified
    cached_content = None if no_cache else cache.get(key)
    if cached_content is not None and not no_cache:
        typer.echo("Cache hit, writing from cache")
        out_file = write_skill_file(cached_content, skill_name, output_dir=output)
        est_tokens = estimate_tokens(cached_content)

        typer.echo("Summary:")
        typer.echo("  Documents loaded: cached")
        typer.echo(f"  Estimated tokens: {est_tokens}")
        typer.echo(f"  Output path: {out_file}")
        return

    # Cache miss or bypass: run LangGraph pipeline
    graph = build_pipeline_graph()
    initial_state = create_initial_state(
        source_path=str(source_path),
        max_tokens=max_tokens,
        output_dir=str(output),
    )

    result = graph.invoke(initial_state)

    if result.get("errors"):
        typer.secho(
            "Errors encountered during pipeline execution:",
            fg=typer.colors.RED,
            err=True,
        )
        for error in result["errors"]:
            typer.secho(f"  - {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    skill_md = result.get("skill_md", "")
    output_path = result.get("output_path") or str(
        write_skill_file(skill_md, skill_name, output_dir=output)
    )

    # Store result in cache
    cache.set(key, skill_md)

    docs_loaded = len(result.get("documents", []))
    estimated_tokens = result.get("estimated_tokens", 0)

    typer.echo("Summary:")
    typer.echo(f"  Documents loaded: {docs_loaded}")
    typer.echo(f"  Estimated tokens: {estimated_tokens}")
    typer.echo(f"  Output path: {output_path}")


if __name__ == "__main__":
    app()
