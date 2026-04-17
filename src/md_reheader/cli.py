import sys
from pathlib import Path
from typing import Annotated

import typer

DEFAULT_MODEL = "joelbarmettler/md-reheader"
MARKDOWN_SUFFIXES = {".md", ".markdown"}


def _resolve_device(gpu: bool, cpu: bool) -> str:
    if gpu and cpu:
        typer.secho("Error: --gpu and --cpu are mutually exclusive.", fg="red", err=True)
        raise typer.Exit(2)

    import torch

    if gpu:
        if not torch.cuda.is_available():
            typer.secho(
                "Error: --gpu requested but no CUDA device is available.",
                fg="red",
                err=True,
            )
            raise typer.Exit(1)
        return "cuda"
    if cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(
    input: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to the input markdown file (.md or .markdown).",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Where to write the reheaded markdown. If omitted, writes to stdout.",
            show_default=False,
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="HuggingFace repo id, local checkpoint, or model name served by --endpoint.",
        ),
    ] = DEFAULT_MODEL,
    endpoint: Annotated[
        str | None,
        typer.Option(
            "--endpoint",
            help=(
                "OpenAI-compatible API base URL (e.g. http://localhost:8000/v1). "
                "When set, inference runs remotely and --gpu/--cpu are ignored."
            ),
            show_default=False,
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="Bearer token for --endpoint. Ignored when --endpoint is unset.",
            show_default=False,
            envvar="MD_REHEADER_API_KEY",
        ),
    ] = None,
    gpu: Annotated[
        bool,
        typer.Option("--gpu", help="Force CUDA inference (local mode only)."),
    ] = False,
    cpu: Annotated[
        bool,
        typer.Option("--cpu", help="Force CPU inference (local mode only)."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite the output file if it already exists.",
        ),
    ] = False,
) -> None:
    """Restore heading hierarchy in a flattened markdown document."""
    if not input.exists():
        typer.secho(f"Error: input file does not exist: {input}", fg="red", err=True)
        raise typer.Exit(1)
    if not input.is_file():
        typer.secho(f"Error: input path is not a file: {input}", fg="red", err=True)
        raise typer.Exit(1)
    if input.suffix.lower() not in MARKDOWN_SUFFIXES:
        typer.secho(
            f"Error: input must be a .md or .markdown file (got {input.suffix or '<none>'}).",
            fg="red",
            err=True,
        )
        raise typer.Exit(1)

    if output is not None and output.exists() and not force:
        typer.secho(
            f"Error: output file already exists: {output}. Pass --force to overwrite.",
            fg="red",
            err=True,
        )
        raise typer.Exit(1)

    if endpoint is None and api_key is not None:
        typer.secho(
            "Error: --api-key requires --endpoint.",
            fg="red",
            err=True,
        )
        raise typer.Exit(2)
    if endpoint is not None and (gpu or cpu):
        typer.secho(
            "Error: --gpu/--cpu are ignored in remote mode — drop them or drop --endpoint.",
            fg="red",
            err=True,
        )
        raise typer.Exit(2)

    md_text = input.read_text()

    if endpoint is not None:
        from md_reheader.inference.remote import reheader_document_remote

        typer.secho(f"Calling {endpoint} with model {model!r}...", fg="blue", err=True)
        result = reheader_document_remote(
            md_text, endpoint=endpoint, api_key=api_key, model=model,
        )
    else:
        device = _resolve_device(gpu, cpu)

        from md_reheader.inference.predict import load_model, reheader_document

        typer.secho(f"Loading {model} on {device}...", fg="blue", err=True)
        loaded_model, tokenizer = load_model(model, device=device)

        typer.secho(f"Reheading {input}...", fg="blue", err=True)
        result = reheader_document(md_text, loaded_model, tokenizer)

    if output is None:
        sys.stdout.write(result)
        if not result.endswith("\n"):
            sys.stdout.write("\n")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result)
    typer.secho(f"Wrote {output}", fg="green", err=True)


def app() -> None:
    typer.run(main)


if __name__ == "__main__":
    app()
