"""vitrum — the Vitrum command line.

One entry point, lazily dispatched subcommands. Heavy imports (torch, the
architecture zoo) happen only inside the command actually being run.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import sys

_COMMANDS: dict[str, tuple[str, str]] = {
    "list": (
        "vitrum.cli.list_architectures",
        "List every architecture visible to the registry",
    ),
    "train": (
        "vitrum.cli.train",
        "Train one model (env-var configured; --nproc>1 wraps torchrun DDP)",
    ),
    "run-all": (
        "vitrum.cli.run_all",
        "Train every registered architecture sequentially, then print results",
    ),
    "benchmark": (
        "vitrum.cli.benchmark",
        "Run versions under identical wallclock conditions and compare",
    ),
    "observe": (
        "vitrum.cli.observe",
        "Observability toolkit: trace, wordmap, causality, coverage, ...",
    ),
    "results": (
        "vitrum.cli.results",
        "Aggregate logs/*_manifest.json into a results table",
    ),
    "microbench": (
        "vitrum.cli.microbench",
        "Fast forward/backward sanity benchmark across architectures",
    ),
    "data": (
        "vitrum.cli.data",
        "Dataset commands: download, prepare-code",
    ),
}


def _usage() -> str:
    width = max(map(len, _COMMANDS))
    lines = ["usage: vitrum <command> [args]", "", "commands:"]
    lines += [f"  {name:<{width}}  {help_text}" for name, (_, help_text) in _COMMANDS.items()]
    lines += ["", "run `vitrum <command> --help` for command-specific options"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(_usage())
        return 0
    if args[0] in ("-V", "--version"):
        print(f"vitrum {importlib.metadata.version('vitrum-cli')}")
        return 0
    command, rest = args[0], args[1:]
    if command not in _COMMANDS:
        print(f"vitrum: unknown command {command!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(_COMMANDS[command][0])
    return module.main(rest) or 0
