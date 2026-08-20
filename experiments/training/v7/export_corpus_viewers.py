"""Export corpus HTML viewers for Tokyo Eye v7 checkpoints."""

from __future__ import annotations

import sys


def main() -> None:
    from experiments.training.v66.export_corpus_viewers import main as shared_main

    shared_main()


if __name__ == "__main__":
    main()
