"""Build Noema's private local reference index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.indexer import build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Path to the user-provided reference PDF")
    args = parser.parse_args()
    result = build_index(args.pdf)
    print(
        f"Indexed {result['pages']} pages into {result['chunks']} local chunks "
        f"at {result['index']} ({result['skipped_pages']} pages skipped)."
    )


if __name__ == "__main__":
    main()
