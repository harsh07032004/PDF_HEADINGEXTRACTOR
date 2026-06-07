#!/usr/bin/env python3
"""
process_pdfs.py — Legacy compatibility wrapper.

The extraction logic now lives in the `extractor` package.
This file is kept for backwards compatibility with the original
Challenge_1a specification (which expected this exact filename).

For new usage, prefer:
    extract-pdf <input_dir> <output_dir> [--lang en]
or import the package directly:
    from extractor import ExtractorEngine
"""

import sys
from pathlib import Path

# Delegate entirely to the new CLI entry point
from extractor.cli import main

if __name__ == "__main__":
    # Re-map legacy positional args: input output [lang]
    # New CLI: extract-pdf INPUT OUTPUT --lang LANG
    argv = list(sys.argv[1:])
    if len(argv) >= 3:
        # Third positional arg was the language code — convert to flag
        lang = argv.pop(2)
        argv += ["--lang", lang]
    sys.exit(main(argv))
