"""
extractor — Production-grade PDF document-structure extraction engine.

Public API:
    ExtractorEngine   — main class to process a single PDF
    DocumentOutline   — output data model
    Heading           — individual heading data model
"""

from extractor.core import ExtractorEngine
from extractor.models import DocumentOutline, Heading

__all__ = ["ExtractorEngine", "DocumentOutline", "Heading"]
__version__ = "0.1.0"
