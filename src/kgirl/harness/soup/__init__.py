"""Soup: shared, task-adaptive memory with staged, regularized self-improvement."""

from .curator import CurationPolicy, CurationReport, Curator
from .pool import KINDS, STATUSES, Fragment, Recall, Soup

__all__ = ["Soup", "Fragment", "Recall", "KINDS", "STATUSES", "Curator", "CurationPolicy", "CurationReport"]
