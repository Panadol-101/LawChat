"""Human-readable legal source resolution."""

from .resolver import (
    DocumentNotFoundError,
    SourceResolution,
    SourceResolutionService,
    create_source_resolution_service,
)

__all__ = [
    "DocumentNotFoundError",
    "SourceResolution",
    "SourceResolutionService",
    "create_source_resolution_service",
]
