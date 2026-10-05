from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import ChunkingContext


class BaseChunkStrategy(ABC):
    name = "base"

    @abstractmethod
    def chunk(
        self,
        context: ChunkingContext,
    ) -> list[dict]:
        raise NotImplementedError
