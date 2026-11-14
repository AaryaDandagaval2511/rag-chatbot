"""Application settings, read from environment variables (or a .env file at the repo root).

Every tunable value lives here, so the rest of the code never hard-codes paths or names.
Example: setting CHROMA_PERSIST_DIR=/data/chroma overrides the default storage folder.
That matters later in Docker, where paths and service addresses differ from local dev.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> parents[0]=app, [1]=backend, [2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unrelated variables in .env are not an error
    )

    # --- Vector store ---------------------------------------------------------------
    # Folder where ChromaDB keeps its files. Gitignored; deleting it clears the index.
    chroma_persist_dir: Path = REPO_ROOT / "vectorstore"
    chroma_collection: str = Field(default="documents", min_length=1)

    # --- Embeddings -----------------------------------------------------------------
    # Only the offline mock exists for now. A real provider (e.g. "ollama") is added by
    # extending this Literal and one branch in services/embeddings.py.
    embedding_provider: Literal["hash"] = "hash"
    embedding_dimensions: int = Field(default=256, gt=0)

    @field_validator("chroma_persist_dir")
    @classmethod
    def _resolve_relative_to_repo(cls, value: Path) -> Path:
        """A relative path in .env means "relative to the repo", not to wherever the
        command happens to be run from."""
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()


@lru_cache
def get_settings() -> Settings:
    """The app-wide settings, read once and then reused."""
    return Settings()
