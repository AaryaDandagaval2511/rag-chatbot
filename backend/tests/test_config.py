"""Tests for app.config.Settings."""

import pytest
from pydantic import ValidationError

from app.config import REPO_ROOT, Settings, get_settings

ENV_VARS = (
    "CHROMA_PERSIST_DIR",
    "CHROMA_COLLECTION",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_DIMENSIONS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ignore any of these variables set in the developer's shell."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults():
    settings = Settings(_env_file=None)

    assert settings.chroma_persist_dir == REPO_ROOT / "vectorstore"
    assert settings.chroma_collection == "documents"
    assert settings.embedding_provider == "hash"
    assert settings.embedding_dimensions == 256


def test_environment_variables_override_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path))
    monkeypatch.setenv("CHROMA_COLLECTION", "my_docs")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "64")

    settings = Settings(_env_file=None)

    assert settings.chroma_persist_dir == tmp_path
    assert settings.chroma_collection == "my_docs"
    assert settings.embedding_dimensions == 64  # converted from the string "64"


def test_values_are_read_from_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("CHROMA_COLLECTION=from_file\nUNRELATED_VARIABLE=ignored\n")

    settings = Settings(_env_file=env_file)

    assert settings.chroma_collection == "from_file"


def test_relative_persist_dir_is_relative_to_repo_root(monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "data/my_store")

    settings = Settings(_env_file=None)

    assert settings.chroma_persist_dir == REPO_ROOT / "data" / "my_store"


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("EMBEDDING_DIMENSIONS", "0"),
        ("EMBEDDING_DIMENSIONS", "-5"),
        ("EMBEDDING_DIMENSIONS", "lots"),
        ("EMBEDDING_PROVIDER", "ollama"),  # not implemented yet, so rejected up front
        ("CHROMA_COLLECTION", ""),
    ],
)
def test_invalid_values_are_rejected(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
