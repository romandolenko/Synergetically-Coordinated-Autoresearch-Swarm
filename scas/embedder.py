"""MiniLM-based embedding with on-disk SQLite cache.

Cache key is sha256(text); values are float32 vectors stored as numpy blobs.
First call downloads the model from HuggingFace; subsequent calls hit the cache.
"""

from __future__ import annotations

import hashlib
import io
import sqlite3
from pathlib import Path

import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"
_DIM = 384
_CACHE_DIR = Path.home() / ".cache" / "scas"
_CACHE_PATH = _CACHE_DIR / "embeddings.sqlite"


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Embedder:
    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        cache_path: Path = _CACHE_PATH,
    ):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._model_name = model_name
        self._conn = sqlite3.connect(str(cache_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "key TEXT PRIMARY KEY, vec BLOB NOT NULL)"
        )
        self._model = None
        self.hits = 0
        self.misses = 0

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

    def embed(self, text: str) -> np.ndarray:
        k = _key(text)
        row = self._conn.execute(
            "SELECT vec FROM embeddings WHERE key = ?", (k,)
        ).fetchone()
        if row is not None:
            self.hits += 1
            return np.load(io.BytesIO(row[0]), allow_pickle=False)

        self._ensure_model()
        vec = self._model.encode(text, normalize_embeddings=True).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, vec, allow_pickle=False)
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (key, vec) VALUES (?, ?)",
            (k, buf.getvalue()),
        )
        self._conn.commit()
        self.misses += 1
        return vec
