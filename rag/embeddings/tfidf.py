from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _PROJECT_ROOT / "outputs" / "rag" / "tfidf.pkl"
_CORPUS_PATH = _PROJECT_ROOT / "outputs" / "rag" / "tfidf_corpus.json"
_DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']+")
_OOV_RATIO_REFIT_THRESHOLD = 0.02
_OOV_ABSOLUTE_REFIT_THRESHOLD = 25


class TfidfEmbeddings:
    def __init__(
        self,
        max_features: int = 4096,
        ngram_range: tuple[int, int] = (1, 2),
        lowercase: bool = True,
        model_path: Path | None = None,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        self.model_path = Path(model_path) if model_path else _DEFAULT_PATH
        self.corpus_path = _CORPUS_PATH
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, lowercase=lowercase
        )
        self._fitted = False
        self._corpus: dict[str, str] = {}
        self._load()

    # ---------------------------
    # Persistence
    # ---------------------------
    def _load(self) -> None:
        if self.model_path.exists():
            try:
                with self.model_path.open("rb") as fh:
                    self.vectorizer = pickle.load(fh)
                self._fitted = True
            except Exception:
                self._fitted = False
        if self.corpus_path.exists():
            try:
                raw = json.loads(self.corpus_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._corpus = {str(k): str(v) for k, v in raw.items()}
            except Exception:
                self._corpus = {}

    def _save_model(self) -> None:
        try:
            with self.model_path.open("wb") as fh:
                pickle.dump(self.vectorizer, fh)
        except Exception:
            pass

    def _save_corpus(self) -> None:
        try:
            self.corpus_path.write_text(
                json.dumps(self._corpus, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def save(self) -> None:
        self._save_model()
        self._save_corpus()

    # ---------------------------
    # Properties
    # ---------------------------
    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def vocabulary_size(self) -> int:
        return len(getattr(self.vectorizer, "vocabulary_", {}) or {})

    @property
    def vector_size(self) -> int:
        if self._fitted:
            try:
                return int(getattr(self.vectorizer, "n_features_in_", self.vocabulary_size))
            except Exception:
                return self.vocabulary_size
        return 0

    @property
    def corpus(self) -> dict[str, str]:
        return dict(self._corpus)

    # ---------------------------
    # Vocabulary analysis
    # ---------------------------
    @staticmethod
    def _tokens(texts: list[str]) -> set[str]:
        tokens: set[str] = set()
        for t in texts:
            if not isinstance(t, str):
                continue
            for match in _WORD_RE.findall(t.lower() if True else t):
                tokens.add(match)
        return tokens

    def oov_ratio(self, texts: list[str]) -> tuple[float, int, int]:
        """Return (ratio_oov, count_oov, total_tokens) for given texts."""
        if not self._fitted:
            text_tokens = self._tokens(texts)
            return (1.0 if text_tokens else 0.0, len(text_tokens), len(text_tokens))
        vocab_tokens: set[str] = set()
        for tok in (getattr(self.vectorizer, "vocabulary_", {}) or {}).keys():
            if isinstance(tok, str) and " " not in tok:
                vocab_tokens.add(tok.lower())
        text_tokens = self._tokens(texts)
        if not text_tokens:
            return (0.0, 0, 0)
        oov = len(text_tokens - vocab_tokens)
        total = len(text_tokens)
        return ((oov / total) if total else 0.0, oov, total)

    def should_refit(self, texts: list[str]) -> bool:
        if not self._fitted:
            return True
        ratio, oov, _total = self.oov_ratio(texts)
        return (ratio >= _OOV_RATIO_REFIT_THRESHOLD) or (oov >= _OOV_ABSOLUTE_REFIT_THRESHOLD)

    # ---------------------------
    # Fitting / encoding with vocabulary stability
    # ---------------------------
    def fit(self, texts: list[str], corpus_ids: list[str] | None = None) -> None:
        """Fit the TF-IDF vocabulary. Caller should already have decided refit is required.

        Updates the persisted corpus: maps corpus_id -> original text so we can re-encode
        existing chunks when vocabulary changes.
        """
        if not texts:
            raise ValueError("Cannot fit on empty text list.")
        if corpus_ids is None:
            corpus_ids = [f"text:{i}" for i in range(len(texts))]
        if len(corpus_ids) != len(texts):
            raise ValueError("texts and corpus_ids must have equal length.")

        combined_ids: list[str] = list(self._corpus.keys())
        combined_texts: list[str] = [self._corpus[cid] for cid in combined_ids]
        for cid, text in zip(corpus_ids, texts):
            if not isinstance(text, str):
                continue
            if cid in self._corpus and self._corpus[cid] == text:
                continue
            if cid not in self._corpus:
                combined_ids.append(cid)
                combined_texts.append(text)
            else:
                idx = combined_ids.index(cid)
                combined_texts[idx] = text
            self._corpus[cid] = text

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=self.lowercase,
        )
        self.vectorizer.fit(combined_texts)
        self._fitted = True
        self.save()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbeddings must be fit before encoding.")
        matrix = self.vectorizer.transform(texts)
        return [list(map(float, row.toarray()[0])) for row in matrix]

    def fit_encode(
        self, texts: list[str], corpus_ids: list[str] | None = None
    ) -> list[list[float]]:
        self.fit(texts, corpus_ids=corpus_ids)
        return self.encode(texts)

    # ---------------------------
    # Corpus management (for re-encoding existing points after refit)
    # ---------------------------
    def update_corpus(self, items: dict[str, str]) -> None:
        changed = False
        for cid, text in items.items():
            if not isinstance(cid, str) or not isinstance(text, str):
                continue
            if self._corpus.get(cid) != text:
                self._corpus[cid] = text
                changed = True
        if changed:
            self._save_corpus()

    def ordered_corpus(self) -> tuple[list[str], list[str]]:
        ids = sorted(self._corpus.keys())
        return ids, [self._corpus[i] for i in ids]
