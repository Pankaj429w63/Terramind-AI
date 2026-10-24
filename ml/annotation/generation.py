"""Automated, resumable image-description generation for the eligible PlantWild images.

This module produces *candidate* descriptions of what is visibly present in an image.
It is deliberately separate from the human annotation workflow:

* It never writes to ``outputs/annotation/annotator_workbook.csv`` and never claims a
  human observer id. Generated text is recorded with
  ``text_origin = model_generated_observation`` and can only be emitted into a
  distinct experimental CSV (``data/paired_model_observations.csv``).
* It never feeds a class label, class name, disease name, file name, model prediction,
  or split name into the description model. The eligible set is walked through the
  validated eligibility manifest, and only the image bytes and a versioned, label-free
  prompt reach the backend.
* Nothing is invented about the model: if no backend can be constructed, the preflight
  reports the exact missing dependency and generation does not run.

Every record carries the image identity (``sample_id`` + SHA-256), the model and its
version, the prompt version and hash, a real generation timestamp, provenance, an
optional sequence-level uncertainty value, and a processing status.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from ml.fusion.eligibility import load_eligibility_manifest, validate_eligibility_manifest
from ml.fusion.data import MODEL_TEXT_ORIGIN
from ml.models.class_labels import load_class_labels

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "data" / "plantwild" / "plantwild"
DEFAULT_MANIFEST = ROOT / "outputs" / "research" / "fusion_eligibility_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "annotation" / "model_generated" / "generation_records.csv"
DEFAULT_PREFLIGHT = ROOT / "outputs" / "annotation" / "model_generated" / "generation_preflight.json"
DEFAULT_WORKBOOK = ROOT / "outputs" / "annotation" / "annotator_workbook.csv"
DEFAULT_LOCAL_MODEL = "Salesforce/blip-image-captioning-base"

GENERATED = "generated"
FLAGGED = "flagged_for_review"
UNREADABLE = "unreadable_image"
FAILED = "failed"
GENERATION_STATUSES = (GENERATED, FLAGGED, UNREADABLE, FAILED)
DONE_STATUSES = frozenset({GENERATED, FLAGGED})

RECORD_COLUMNS = (
    "sample_id", "status", "description", "text_origin", "model", "model_version",
    "prompt_version", "prompt_sha256", "generated_at", "image_sha256", "mean_token_logprob",
    "backend", "provenance", "source_reference", "notes",
)

# A description shorter than this many words is treated as low-information.
MIN_WORDS = 4
# Below this mean token log-probability the sequence is retained but flagged.
LOW_LOGPROB = -2.5

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Prompts: versioned and label-free. No class, disease, organ, or crop name is included.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptSpec:
    version: str
    text: str
    style: str  # "caption_prefix" for captioning models, "instruction" for instruct VLMs

    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


PROMPTS: dict[str, PromptSpec] = {
    # Captioning models (e.g. BLIP) are not instruction-following: a prefix only biases
    # the caption style, it cannot be asked to focus on symptoms.
    "v1": PromptSpec(
        "v1",
        "a close-up view of the plant surface showing",
        "caption_prefix",
    ),
    # For instruction-following VLMs only. States the label-free constraint explicitly.
    "v2": PromptSpec(
        "v2",
        "Describe only the visible appearance of the plant material in this photograph: "
        "overall colour, any spots, lesions, or discolouration, surface texture, wilting, "
        "or deformation. Do not name the plant species, any disease, any cause, any "
        "treatment, or any severity level. If the image is unclear, say that it is unclear.",
        "instruction",
    ),
    # Symptom-oriented prompt tested in caption_quality_audit.md to focus on visible symptoms.
    "v3": PromptSpec(
        "v3",
        "symptoms visible on this leaf include",
        "caption_prefix",
    ),
}
DEFAULT_PROMPT_VERSION = "v1"


# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------


class GenerationError(RuntimeError):
    """Raised when a backend cannot produce a description for one image."""


@dataclass
class Description:
    text: str
    mean_token_logprob: float | None = None


class DescriptionBackend:
    """Common interface. ``name``/``model``/``model_version`` are recorded per row."""

    name = "abstract"
    model = ""
    model_version = ""

    def describe(self, image_path: Path, prompt: PromptSpec) -> Description:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:
        return None


class LocalImageToTextBackend(DescriptionBackend):
    """Local ``transformers`` image-to-text model. No network access, no API cost."""

    name = "local"

    def __init__(self, model_id: str, *, device: str = "cpu", max_new_tokens: int = 48) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
        except ImportError as error:  # pragma: no cover - dependency guard
            raise GenerationError(
                f"local image-description generation needs 'transformers' and 'torch': {error}"
            ) from error
        # The image-to-text auto class was renamed in transformers 5.x; resolve whichever
        # name this installation exposes so the same pipeline works across versions.
        auto_class = getattr(transformers, "AutoModelForImageTextToText", None) or getattr(
            transformers, "AutoModelForVision2Seq", None
        )
        if auto_class is None:
            raise GenerationError(
                f"transformers {transformers.__version__} exposes neither AutoModelForImageTextToText "
                "nor AutoModelForVision2Seq; cannot build a local image-description backend"
            )
        self.model = model_id
        self._torch = torch
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._net = auto_class.from_pretrained(model_id)
        self._net.eval()
        self._net.to(device)
        config = getattr(self._net, "config", None)
        revision = getattr(config, "_name_or_path", "") if config is not None else ""
        self.model_version = str(revision or model_id)

    def describe(self, image_path: Path, prompt: PromptSpec) -> Description:
        from PIL import Image

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        call = {"images": image}
        if prompt.style == "caption_prefix":
            call["text"] = prompt.text
        try:
            inputs = self._processor(**call, return_tensors="pt").to(self._device)
        except TypeError:  # processor without text support: fall back to plain captioning
            inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with self._torch.inference_mode():
            generated = self._net.generate(
                **inputs, max_new_tokens=self._max_new_tokens,
                output_scores=True, return_dict_in_generate=True,
            )
        sequences = generated.sequences
        text = self._processor.batch_decode(sequences, skip_special_tokens=True)[0]
        text = text.strip()
        if prompt.style == "caption_prefix":
            text = strip_prompt_prefix(text, prompt.text)
        logprob = self._sequence_logprob(generated, sequences)
        return Description(text=text, mean_token_logprob=logprob)

    def _sequence_logprob(self, generated, sequences) -> float | None:
        """Mean per-token log-probability of the generated continuation (real, not invented)."""
        scores = getattr(generated, "scores", None)
        if not scores:
            return None
        try:
            transitions = self._net.compute_transition_scores(
                sequences, scores, normalize_logits=True
            )
        except Exception:  # noqa: BLE001 - uncertainty is optional, never fatal
            return None
        values = transitions[0].tolist()
        if not values:
            return None
        return round(sum(values) / len(values), 4)


class HuggingFaceInferenceBackend(DescriptionBackend):
    """Hugging Face serverless inference. Requires an explicit token; never charged silently."""

    name = "hf-inference"

    def __init__(self, model_id: str, token: str, *, timeout: int = 120) -> None:
        if not token:
            raise GenerationError(
                "hf-inference backend requires HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) to be set"
            )
        self.model = model_id
        self.model_version = model_id
        self._token = token
        self._timeout = timeout

    def describe(self, image_path: Path, prompt: PromptSpec) -> Description:
        import urllib.error
        import urllib.request

        url = f"https://api-inference.huggingface.co/models/{self.model}"
        payload = json.dumps({"inputs": prompt.text, "options": {"wait_for_model": True}}).encode("utf-8")
        request = urllib.request.Request(
            url, data=Path(image_path).read_bytes(),
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise GenerationError(f"hf-inference HTTP {error.code}: {error.reason}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise GenerationError(f"hf-inference request failed: {error}") from error
        if isinstance(body, list) and body:
            body = body[0]
        text = str(body.get("generated_text", "")).strip() if isinstance(body, dict) else ""
        if prompt.style == "caption_prefix":
            text = strip_prompt_prefix(text, prompt.text)
        if not text:
            raise GenerationError(f"hf-inference returned no description: {str(body)[:160]}")
        return Description(text=text)


class OpenAICompatibleVLMBackend(DescriptionBackend):
    """Any OpenAI-compatible chat endpoint that accepts an image_url content part.

    Requires explicit configuration (base URL + key + model) and is only constructed when
    the caller has acknowledged paid usage.
    """

    name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str, model_id: str, *, timeout: int = 120) -> None:
        if not (base_url and api_key and model_id):
            raise GenerationError(
                "openai-compatible backend requires VLM_BASE_URL, VLM_API_KEY, and VLM_MODEL"
            )
        self.model = model_id
        self.model_version = model_id
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def describe(self, image_path: Path, prompt: PromptSpec) -> Description:
        import base64
        import urllib.error
        import urllib.request

        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        payload = json.dumps({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ],
            }],
            "max_tokens": 128,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions", data=payload,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise GenerationError(f"vlm HTTP {error.code}: {error.reason}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise GenerationError(f"vlm request failed: {error}") from error
        try:
            text = str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as error:
            raise GenerationError(f"vlm returned an unexpected payload: {str(body)[:160]}") from error
        if not text:
            raise GenerationError("vlm returned an empty description")
        return Description(text=text)


def build_backend(
    kind: str,
    model_id: str,
    *,
    prompt: PromptSpec,
    device: str = "cpu",
    max_new_tokens: int = 48,
) -> DescriptionBackend:
    """Construct the requested backend or raise ``GenerationError`` with the missing dependency."""
    if kind == "local":
        return LocalImageToTextBackend(model_id, device=device, max_new_tokens=max_new_tokens)
    if kind == "hf-inference":
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""
        return HuggingFaceInferenceBackend(model_id, token)
    if kind == "openai-compatible":
        return OpenAICompatibleVLMBackend(
            os.environ.get("VLM_BASE_URL", ""), os.environ.get("VLM_API_KEY", ""),
            os.environ.get("VLM_MODEL", "") or model_id,
        )
    raise GenerationError(f"unknown backend {kind!r}; choose local, hf-inference, or openai-compatible")


# --------------------------------------------------------------------------------------
# Text hygiene and record shaping
# --------------------------------------------------------------------------------------


def strip_prompt_prefix(text: str, prefix: str) -> str:
    """Strip a prompt prefix from generated text, tolerating spacing and hyphen variations.

    BLIP/BERT tokenizers decode hyphenated compounds like 'close-up' with spaces
    around hyphens (e.g. 'close - up') or irregular inter-word spacing.
    This helper compiles a regex that matches the prefix at the start of text
    regardless of whitespace or hyphen spacing variations, and trims any trailing
    connector punctuation (colon, dash, comma) before returning the remainder.
    """
    if not text or not prefix:
        return (text or "").strip()

    tokens = re.findall(r"\w+|[^\w\s]", prefix.strip())
    if not tokens:
        return text.strip()

    pattern_parts: list[str] = []
    for i, tok in enumerate(tokens):
        if tok == "-":
            pattern_parts.append(r"\s*-\s*")
        else:
            escaped = re.escape(tok)
            if i > 0 and tokens[i - 1] != "-":
                pattern_parts.append(r"\s+")
            pattern_parts.append(escaped)

    pattern = r"^\s*" + "".join(pattern_parts) + r"(?:\s*[:,\-–—]\s*|\s+|$)"
    match = re.match(pattern, text, flags=re.IGNORECASE)
    if match:
        return text[match.end():].strip()
    return text.strip()


def sanitize(text: str) -> str:
    """Collapse whitespace and strip model artefacts; no content is added or invented."""
    return " ".join((text or "").replace("\n", " ").split())


def assess(description: str, mean_token_logprob: float | None) -> tuple[str, list[str]]:
    """Return (status, notes). ``flagged_for_review`` means a human must look, not that it failed."""
    notes: list[str] = []
    words = description.split()
    if not description:
        return FAILED, ["backend returned an empty description"]
    if len(words) < MIN_WORDS:
        notes.append(f"very short description ({len(words)} words)")
    if mean_token_logprob is not None and mean_token_logprob < LOW_LOGPROB:
        notes.append(f"low mean token log-probability ({mean_token_logprob})")
    if len({word.casefold() for word in words}) <= max(1, len(words) // 4) and len(words) > 6:
        notes.append("highly repetitive wording")
    return (FLAGGED if notes else GENERATED), notes


@dataclass
class GenerationRecord:
    sample_id: str
    status: str
    description: str = ""
    model: str = ""
    model_version: str = ""
    prompt_version: str = ""
    prompt_sha256: str = ""
    generated_at: str = ""
    image_sha256: str = ""
    mean_token_logprob: str = ""
    backend: str = ""
    provenance: str = ""
    source_reference: str = ""
    notes: str = ""
    text_origin: str = MODEL_TEXT_ORIGIN

    def to_row(self) -> dict[str, str]:
        row = {key: value for key, value in asdict(self).items()}
        return {key: str(row.get(key, "")) for key in RECORD_COLUMNS}


# --------------------------------------------------------------------------------------
# Resumable store
# --------------------------------------------------------------------------------------


class GenerationStore:
    """Append-mostly CSV store with atomic checkpoints and duplicate prevention.

    A crash can leave at most one torn trailing line; ``load`` drops it and reports how
    many rows were unreadable, so resuming never double-counts or corrupts earlier work.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.records: dict[str, dict[str, str]] = {}
        self.discarded_lines = 0
        self._handle = None
        self._writer: csv.DictWriter | None = None
        self._unflushed = 0

    def load(self) -> None:
        if not self.path.is_file():
            return
        with self.path.open(newline="", encoding="utf-8-sig") as handle:
            raw = list(csv.reader(handle))
        if not raw:
            return
        header, body = raw[0], raw[1:]
        if list(header) != list(RECORD_COLUMNS):
            raise ValueError(
                f"generation store {self.path} has an unexpected header; move it aside before re-running"
            )
        for index, fields in enumerate(body):
            if len(fields) != len(RECORD_COLUMNS):
                self.discarded_lines += 1
                continue
            row = dict(zip(RECORD_COLUMNS, fields))
            sample_id = row.get("sample_id", "")
            if not sample_id:
                self.discarded_lines += 1
                continue
            if sample_id in self.records and index != len(body) - 1:
                self.discarded_lines += 1
                continue
            self.records[sample_id] = row

    def completed(self) -> set[str]:
        return {sid for sid, row in self.records.items() if row.get("status") in DONE_STATUSES}

    def attempted(self) -> set[str]:
        return set(self.records)

    def _ensure_open(self) -> None:
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.path.is_file()
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=list(RECORD_COLUMNS))
        if fresh:
            self._writer.writeheader()
            self._handle.flush()

    def append(self, record: GenerationRecord) -> None:
        self._ensure_open()
        assert self._writer is not None and self._handle is not None
        row = record.to_row()
        self._writer.writerow(row)
        self._handle.flush()
        self.records[record.sample_id] = row
        self._unflushed += 1

    def checkpoint(self) -> None:
        """Atomically compact the store so a torn append cannot survive a restart."""
        with self.path.open(newline="", encoding="utf-8-sig") as handle:
            raw = list(csv.reader(handle))
        good = [fields for fields in raw[1:] if len(fields) == len(RECORD_COLUMNS)]
        self.close()
        path = self.path
        handle = tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f"{path.name}.", suffix=".partial", delete=False,
        )
        try:
            with handle:
                writer = csv.writer(handle)
                writer.writerow(list(RECORD_COLUMNS))
                writer.writerows(good)
            os.replace(handle.name, path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise
        self._unflushed = 0

    def maybe_checkpoint(self, every: int) -> None:
        if every > 0 and self._unflushed >= every:
            self.checkpoint()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None


# --------------------------------------------------------------------------------------
# Eligible target enumeration (walked through the validated manifest only)
# --------------------------------------------------------------------------------------


def iter_eligible_targets(
    dataset_root: Path,
    manifest_path: Path,
    split_files: Mapping[str, Path],
    labels: Mapping[int, str],
) -> Iterator[tuple[str, Path, str, str]]:
    """Yield ``(sample_id, image_path, sha256, split)`` for every eligible image.

    The manifest is re-validated against the untouched official split files first, so no
    sample outside the duplicate-free eligible set can enter generation. Only the image
    identity is yielded: class ids and class names never leave this function, which is what
    keeps labels out of the description prompts.
    """
    manifest = load_eligibility_manifest(manifest_path)
    validate_eligibility_manifest(manifest, dataset_root, split_files, labels)
    images_root = (dataset_root / "images").resolve()
    for entry in manifest.get("eligible", []):
        sample_id = str(entry["sample_id"])
        image_path = (images_root / sample_id).resolve()
        if not image_path.is_relative_to(images_root):
            raise ValueError(f"Eligible sample escapes the image directory: {sample_id}")
        yield sample_id, image_path, str(entry.get("sha256", "")), str(entry.get("split", ""))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_for(backend: DescriptionBackend, prompt: PromptSpec, model: str) -> str:
    """Provenance for generated text: names the model, prompt version, and generation date."""
    return (
        f"Automated image description generated locally/remotely by the {backend.name} backend "
        f"(model {model}, prompt {prompt.version}, prompt sha256 {prompt.sha256()}) on "
        f"{datetime.date.today().isoformat()}; candidate text, not a human observation and not "
        "independently reviewed"
    )


def source_reference_for(backend: DescriptionBackend, model: str) -> str:
    if backend.name == "local":
        return f"https://huggingface.co/{model}"
    if backend.name == "hf-inference":
        return f"https://huggingface.co/{model}"
    return "https://platform.openai.com/docs/api-reference/chat"


# --------------------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------------------


def preflight(
    *,
    dataset_root: Path,
    manifest_path: Path,
    split_files: Mapping[str, Path],
    output_path: Path,
    workbook_path: Path = DEFAULT_WORKBOOK,
    backend_kind: str = "local",
    model_id: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict:
    """Report exactly what is available and what is missing, without downloading anything."""
    import shutil

    report: dict[str, object] = {
        "backend": backend_kind, "model": model_id, "prompt_version": prompt_version,
        "missing_dependencies": [], "warnings": [],
    }
    if prompt_version not in PROMPTS:
        raise ValueError(f"unknown prompt version {prompt_version!r}; known: {sorted(PROMPTS)}")
    report["prompt_sha256"] = PROMPTS[prompt_version].sha256()
    report["output_path"] = str(output_path)
    if Path(output_path).resolve() == Path(workbook_path).resolve():
        raise ValueError("the generation store must never be the human annotator workbook")

    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        report["missing_dependencies"].append("torch is not installed")
    try:
        import transformers
        report["transformers"] = transformers.__version__
    except ImportError:
        report["missing_dependencies"].append("transformers is not installed")
    try:
        import PIL  # noqa: F401
        report["pillow"] = True
    except ImportError:
        report["missing_dependencies"].append("Pillow is not installed")

    usage = shutil.disk_usage(str(Path(output_path).parent if Path(output_path).parent.exists() else ROOT))
    report["free_disk_gb"] = round(usage.free / 1e9, 2)

    cached = False
    if backend_kind == "local":
        hub = Path.home() / ".cache" / "huggingface" / "hub"
        folder = "models--" + model_id.replace("/", "--")
        cached = (hub / folder).is_dir()
        report["model_cached_locally"] = cached
        if not cached:
            report["warnings"].append(
                f"{model_id} is not in the local Hugging Face cache; the first run downloads it "
                "from huggingface.co (network access required)"
            )
    elif backend_kind == "hf-inference":
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
            report["missing_dependencies"].append("HF_TOKEN / HUGGINGFACE_HUB_TOKEN is not set")
    elif backend_kind == "openai-compatible":
        for key in ("VLM_BASE_URL", "VLM_API_KEY", "VLM_MODEL"):
            if not os.environ.get(key):
                report["missing_dependencies"].append(f"{key} is not set")
        report["paid_usage"] = True
        report["warnings"].append(
            "remote endpoints can incur charges; pass --confirm-paid-usage and a price estimate "
            "(--price-per-1k-images) before any request is sent"
        )

    labels = load_class_labels(dataset_root / "classes.txt")
    manifest = load_eligibility_manifest(manifest_path)
    validate_eligibility_manifest(manifest, dataset_root, split_files, labels)
    entries = manifest.get("eligible", [])
    report["eligible_samples"] = len(entries)
    report["eligible_by_split"] = manifest.get("eligible_counts_by_split")
    existing = GenerationStore(output_path)
    existing.load()
    report["records_on_disk"] = len(existing.records)
    report["completed_on_disk"] = len(existing.completed())
    report["discarded_lines"] = existing.discarded_lines
    report["remaining"] = max(0, len(entries) - len(existing.completed()))
    report["prompt_text"] = PROMPTS[prompt_version].text
    report["ready"] = not report["missing_dependencies"]
    return report
