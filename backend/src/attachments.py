"""Files attached to chats and projects: images and documents.

Each owner (a chat or a project) keeps its files in its own folder:

    <root>/<file id><ext>         the original bytes
    <root>/<file id>.extracted.txt   extracted text (documents only), cached at upload time
    <root>/<file id>.meta.json       the metadata below

The metadata dict returned by `save()` is what messages and projects store:

    {"id", "name", "mime", "size", "kind": "image" | "document", "chars"}
"""
from __future__ import annotations

import base64
import csv
import io
import json
import mimetypes
import re
import uuid
from pathlib import Path

from src.logger import get_logger

_log = get_logger(__name__)

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_TEXT_CHARS = 400_000          # stored per document; the context builder trims further to fit the model
IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
_TEXT_EXT = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".xml", ".html", ".htm",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb", ".php", ".c", ".h", ".cpp", ".hpp", ".cs",
    ".swift", ".kt", ".sql", ".sh", ".toml", ".ini", ".cfg", ".log", ".tex", ".css", ".scss", ".r", ".m", ".lua",
}
_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_TEXT_SUFFIX = ".extracted.txt"
_META_SUFFIX = ".meta.json"


class AttachmentError(ValueError):
    """The upload can't be stored (too large, unsupported type, unreadable)."""


def _ext(name: str) -> str:
    return Path(name).suffix.lower()[:12]


def classify(name: str, mime: str | None) -> tuple[str, str]:
    """(kind, mime) for an upload, or raise AttachmentError if it isn't supported."""
    mime = (mime or mimetypes.guess_type(name)[0] or "application/octet-stream").lower()
    ext = _ext(name)
    if mime in IMAGE_TYPES:
        return "image", mime
    if ext in (".pdf", ".docx", ".xlsx") or ext in _TEXT_EXT or mime.startswith("text/") or mime == "application/json":
        return "document", mime
    raise AttachmentError(f"Unsupported file type: {name}. Attach images, PDFs, Word or Excel files, or text/code files.")


def save(root: Path, name: str, data: bytes, mime: str | None = None) -> dict:
    """Store an upload under root and return its metadata. Documents get their text extracted now."""
    if len(data) > MAX_FILE_BYTES:
        raise AttachmentError(f"{name} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
    kind, mime = classify(name, mime)
    file_id = uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{file_id}{_ext(name)}").write_bytes(data)
    meta = {"id": file_id, "name": Path(name).name[:200] or "file", "mime": mime, "size": len(data), "kind": kind}
    if kind == "document":
        text = extract_text(name, data)
        (root / f"{file_id}{_TEXT_SUFFIX}").write_text(text, encoding="utf-8")
        meta["chars"] = len(text)
    (root / f"{file_id}{_META_SUFFIX}").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def load_meta(root: Path, file_id: str) -> dict | None:
    """Metadata for a stored file id, or None."""
    if not _ID_RE.match(file_id or ""):
        return None
    try:
        return json.loads((root / f"{file_id}{_META_SUFFIX}").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_base64(root: Path, name: str, b64: str, mime: str | None = None) -> dict:
    try:
        data = base64.b64decode(b64.split(",", 1)[-1], validate=False)
    except Exception as exc:
        raise AttachmentError(f"{name}: not valid base64") from exc
    return save(root, name, data, mime)


def path_for(root: Path, meta_or_id: dict | str) -> Path | None:
    """The stored original for a file id (None if missing or the id is malformed)."""
    file_id = meta_or_id["id"] if isinstance(meta_or_id, dict) else meta_or_id
    if not _ID_RE.match(file_id or ""):
        return None
    return next((p for p in root.glob(f"{file_id}*")
                 if not p.name.endswith((_TEXT_SUFFIX, _META_SUFFIX))), None)


def image_base64(root: Path, meta: dict) -> str | None:
    p = path_for(root, meta)
    return base64.b64encode(p.read_bytes()).decode() if p else None


def text_of(root: Path, meta: dict) -> str:
    p = root / f"{meta['id']}{_TEXT_SUFFIX}"
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


def delete(root: Path, file_id: str) -> bool:
    if not _ID_RE.match(file_id or ""):
        return False
    found = False
    for p in root.glob(f"{file_id}*"):
        p.unlink(missing_ok=True)
        found = True
    return found


# ── Text extraction ─────────────────────────────────────────────────────

def extract_text(name: str, data: bytes) -> str:
    ext = _ext(name)
    try:
        if ext == ".pdf":
            text = _pdf(data)
        elif ext == ".docx":
            text = _docx(data)
        elif ext == ".xlsx":
            text = _xlsx(data)
        else:
            text = _decode(data)
            if ext == ".json":
                try:
                    text = json.dumps(json.loads(text), indent=2)[:MAX_TEXT_CHARS]
                except ValueError:
                    pass
            elif ext in (".csv", ".tsv"):
                text = _table_preview(text, "\t" if ext == ".tsv" else ",")
    except AttachmentError:
        raise
    except Exception as exc:
        _log.warning("Text extraction failed for %s: %s", name, exc)
        raise AttachmentError(f"Couldn't read {name}: {exc}") from exc
    text = text.strip()
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + f"\n\n[… truncated: the file has more than {MAX_TEXT_CHARS:,} characters]"
    return text


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _table_preview(text: str, delimiter: str) -> str:
    """CSV/TSV pass through as-is (models read them well); just note the shape."""
    rows = text.count("\n") + (0 if text.endswith("\n") else 1)
    try:
        header = next(csv.reader(io.StringIO(text.split("\n", 1)[0]), delimiter=delimiter))
        return f"[{rows} rows; columns: {', '.join(header[:40])}]\n{text}"
    except Exception:
        return text


def _pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise AttachmentError("Reading PDFs needs the `pypdf` package (pip install pypdf)") from exc
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        pages.append(f"--- page {i} ---\n{(page.extract_text() or '').strip()}")
    text = "\n\n".join(pages)
    if not re.sub(r"--- page \d+ ---|\s", "", text):
        raise AttachmentError("This PDF has no extractable text (it may be scanned images)")
    return text


def _docx(data: bytes) -> str:
    try:
        import docx
    except ImportError as exc:
        raise AttachmentError("Reading Word files needs the `python-docx` package (pip install python-docx)") from exc
    doc = docx.Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)


def _xlsx(data: bytes) -> str:
    try:
        import openpyxl
    except ImportError as exc:
        raise AttachmentError("Reading Excel files needs the `openpyxl` package (pip install openpyxl)") from exc
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        out.append(f"--- sheet: {ws.title} ---")
        for row in ws.iter_rows(values_only=True):
            if any(v is not None for v in row):
                out.append(",".join("" if v is None else str(v) for v in row))
    return "\n".join(out)
