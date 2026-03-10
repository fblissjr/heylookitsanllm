"""Directory scanning and file hashing for batch image labeling."""

import hashlib
import os
from pathlib import Path

# Single source of truth for supported image types.
# IMAGE_EXTENSIONS is derived from MIME_TYPES so they stay in sync.
MIME_TYPES = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.webp': 'image/webp',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp',
    '.tiff': 'image/tiff',
    '.tif': 'image/tiff',
    '.heic': 'image/heic',
    '.heif': 'image/heif',
}
IMAGE_EXTENSIONS = set(MIME_TYPES.keys())


def file_hash(path: str) -> str:
    """Fast file hash using first 8KB + file size. Good enough for dedup."""
    h = hashlib.blake2b(digest_size=16)
    with open(path, 'rb') as f:
        size = os.fstat(f.fileno()).st_size
        h.update(size.to_bytes(8, 'little'))
        h.update(f.read(8192))
    return h.hexdigest()


def scan_images(
    image_dir: str,
    recursive: bool = True,
    extensions: set[str] | None = None,
) -> list[Path]:
    """Walk directory and collect image files, sorted by name for determinism."""
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    ext = extensions or IMAGE_EXTENSIONS
    pattern = '**/*' if recursive else '*'
    # Check suffix before is_file() to skip stat on non-image entries
    files = [p for p in root.glob(pattern) if p.suffix.lower() in ext and p.is_file()]
    files.sort()
    return files
