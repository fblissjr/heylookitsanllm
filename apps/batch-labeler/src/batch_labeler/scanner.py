"""Directory scanning and file hashing for batch image labeling."""

import hashlib
import os
from pathlib import Path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.tiff', '.tif', '.bmp'}


def file_hash(path: str) -> str:
    """Fast file hash using first 8KB + file size. Good enough for dedup."""
    h = hashlib.blake2b(digest_size=16)
    size = os.path.getsize(path)
    h.update(size.to_bytes(8, 'little'))
    with open(path, 'rb') as f:
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
    files = [p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in ext]
    files.sort()
    return files
