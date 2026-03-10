# src/heylook_llm/batch_vision_pipeline.py
"""
Batch vision labeling pipeline for processing directories of images.

Orchestrates a multi-stage pipeline:
1. Directory scan + deduplication (file hash)
2. Shared system prompt prefix cache (computed once, forked per-image)
3. Sequential VLM processing with progress tracking
4. SQLite result storage with resume support

Designed for long-running jobs (50K+ images over hours). Key properties:
- Resume from crashes: skips already-processed files via SQLite lookup
- Model pinning: prevents LRU eviction during multi-hour runs
- Memory hygiene: periodic mx.clear_cache() to prevent pool growth
- Progress tracking: real-time status via job manager
"""

import gc
import hashlib
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import mlx.core as mx

from heylook_llm.optimizations import fast_json as json


# Supported image extensions (lowercase)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.tiff', '.tif', '.bmp'}

# How often to call mx.clear_cache() to prevent memory pool growth
CACHE_CLEAR_INTERVAL = 50


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchVisionConfig:
    """Configuration for a batch vision labeling job."""
    image_dir: str
    model_id: str
    system_prompt: str
    output_db: str  # Path to SQLite database for results
    batch_size: int = 1  # Images per processing cycle (vision encoding is sequential for now)
    max_tokens: int = 1024  # Max tokens for JSON output
    temperature: float = 0.1  # Low temp for structured output
    recursive: bool = True  # Scan subdirectories
    extensions: set[str] = field(default_factory=lambda: IMAGE_EXTENSIONS.copy())


@dataclass
class JobProgress:
    """Snapshot of a running job's progress."""
    job_id: str
    status: JobStatus
    total_images: int
    completed: int
    failed: int
    skipped: int  # Already processed (resume)
    images_per_second: float
    elapsed_seconds: float
    eta_seconds: float  # Estimated time remaining
    current_file: str = ""
    error: str = ""


class ResultDatabase:
    """SQLite storage for batch labeling results.

    Schema is the three-layer design: technical metadata, AI labels, human overrides.
    This class handles layer 1 (file metadata) and layer 2 (AI labels).
    Layer 3 (human corrections) is left for the review UI.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            self._conn.executescript("""
                -- Layer 1: immutable file metadata (extracted once)
                CREATE TABLE IF NOT EXISTS files (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_size INTEGER,
                    file_modified TEXT,
                    exif_json TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- Layer 2: AI-generated labels (re-runnable, versioned)
                CREATE TABLE IF NOT EXISTS labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT NOT NULL REFERENCES files(file_hash),
                    model_id TEXT NOT NULL,
                    label_json TEXT NOT NULL,
                    raw_output TEXT,
                    tokens_used INTEGER,
                    generation_time_ms INTEGER,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(file_hash, model_id)
                );

                -- Layer 3: human corrections (never overwritten by AI)
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT NOT NULL REFERENCES files(file_hash),
                    correction_json TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- Job tracking for resume support
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    started_at TEXT,
                    completed_at TEXT,
                    total_images INTEGER DEFAULT 0,
                    processed INTEGER DEFAULT 0,
                    failed INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_labels_file ON labels(file_hash);
                CREATE INDEX IF NOT EXISTS idx_labels_model ON labels(model_id);
                CREATE INDEX IF NOT EXISTS idx_files_path ON files(file_path);
            """)

    def is_processed(self, file_hash: str, model_id: str) -> bool:
        """Check if a file has already been labeled by this model."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT 1 FROM labels WHERE file_hash = ? AND model_id = ?",
                (file_hash, model_id)
            )
            return cur.fetchone() is not None

    def store_file(self, file_hash: str, file_path: str, file_name: str,
                   file_size: int, file_modified: str):
        """Insert or ignore file metadata (layer 1)."""
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO files (file_hash, file_path, file_name, file_size, file_modified)
                   VALUES (?, ?, ?, ?, ?)""",
                (file_hash, file_path, file_name, file_size, file_modified)
            )
            self._conn.commit()

    def store_label(self, file_hash: str, model_id: str, label_json: str,
                    raw_output: str, tokens_used: int, generation_time_ms: int):
        """Insert or replace AI label (layer 2). New model runs overwrite old ones."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO labels
                   (file_hash, model_id, label_json, raw_output, tokens_used, generation_time_ms)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (file_hash, model_id, label_json, raw_output, tokens_used, generation_time_ms)
            )
            self._conn.commit()

    def store_job(self, job_id: str, config: BatchVisionConfig):
        """Record a job for resume tracking."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO jobs (job_id, config_json, status, started_at)
                   VALUES (?, ?, 'running', datetime('now'))""",
                (job_id, json.dumps({
                    'image_dir': config.image_dir,
                    'model_id': config.model_id,
                    'output_db': config.output_db,
                }))
            )
            self._conn.commit()

    def update_job(self, job_id: str, status: str, processed: int, failed: int, total: int):
        with self._lock:
            self._conn.execute(
                """UPDATE jobs SET status = ?, processed = ?, failed = ?, total_images = ?,
                   completed_at = CASE WHEN ? IN ('completed', 'failed', 'cancelled')
                                       THEN datetime('now') ELSE completed_at END
                   WHERE job_id = ?""",
                (status, processed, failed, total, status, job_id)
            )
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()


def _file_hash(path: str) -> str:
    """Fast file hash using first 8KB + file size. Good enough for dedup."""
    h = hashlib.blake2b(digest_size=16)
    size = os.path.getsize(path)
    h.update(size.to_bytes(8, 'little'))
    with open(path, 'rb') as f:
        h.update(f.read(8192))
    return h.hexdigest()


def _scan_images(image_dir: str, recursive: bool, extensions: set[str]) -> list[Path]:
    """Walk directory and collect image files, sorted by name for determinism."""
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    files = []
    pattern = '**/*' if recursive else '*'
    for p in root.glob(pattern):
        if p.is_file() and p.suffix.lower() in extensions:
            files.append(p)

    files.sort()
    return files


def _extract_json(text: str) -> str | None:
    """Try to extract valid JSON from model output, handling markdown fences."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith('```'):
        lines = text.split('\n')
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == '```':
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = '\n'.join(lines).strip()

    # Try parsing as-is
    try:
        parsed = json.loads(text)
        return json.dumps(parsed)  # Re-serialize for consistent formatting
    except (ValueError, TypeError):
        pass

    # Try finding JSON object boundaries
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            return json.dumps(parsed)
        except (ValueError, TypeError):
            pass

    return None


class BatchVisionLabelJob:
    """Long-running job that processes a directory of images through VLM labeling.

    Designed for resilience:
    - Pins model in router cache to prevent eviction
    - Writes results to SQLite as they complete
    - Skips already-processed images on resume
    - Periodic Metal cache cleanup to prevent memory bloat
    """

    def __init__(self, config: BatchVisionConfig, router):
        self.config = config
        self.router = router
        self.job_id = f"batch-vision-{uuid.uuid4().hex[:12]}"
        self.db = ResultDatabase(config.output_db)

        # Progress tracking
        self.status = JobStatus.PENDING
        self.total_images = 0
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.current_file = ""
        self.error = ""
        self._start_time = 0.0
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    def get_progress(self) -> JobProgress:
        """Thread-safe progress snapshot."""
        with self._lock:
            elapsed = time.time() - self._start_time if self._start_time else 0.0
            processed = self.completed + self.failed
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = self.total_images - processed - self.skipped
            eta = remaining / rate if rate > 0 else 0.0

            return JobProgress(
                job_id=self.job_id,
                status=self.status,
                total_images=self.total_images,
                completed=self.completed,
                failed=self.failed,
                skipped=self.skipped,
                images_per_second=round(rate, 2),
                elapsed_seconds=round(elapsed, 1),
                eta_seconds=round(eta, 1),
                current_file=self.current_file,
                error=self.error,
            )

    def cancel(self):
        """Request cancellation (cooperative, checked between images)."""
        self._cancel.set()

    def run(self):
        """Main job loop. Blocks until complete, cancelled, or failed.

        Call from a background thread (the API endpoint wraps this in asyncio.to_thread).
        """
        self._start_time = time.time()
        self.status = JobStatus.RUNNING

        try:
            # 1. Scan directory
            logging.info(f"[BATCH VISION] Scanning {self.config.image_dir}")
            image_files = _scan_images(
                self.config.image_dir,
                self.config.recursive,
                self.config.extensions,
            )
            self.total_images = len(image_files)
            logging.info(f"[BATCH VISION] Found {self.total_images} images")

            if self.total_images == 0:
                self.status = JobStatus.COMPLETED
                return

            # 2. Get and pin the VLM provider
            provider = self.router.get_provider(self.config.model_id)
            self.router.pin_model(self.config.model_id)

            try:
                # 3. Build shared prefix cache from system prompt
                logging.info("[BATCH VISION] Building shared prefix cache")
                prefix_cache = provider.build_shared_prefix(self.config.system_prompt)
                logging.info(
                    f"[BATCH VISION] Prefix cache ready: {prefix_cache.num_tokens} tokens"
                )

                # 4. Build effective request config
                effective_request = {
                    'temperature': self.config.temperature,
                    'top_p': 1.0,
                    'top_k': 0,
                    'min_p': 0.0,
                    'max_tokens': self.config.max_tokens,
                    'repetition_penalty': 1.0,
                    'presence_penalty': 0.0,
                    'cache_type': provider.config.get('cache_type', 'standard'),
                    'kv_bits': provider.config.get('kv_bits'),
                    'kv_group_size': provider.config.get('kv_group_size', 64),
                }

                # 5. Record job in DB
                self.db.store_job(self.job_id, self.config)

                # 6. Process images
                for i, image_path in enumerate(image_files):
                    if self._cancel.is_set():
                        self.status = JobStatus.CANCELLED
                        logging.info(f"[BATCH VISION] Job cancelled at image {i}/{self.total_images}")
                        break

                    str_path = str(image_path)
                    self.current_file = image_path.name

                    # Check if already processed (resume support)
                    try:
                        fhash = _file_hash(str_path)
                    except OSError as e:
                        logging.warning(f"[BATCH VISION] Cannot read {str_path}: {e}")
                        self.failed += 1
                        continue

                    if self.db.is_processed(fhash, self.config.model_id):
                        self.skipped += 1
                        continue

                    # Store file metadata
                    stat = image_path.stat()
                    self.db.store_file(
                        file_hash=fhash,
                        file_path=str_path,
                        file_name=image_path.name,
                        file_size=stat.st_size,
                        file_modified=time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(stat.st_mtime)),
                    )

                    # Process image
                    gen_start = time.time()
                    try:
                        # Acquire generation lock for thread safety with Metal
                        with provider._generation_lock:
                            raw_output = provider.process_single_image_with_prefix(
                                str_path, prefix_cache, effective_request,
                            )

                        gen_ms = int((time.time() - gen_start) * 1000)

                        # Try to extract valid JSON
                        label_json = _extract_json(raw_output)
                        if label_json is None:
                            logging.warning(
                                f"[BATCH VISION] Non-JSON output for {image_path.name}, "
                                f"storing raw text"
                            )
                            label_json = json.dumps({"raw_caption": raw_output})

                        self.db.store_label(
                            file_hash=fhash,
                            model_id=self.config.model_id,
                            label_json=label_json,
                            raw_output=raw_output,
                            tokens_used=len(raw_output.split()),  # Approximate
                            generation_time_ms=gen_ms,
                        )
                        self.completed += 1

                    except Exception as e:
                        logging.error(f"[BATCH VISION] Failed on {image_path.name}: {e}")
                        self.failed += 1

                    # Periodic memory cleanup
                    processed = self.completed + self.failed
                    if processed % CACHE_CLEAR_INTERVAL == 0:
                        mx.clear_cache()
                        gc.collect()

                    # Progress logging
                    if processed % 100 == 0 and processed > 0:
                        progress = self.get_progress()
                        logging.info(
                            f"[BATCH VISION] Progress: {progress.completed}/{progress.total_images} "
                            f"({progress.images_per_second:.1f} img/s, "
                            f"ETA: {progress.eta_seconds / 3600:.1f}h)"
                        )
                        self.db.update_job(
                            self.job_id, 'running',
                            self.completed, self.failed, self.total_images
                        )

                if self.status == JobStatus.RUNNING:
                    self.status = JobStatus.COMPLETED

            finally:
                # Always unpin, even on error
                self.router.unpin_model(self.config.model_id)
                mx.clear_cache()

        except Exception as e:
            self.status = JobStatus.FAILED
            self.error = str(e)
            logging.error(f"[BATCH VISION] Job failed: {e}", exc_info=True)

        finally:
            elapsed = time.time() - self._start_time
            self.db.update_job(
                self.job_id, self.status.value,
                self.completed, self.failed, self.total_images
            )
            logging.info(
                f"[BATCH VISION] Job {self.job_id} {self.status.value}: "
                f"{self.completed} completed, {self.failed} failed, "
                f"{self.skipped} skipped in {elapsed:.0f}s"
            )


class BatchVisionJobManager:
    """Manages running batch vision jobs. Thread-safe singleton."""

    def __init__(self):
        self._jobs: dict[str, BatchVisionLabelJob] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def submit(self, job: BatchVisionLabelJob) -> str:
        """Start a job in a background thread. Returns job_id."""
        with self._lock:
            self._jobs[job.job_id] = job
            thread = threading.Thread(
                target=job.run,
                name=f"batch-vision-{job.job_id}",
                daemon=True,
            )
            self._threads[job.job_id] = thread
            thread.start()
        return job.job_id

    def get_progress(self, job_id: str) -> JobProgress | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return job.get_progress()

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job.cancel()
            return True

    def list_jobs(self) -> list[JobProgress]:
        with self._lock:
            return [job.get_progress() for job in self._jobs.values()]


# Module-level singleton
_job_manager = BatchVisionJobManager()


def get_job_manager() -> BatchVisionJobManager:
    return _job_manager


def _reset_for_testing():
    """Reset the module-level singleton for test isolation."""
    global _job_manager
    _job_manager = BatchVisionJobManager()
