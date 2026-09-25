# src/heylook_llm/process_memory.py
"""A process's resident memory, for the llama-server child a gguf model runs in.

Resident size, not physical footprint: llama-server maps its weights from the
GGUF file, and footprint (Activity Monitor's "Memory") leaves those
file-backed pages out. Measured 2026-09-25 on a 27 GiB Q8 model: footprint
was the KV cache and buffers alone, and resident size was footprint plus the
weights file almost exactly. Read with ``proc_pid_rusage`` -- one syscall, no
subprocess, cheap enough for a metrics call (``LlamaServerProvider.get_metrics``).

``scripts/perf_ab.py`` carries its own copy of the struct on purpose (it
samples footprint, alongside other counters): its arms run older git
revisions through PYTHONPATH, which would not have this module.
"""

from __future__ import annotations

import ctypes
from typing import Optional


class _RusageInfoV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten")]


_RUSAGE_INFO_V2 = 2

try:
    _LIBPROC: Optional[ctypes.CDLL] = ctypes.CDLL("/usr/lib/libproc.dylib")
except OSError:  # not macOS
    _LIBPROC = None


def resident_mb(pid: int) -> Optional[float]:
    """The process's resident memory in MiB (mapped weights included), or
    None where it cannot be read (not macOS, the process is gone, or not ours
    to inspect)."""
    if _LIBPROC is None:
        return None
    info = _RusageInfoV2()
    if _LIBPROC.proc_pid_rusage(int(pid), _RUSAGE_INFO_V2, ctypes.byref(info)) != 0:
        return None
    return round(info.ri_resident_size / (1024 * 1024), 1)
