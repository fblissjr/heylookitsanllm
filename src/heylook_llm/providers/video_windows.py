"""Windowed video embeddings on MLX: one vector per window of a long video.

heylook-owned runtime over mlx-vlm (VISION principle 6). The model trained on
at most ``max_frames`` frames at ``fps`` (EmbeddingGemma 2: 32 at 1 fps, report
arXiv 2610.37364 §2.2); mlx-vlm's own sampler spreads 32 frames over the whole
clip, so a long video's brief moments fall between frames. Windows keep every
vector inside the trained regime. Inside a window, ``content`` sampling decodes
candidates at ``CONTENT_CANDIDATE_FPS`` and keeps half the budget as evenly
spaced anchors plus the frames farthest from the anchors' per-pixel median;
``fps`` takes frames at the model's own rate. Evidence for the picker and its
decode cost: internal/claude/embeddinggemma2_probes/video_events/ (results_v3).

The decoder hook (``video_processor._decode_video(path, sampler)``) and the
pre-decoded frames path (``do_sample_frames=False`` with ``video_metadata``)
are mlx-vlm internals; tests/contract/test_mlxvlm_surface.py pins them.
"""
import copy
import logging

import numpy as np

CONTENT_CANDIDATE_FPS = 4.0   # candidate rate for content sampling (results_v3: picks 0.5 s events)
SUPPRESSION_RADIUS = 1        # candidates around a chosen frame that cannot also be chosen
THUMB_STRIDE = 8              # pixel stride for the frame comparison (colour kept)


def pick_fps(times, lo, hi, fps, max_frames):
    """Positions in ``times`` nearest each 1/fps step from ``lo``."""
    steps = np.arange(lo, hi, 1.0 / fps)[:max_frames]
    return sorted({int(np.argmin(np.abs(times - t))) for t in steps})


def pick_content(frames, times, lo, hi, max_frames, fps=1.0):
    """Anchors plus the frames least like the anchors' median, within [lo, hi)."""
    idx = np.where((times >= lo) & (times < hi))[0]
    budget = min(max_frames, max(2, int(np.ceil((hi - lo) * fps))))
    if len(idx) <= budget:
        return [int(i) for i in idx]
    small = frames[idx][:, ::THUMB_STRIDE, ::THUMB_STRIDE].astype(np.float32)
    anchors = np.linspace(0, len(idx) - 1, max(1, budget // 2)).round().astype(int)
    background = np.median(small[anchors] if len(anchors) >= 3 else small, axis=0)
    novelty = np.abs(small - background).mean(axis=(1, 2, 3))
    chosen, blocked = set(anchors.tolist()), set()
    for j in np.argsort(-novelty):
        if len(chosen) >= budget:
            break
        if j in chosen or j in blocked:
            continue
        chosen.add(int(j))
        blocked.update(range(j - SUPPRESSION_RADIUS, j + SUPPRESSION_RADIUS + 1))
    return sorted(int(idx[j]) for j in chosen)


def plan(times, duration, seconds, sampling, fps, max_frames, frames=None):
    """[(start, end, positions)] for consecutive windows; empty windows are skipped."""
    out = []
    for lo in np.arange(0.0, duration, seconds):
        hi = min(lo + seconds, duration)
        sel = (pick_content(frames, times, lo, hi, max_frames, fps=fps) if sampling == "content"
               else pick_fps(times, lo, hi, fps, max_frames))
        sel = [s for s in sel if lo <= times[s] < hi] or sel
        if sel:
            out.append((float(lo), float(hi), sel))
    return out


def embed_windows(provider, prompt, part, check_cancel, too_long, forward):
    """Window vectors for one video part. Returns (windows, tokens)."""
    proc = provider.processor
    vp = proc.video_processor
    fps, max_frames = float(vp.fps), int(vp.max_frames)
    options = part["windows"]
    seconds = options.get("seconds") or max_frames / fps
    sampling = options.get("sampling") or "fps"
    rate = CONTENT_CANDIDATE_FPS if sampling == "content" else fps

    def decode(indices):
        return vp._decode_video(part["path"], lambda metadata, **_: np.asarray(indices))

    # Metadata first (one frame), then one window's frames at a time: memory is
    # bounded by a window's candidates, never by the length of the video.
    _, metadata = decode([0])
    step = max(1, int(round(metadata.fps / rate)))
    candidates = np.arange(0, metadata.total_num_frames, step)
    candidate_times = candidates / metadata.fps
    messages = ([{"role": "system", "content": prompt}] if prompt else []) + [
        {"role": "user", "content": [{"type": "video"}]}]
    text = proc.apply_chat_template(messages, tokenize=False)      # the template's own rendering
    windows, tokens = [], 0
    for lo in np.arange(0.0, metadata.duration, seconds):
        check_cancel()
        hi = min(lo + seconds, metadata.duration)
        in_window = candidates[(candidate_times >= lo) & (candidate_times < hi)]
        if not len(in_window):
            continue
        times = in_window / metadata.fps
        if sampling == "content":
            frames, md = decode(in_window)
            in_window = in_window[:len(frames)]
            times = in_window / metadata.fps
            sel = pick_content(frames, times, lo, hi, max_frames, fps=fps)
            frames = frames[sel]
        else:
            sel = pick_fps(times, lo, hi, fps, max_frames)
            frames, md = decode(in_window[sel])
            sel = sel[:len(frames)]
        md = copy.copy(md)
        md.frames_indices = in_window[sel]
        inputs = proc(text=[text], videos=[frames], video_metadata=[md],
                      do_sample_frames=False, return_tensors="mlx")
        length = int(inputs["input_ids"].shape[-1])
        if length > provider.context_length:
            raise too_long(length)
        windows.append({"start": round(float(lo), 3), "end": round(float(hi), 3),
                        "frame_times": [round(float(t), 3) for t in times[sel]],
                        "embedding": forward(dict(inputs))})
        tokens += length
    logging.debug(
        "video_windows: duration=%.2fs sampling=%s windows=%d tokens=%d",
        metadata.duration, sampling, len(windows), tokens,
    )
    return windows, tokens
