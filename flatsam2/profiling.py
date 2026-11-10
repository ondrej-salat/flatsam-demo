"""Lightweight section profiler for tracker timing analysis.

Disabled by default: :meth:`Profiler.section` is a near-zero-cost no-op so the
production tracking loop keeps its readability and speed. Enable it explicitly
(see ``profile_tracker.py``) to get per-section wall-clock timing of the main
building blocks — SAM3, LoMa keypoints/match/RANSAC, Hough mask-H, symmetry,
optical flow — with correct CUDA synchronization for honest GPU attribution.

Usage
-----
Instrument hot paths with the module-level singleton::

    from flatsam2.profiling import profiler
    with profiler.section("sam3"):
        seg_mask = self.segmenter.track(frame)

Sections nest via a name stack, producing a tree keyed by ``parent/child``
paths. A driver enables timing, wraps each frame, then prints a report::

    profiler.enable(sync_cuda=True)
    for frame in frames:
        with profiler.section("frame"):
            tracker.track(frame)
    print(profiler.report())
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List

import torch


@dataclass
class _SectionStat:
    """Accumulated timing for one fully-qualified section path.

    Attributes:
        path: Slash-joined section path, e.g. ``"frame/loma/keypoints"``.
        depth: Nesting depth (0 for top-level), used for report indentation.
        n_calls: Number of times the section was entered.
        total_s: Summed wall-clock seconds across all calls.
    """

    path: str
    depth: int
    n_calls: int = 0
    total_s: float = 0.0


class Profiler:
    """Hierarchical wall-clock profiler with optional CUDA synchronization.

    When :attr:`enabled` is False every :meth:`section` call is a trivial
    generator that yields immediately, so leaving instrumentation in production
    code costs essentially nothing. When enabled, section boundaries optionally
    call :func:`torch.cuda.synchronize` so the measured wall time actually
    reflects completed GPU work rather than just kernel-launch latency.
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self.sync_cuda: bool = False
        self._stats: dict[str, _SectionStat] = {}
        self._order: List[str] = []  # insertion order of paths for stable reports
        self._stack: List[str] = []  # active section names (for qualified paths)

    def enable(self, sync_cuda: bool = True) -> None:
        """Turn timing on.

        Args:
            sync_cuda: If True (and CUDA is available), synchronize the device at
                every section boundary so GPU time is attributed to the section
                that launched it. Disable to measure CPU-launch time only.
        """
        self.enabled = True
        self.sync_cuda = bool(sync_cuda and torch.cuda.is_available())

    def disable(self) -> None:
        """Turn timing off (sections become no-ops again)."""
        self.enabled = False

    def reset(self) -> None:
        """Clear all accumulated stats (e.g. after warm-up frames)."""
        self._stats.clear()
        self._order.clear()
        self._stack.clear()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Time the wrapped block under the current section stack.

        Args:
            name: Leaf name for this section; combined with the active stack into
                a qualified ``parent/child`` path.

        Yields:
            None.
        """
        if not self.enabled:
            yield
            return

        if self.sync_cuda:
            torch.cuda.synchronize()
        depth = len(self._stack)
        self._stack.append(name)
        path = "/".join(self._stack)
        # Register on enter so parents precede their children in report order.
        stat = self._stats.get(path)
        if stat is None:
            stat = _SectionStat(path=path, depth=depth)
            self._stats[path] = stat
            self._order.append(path)
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            self._stack.pop()
            stat.n_calls += 1
            stat.total_s += elapsed

    def report(self, n_frames: int = 0) -> str:
        """Format a per-section timing table.

        Percentages are relative to the summed time of all top-level
        (depth-0) sections, so wrapping each frame in one top-level section makes
        the column read as "fraction of frame".

        Args:
            n_frames: Frame count for the per-frame (ms/frame) and FPS lines;
                ``0`` omits them.

        Returns:
            Multi-line report string.
        """
        if not self._order:
            return "Profiler: no sections recorded."

        root_total = sum(
            s.total_s for s in self._stats.values() if s.depth == 0
        )
        root_total = root_total or 1e-9

        header = f"{'section':<34}{'calls':>7}{'total':>11}{'ms/call':>11}{'%root':>8}"
        lines = [header, "-" * len(header)]
        for path in self._order:
            stat = self._stats[path]
            leaf = path.rsplit("/", 1)[-1]
            label = "  " * stat.depth + leaf
            ms_per_call = 1000.0 * stat.total_s / stat.n_calls
            pct = 100.0 * stat.total_s / root_total
            lines.append(
                f"{label:<34}{stat.n_calls:>7}{stat.total_s:>10.3f}s"
                f"{ms_per_call:>11.2f}{pct:>7.1f}%"
            )

        if n_frames > 0:
            ms_per_frame = 1000.0 * root_total / n_frames
            fps = n_frames / root_total if root_total > 0 else 0.0
            lines.append("-" * len(header))
            lines.append(
                f"{'frames':<34}{n_frames:>7}{root_total:>10.3f}s"
                f"{ms_per_frame:>11.2f}{100.0:>7.1f}%"
            )
            lines.append(f"Overall: {fps:.3f} FPS ({ms_per_frame:.1f} ms/frame)")

        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            lines.append(f"Peak GPU mem: {peak_gb:.2f} GB")

        return "\n".join(lines)


# Module-level singleton shared by all instrumented modules.
profiler = Profiler()
