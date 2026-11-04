"""Line-profiler runner for synthetic data generation.

Usage
-----
# Profile and collect line timings:
    kernprof -l profile_datagen.py

# Print the report:
    python -m line_profiler profile_datagen.py.lprof

# Quick wall-clock check without profiling (plain python):
    python profile_datagen.py

Options (edit the constants below or pass as env vars):
    N_SAMPLES      — how many successful samples to generate  (default 50)
    ARCHITECTURE   — "option_e" | "option_c" | "both"        (default "option_e")
    SDF_DOWNSAMPLE — 1 (full) / 4 (16× faster)              (default 1, so you
                     can see the raw cost before the fix)
"""

from __future__ import annotations

import os
import time

import numpy as np

N_SAMPLES = int(os.environ.get("N_SAMPLES", 50))
ARCHITECTURE = os.environ.get("ARCHITECTURE", "option_e")
# SDF_DOWNSAMPLE only matters for architecture != "option_e" (full SDF path)
SDF_DOWNSAMPLE = int(os.environ.get("SDF_DOWNSAMPLE", 1))


def main() -> None:
    from flatsam2.nn.data_gen import generate_sample

    rng = np.random.default_rng(42)
    generated = 0
    attempts = 0

    print(f"Generating {N_SAMPLES} samples  "
          f"arch={ARCHITECTURE}  sdf_downsample={SDF_DOWNSAMPLE}")

    t0 = time.perf_counter()

    while generated < N_SAMPLES:
        sample = generate_sample(
            rng,
            architecture=ARCHITECTURE,
            sdf_downsample=SDF_DOWNSAMPLE,
        )
        attempts += 1
        if sample is not None:
            generated += 1

    elapsed = time.perf_counter() - t0
    success_rate = generated / attempts
    per_sample = elapsed / generated

    print(f"\nDone in {elapsed:.2f}s")
    print(f"  {generated} samples / {attempts} attempts  "
          f"(success rate {success_rate:.1%})")
    print(f"  {per_sample * 1000:.1f} ms per successful sample")
    print(f"  → ~{3600 / per_sample / 128:.0f} batches/hour  "
          f"(batch_size=128, single worker)")


if __name__ == "__main__":
    main()
