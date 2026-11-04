# Agent Guidelines

Read `CLAUDE.md` for project architecture and full workstation/HPC operating rules.

## Environment

- Before running commands, detect the machine with
  `command -v sbatch >/dev/null && echo HPC || echo WORKSTATION`.
- On the workstation, package dependencies are not installed. Do not run imports,
  CLIs, or tests; edit and reason only.
- On the HPC, source `.envrc` before Python commands. Keep the login node light and
  submit model, GPU, dataset-wide, or long-running work through SLURM.
- Never recursively walk or search paths under `/mnt`. Use manifest-driven status
  tools or read known paths; see `CLAUDE.md` for safe commands.
- The HPC checkout is a workstation sync target, not durable storage. Keep artifacts
  under manifest output paths and do not rely on HPC commits or tracked-file edits
  surviving the next sync.

- Imports: standard library -> third-party -> local, separated by a blank line.
- Types: always type-hint; use `Path` for file paths, `Optional[T]` for nullable, `np.ndarray` for arrays.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Docstrings: Google-style (`Args:`, `Returns:`, `Raises:`). Include array shapes when relevant.
- Errors: raise `ValueError` with descriptive f-strings; handle sequence processing with try/except and include traceback on failure.
- Formatting: ~100 char lines; two blank lines between top-level defs; align multi-line function args.
- Always keep the original formatting (no edits) on lines unrelated to the currently performed logical changes. Only do stylistic improvements when the user asks for them.
- Prefer einops (rearrange, reduce, repeat, ...) for all tensor reshaping/reordering. Avoid .reshape, .view, complex transpose chains, or manual shape arithmetic.
- Keep documentation minimal and focused. Maintain only these markdown files at repo root: `README.md` (mainly for users), `DETAILS.md` (mainly for agents)
- Docs should be high-level, non-verbose, and avoid duplicative or experimental `.md` files. Add new docs only when accepted by maintainers.
- If you need reference material outside the repo, ask the user. Do not attempt to search for it.
- long running scripts should save outputs more frequently, not just after everything is processed (allows interrupts)
- long running scripts should be able to continue computing after interrupts (not recomputing already finished items, unless asked for)

## Code structure: linear over dispatch

Prefer code that reads top-to-bottom like the algorithm (as in a paper or diagram).
One operation per line, real calls inline. You should be able to open a function and
see what executes — no chasing factories or strategy objects.

- One unified opaque API at the very top ONLY: e.g. `tracker.track()` dispatches once
  via a flat `match` on an explicit discriminator. Outside sees a single entrypoint;
  inside is plain to follow.
- Make the active algorithm explicit and greppable: select it from an
  `algorithm: Literal[...]` discriminator the top dispatch matches on, NOT inferred from
  a combination of boolean flags. Goal: open a config, read the algorithm name, grep that
  string literal, land on the one function. Survives dynamic config loading.
- Below the top dispatch: NO factories, strategy objects, registries, or
  `build_x()` / `x.dispatch()` indirection. Inline the real calls.
- Two helper kinds. ALLOWED: a helper that NAMES one operation — a verb
  (`mask_to_homography`, `estimate_loma_H`, `resolve_symmetry`, `build_debug`).
  BANNED: a helper that only ROUTES control flow
  (`_finalize_success`, `_track_without_nn`, `_choose_x`, `build_estimator`). Inline routers.
- No recursive / self-calling trackers; no class hierarchy whose base methods you must
  chase to learn behavior. Shared state lives in one thin container; each algorithm is a
  free function taking it (e.g. `algorithms/<name>.py: def track(trk, frame) -> TrackResult`).
- Each distinct algorithm = one linear function in its own module. Parameter-only variants
  share a function (driven by config values), not new code.
- Accept small duplication of orchestration lines to keep each algorithm readable in
  isolation. Share only genuine primitives (the verbs), never extract a shared router to
  de-duplicate. Cross-cutting concerns (logging, timing, debug assembly) stay inline and
  visible; if a block grows heavy, promote it to a named verb, not a wrapper that hides flow.

## Pipeline Manifest

`experiments/pipeline.yml` is the authoritative map of experiment pipelines:
runner scripts, SLURM submit commands, output paths, completeness signals, and
active configs. Update it whenever you:

- Add or rename an experiment script or `slurm/submit_*.py`
- Add a config under `configs/tracker/` or `configs/loma/` intended for active use
- Change an output path or completeness signal in a runner script
- Add a new pipeline that writes results to HPC storage

Schema reference is in `~/.claude/skills/hpc-status/SKILL.md` (pipeline manifest
schema section). Consumed by `scripts/hpc_status.py` and the `/hpc-status` skill.

Respond like smart caveman. Cut all filler, keep technical substance.
- Drop articles (a, an, the), filler (just, really, basically, actually).
- Drop pleasantries (sure, certainly, happy to).
- No hedging. Fragments fine. Short synonyms.
- Technical terms stay exact. Code blocks unchanged.
- Pattern: [thing] [action] [reason]. [next step].
