# Handoff — watch the `corner_fusion_register_final` grids

Self-contained task. Repo `~/git/flatsam2` on the HPC; `source .envrc` first.
Submit anything new with `--no-mail`.

## The job

Four SLURM jobs are queued. Wait for them, then report the result.

| job | what |
|---|---|
| 11344539 | PTi array (300 seqs) — **already ~180/300 done**, resumes; completed sequences are skipped |
| 11344540 | PTi eval, depends on the above |
| 11344541 | POT array (210 seqs) |
| 11344542 | POT eval |

Watch **by job id** — `squeue` truncates the NAME column, so matching on the
config name silently finds nothing:

```bash
squeue -j 11344539,11344540,11344541,11344542 -h -o "%.14i %T %R"
```

Done when both of these exist:

```
/mnt/personal/serycjon/flatsam2/results/corner_fusion_register_final/PlanarTrackInitfix/eval_results.json
/mnt/personal/serycjon/flatsam2/results/corner_fusion_register_final/POT-210/eval_results.json
```

If the arrays finish but an eval did not run, run it by hand:

```bash
python evaluate.py pti --results-dir /mnt/personal/serycjon/flatsam2/results \
    --config configs/tracker/corner_fusion_register_final.py --split test
python evaluate.py pot --results-dir /mnt/personal/serycjon/flatsam2/results \
    --config configs/tracker/corner_fusion_register_final.py
```

## What to report

Read `aggregate` from each `eval_results.json` and compare against the two
reference configs already on disk in the same results tree:

| config | PTi P@5 | PTi P@15 | PTi br-gap | POT P@5 | POT P@15 |
|---|---|---|---|---|---|
| `corner_fusion_register` | 67.94 | 86.52 | 2.83 | 92.07 | 96.77 |
| `corner_fusion_register_kp` | 68.63 | 87.32 | 2.25 | 91.70 | 96.65 |
| `corner_fusion_register_final` | ? | ? | ? | ? | ? |

Use `mean_p@5` / `mean_p@15` (per-sequence means, not the `overall_*` pooled
numbers). **br-gap** = `mean_br_p@15 − mean_p@15`; also report the share of
sequences whose per-sequence gap exceeds 5 pp, which is the convention used
throughout this thread.

## How to read it

`corner_fusion_register_final` is `register_kp` plus a cropped signed-distance
field (`register_sdf_roi_margin=1.0`) and the corner-alias shortcut with an
early exit (`register_alias_shortcut`, `register_sweep_stop_score=0.70`).

None of that changes what the tracker decides on a healthy frame, so **expect
parity with `register_kp`** — PTi P@15 ≈ 87.3–88.4, POT ≈ 96.7. A gain, if any,
comes from the search on lost frames. A *loss* beyond ~0.5 pp is a red flag and
means one of the three knobs misbehaves at grid scale; say so plainly rather
than averaging it away.

## When done

Append a short entry to `docs/journal.md` (iso-date headline, a sentence or two,
per the repo convention), and update the "In flight right now" table in
`docs/HANDOFF-final-candidate.md` — that file also holds the traps and the
open-threads list for this thread. Then commit; the user backsyncs the HPC repo
home with `make backsync`, so committing here is fine and expected.

Do not start new experiments or branch the research — the point of this run is a
clean reference version before any further ablations.
