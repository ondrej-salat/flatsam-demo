"""Build ``docs/tracker-cards.html`` — the standalone design record of the tracker.

Nothing in the page is transcribed by hand. Numbers come from:

* ``<config>/<dataset>/eval_results.json`` — the full-grid benchmark metrics,
* ``figs_facts.json`` — written by ``experiments/register_visuals.py`` while it
  runs the deployed code to produce the figures,
* ``complementarity.json`` — written by ``experiments/corner_complementarity.py``,
* the offline benchmark CSVs written by ``experiments/maskh_bench.py``.

Figures are inlined as data URIs, so the page is standalone.

Regenerate::

    python slurm/submit_cmd.py --gpus 0 --no-mail -- python experiments/register_visuals.py \\
        --dataset pti --sequence Seq_00006 --frame 30 --fusion-frame 30 --out-dir <figdir>
    python experiments/corner_complementarity.py --dataset pti
    python experiments/build_tracker_cards.py --fig-dir <figdir>
"""

import argparse
import base64
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from flatsam2.path_config import paths

NAMES: Dict[str, str] = {
    "WOFTSAM2": "woftsam_samfallback_nl",
    "SAM2H": "sam_H",
    "LMr": "loma_r_allobj_inflate10",
    "SAM3H": "hough_symmetry_only",
    "SWITCH35": "loma_samh_decoupled_rerun",
    "ORACLE": "oracle_LMr_allobj_infl10_X_SAM3H",
    "ALIGN-PURE": "loma_samh_decoupled_alignh_pure",
    "FUSE-ALIGN": "corner_fusion_alignh_calibrated",
    "FUSE-REDETECT": "corner_fusion_alignh_redetect",
    "FUSE-REG": "corner_fusion_register",
    "FUSE-REG-KP": "corner_fusion_register_kp",
    "FUSE-REG-NOAPP": "corner_fusion_register_noapp",
    "FINAL": "corner_fusion_register_final",
    "FINAL+RECHECK": "corner_fusion_register_recheck",
    "WHOLEFRAME": "corner_fusion_wholeframe",
    "SCORE-W11": "corner_fusion_simple_norecheck",
}

DATASETS = [("POT-210", "POT"), ("PlanarTrackInitfix", "PTi"), ("POIC", "POIC"),
            ("TMT", "TMT"), ("UCSB", "UCSB"), ("MPOT-3K", "MPOT")]

REGIMES = ["good", "drift", "lost", "cold"]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_metrics(results_dir: Path) -> Dict[str, Dict[str, dict]]:
    """Pooled benchmark metrics per config and dataset."""
    out: Dict[str, Dict[str, dict]] = {}
    for short, stem in NAMES.items():
        row: Dict[str, dict] = {}
        for label, ds_short in DATASETS:
            path = results_dir / stem / label / "eval_results.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            agg, seqs = payload["aggregate"], payload["sequences"]
            gaps = [s["br_p@15"] - s["p@15"] for s in seqs if "br_p@15" in s]
            row[ds_short] = {
                "p5": agg["overall_p@5"], "p15": agg["overall_p@15"],
                "br15": agg.get("overall_br_p@15"),
                "gap": agg.get("overall_br_p@15", 0.0) - agg["overall_p@15"],
                "n": agg["num_sequences"], "frames": agg["total_frames"],
                "gap5": 100.0 * sum(1 for g in gaps if g > 5.0) / max(len(gaps), 1),
            }
        out[short] = row
    return out


def load_bench(csv_path: Path, dataset: str = "pti") -> Dict[str, Dict[str, dict]]:
    """Per-method, per-regime P@15 / P@5 / p90 error from a maskh_bench CSV.

    Args:
        csv_path: CSV written by ``maskh_bench.py run``.
        dataset: Dataset key to keep.

    Returns:
        ``{method: {regime: {"p5", "p15", "p90"}}}``.
    """
    rows = list(csv.DictReader(csv_path.open()))
    acc: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        if r["dataset"] != dataset:
            continue
        acc.setdefault(r["method"], {}).setdefault(r["regime"], []).append(float(r["err"]))
    out: Dict[str, Dict[str, dict]] = {}
    for method, regimes in acc.items():
        out[method] = {}
        for regime, errs in regimes.items():
            errs_sorted = sorted(errs)
            out[method][regime] = {
                "p5": 100.0 * sum(e <= 5 for e in errs) / len(errs),
                "p15": 100.0 * sum(e <= 15 for e in errs) / len(errs),
                "p90": errs_sorted[int(0.9 * (len(errs_sorted) - 1))],
                "n": len(errs),
            }
    return out


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def scope_table(scope: Dict[str, dict]) -> str:
    """Pooled P@15 per fusion rule, POT and PTi side by side.

    Args:
        scope: ``{dataset label: fusion_rule_replay --out-json payload}``.

    Returns:
        HTML table body of the replay comparison.
    """
    rows = [
        ("per-corner decision (before)", "baseline"),
        ("<b>whole-frame decision</b>", "whole_frame"),
        ("whole-frame + covariance fallback", "whole_frame+covfb10"),
        ("keypoint branch alone", "loma_only"),
        ("mask branch alone", "samh_only"),
        ("oracle: better whole branch", "oracle"),
    ]
    datasets = list(scope)
    head = "".join(f"<th>{d} pooled P@15</th>" for d in datasets)
    body = ""
    for label, key in rows:
        cells = "".join(
            f"<td>{fmt(scope[d]['rules'][key]['pooled_p15'])}</td>" for d in datasets
        )
        body += f'<tr><th class="rowname">{label}</th>{cells}</tr>'
    frames = "".join(
        f"<td>{scope[d]['total_gt_frames']:,} frames</td>" for d in datasets
    )
    return (
        f'<div class="tablewrap"><table><thead><tr><th class="rowname"></th>{head}</tr>'
        f'<tr><th class="rowname"></th>{frames}</tr></thead><tbody>{body}</tbody></table></div>'
    )


def _scope_gain(scope: Dict[str, dict], dataset: str) -> tuple:
    """Whole-frame gain over the per-corner rule, and its share of the oracle gap.

    Args:
        scope: ``{dataset label: replay payload}``.
        dataset: Dataset label to read.

    Returns:
        Tuple ``(gain_pp, percent_of_oracle_headroom)``.
    """
    rules = scope[dataset]["rules"]
    base = rules["baseline"]["pooled_p15"]
    gain = rules["whole_frame"]["pooled_p15"] - base
    headroom = rules["oracle"]["pooled_p15"] - base
    return gain, (100.0 * gain / headroom if headroom else float("nan"))


def fmt(v: Optional[float], d: int = 2) -> str:
    return "—" if v is None else f"{v:.{d}f}"


def delta(a: Optional[float], b: Optional[float]) -> str:
    return "—" if (a is None or b is None) else f"{a - b:+.2f}"


def img_tag(fig_dir: Path, name: str, alt: str) -> str:
    path = fig_dir / name
    if not path.exists():
        return f'<p class="missing">missing figure: {name}</p>'
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


def metric_table(m, rows, datasets, baseline=None) -> str:
    head = "".join(f"<th>{d} P@5</th><th>{d} P@15</th>" for d in datasets)
    body = []
    for short in rows:
        if short not in m or not m[short]:
            continue
        cells = [f'<th class="rowname">{short}</th>']
        for d in datasets:
            e = m[short].get(d)
            base = m[baseline].get(d) if baseline else None
            p5 = e["p5"] if e else None
            p15 = e["p15"] if e else None
            d5 = d15 = ""
            if baseline and short != baseline and e and base:
                d5 = f'<span class="d">{delta(p5, base["p5"])}</span>'
                d15 = f'<span class="d">{delta(p15, base["p15"])}</span>'
            cells.append(f"<td>{fmt(p5)}{d5}</td><td>{fmt(p15)}{d15}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="tablewrap"><table><thead><tr><th></th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def gap_table(m, rows, dataset) -> str:
    body = []
    for short in rows:
        e = m.get(short, {}).get(dataset)
        if not e:
            continue
        body.append(f'<tr><th class="rowname">{short}</th><td>{fmt(e["p15"])}</td>'
                    f'<td>{fmt(e["br15"])}</td><td>{fmt(e["gap"])}</td>'
                    f'<td>{fmt(e["gap5"], 1)}%</td></tr>')
    return ('<div class="tablewrap"><table><thead><tr><th></th>'
            f'<th>{dataset} P@15</th><th>{dataset} br-P@15</th><th>gap (pp)</th>'
            '<th>sequences with gap &gt; 5 pp</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def bench_table(bench: Dict[str, Dict[str, dict]], methods: List[tuple],
                metric: str = "p15", extra_p90: bool = False) -> str:
    """Table of one bench metric per regime, one row per method."""
    head = "".join(f"<th>{r}</th>" for r in REGIMES)
    if extra_p90:
        head += "".join(f"<th>p90 {r}</th>" for r in ("lost", "cold"))
    body = []
    for label, key in methods:
        if key not in bench:
            continue
        cells = [f'<th class="rowname">{label}</th>']
        cells += [f'<td>{fmt(bench[key][r][metric], 1)}</td>' for r in REGIMES]
        if extra_p90:
            cells += [f'<td>{fmt(bench[key][r]["p90"], 0)}</td>' for r in ("lost", "cold")]
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="tablewrap"><table><thead><tr><th></th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


# ---------------------------------------------------------------------------
# MathML — real math, no external renderer (the artifact CSP blocks CDNs)
# ---------------------------------------------------------------------------

def _mi(x, cls=""):
    c = f' class="{cls}"' if cls else ""
    return f"<mi{c}>{x}</mi>"


M = {}

M["error"] = """<math display="block"><mrow>
<msub><mi>e</mi><mi>t</mi></msub><mo>=</mo>
<msqrt><mrow><mfrac><mn>1</mn><mn>4</mn></mfrac>
<munderover><mo>&#8721;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mn>4</mn></munderover>
<msup><mrow><mo>&#8741;</mo><msub><mover><mi>c</mi><mo>&#94;</mo></mover><mi>k</mi></msub>
<mo>&#8722;</mo><msubsup><mi>c</mi><mi>k</mi><mtext>GT</mtext></msubsup><mo>&#8741;</mo></mrow><mn>2</mn></msup>
</mrow></msqrt></mrow></math>"""

M["patk"] = """<math display="block"><mrow>
<mtext>P@</mtext><mi>&#964;</mi><mo>=</mo>
<mfrac><mrow><mn>100</mn></mrow><mrow><mo>|</mo><mi>V</mi><mo>|</mo></mrow></mfrac>
<mrow><mo>|</mo><mo>{</mo><mi>t</mi><mo>&#8712;</mo><mi>V</mi><mo>:</mo>
<msub><mi>e</mi><mi>t</mi></msub><mo>&#8804;</mo><mi>&#964;</mi><mo>}</mo><mo>|</mo></mrow>
</mrow></math>"""

M["residual"] = """<math display="block"><mrow>
<msub><mi>r</mi><mi>i</mi></msub><mo>(</mo><mi>&#952;</mi><mo>)</mo><mo>=</mo>
<msub><mi>D</mi><mi>t</mi></msub><mo>(</mo>
<mi>H</mi><mo>(</mo><mi>C</mi><mo>)</mo><mo>&#8901;</mo><msub><mi>b</mi><mi>i</mi></msub>
<mo>)</mo><mo>,</mo><mspace width="1.2em"/>
<mi>i</mi><mo>=</mo><mn>1</mn><mo>&#8230;</mo><mi>N</mi>
</mrow></math>"""

M["objective"] = """<math display="block"><mrow>
<mi>E</mi><mo>(</mo><mi>&#952;</mi><mo>)</mo><mo>=</mo>
<munder><mo>&#8721;</mo><mi>i</mi></munder>
<msub><mi>&#961;</mi><mi>&#948;</mi></msub><mo>(</mo><msub><mi>r</mi><mi>i</mi></msub>
<mo>(</mo><mi>&#952;</mi><mo>)</mo><mo>)</mo><mo>,</mo><mspace width="1.4em"/>
<msub><mi>&#961;</mi><mi>&#948;</mi></msub><mo>(</mo><mi>r</mi><mo>)</mo><mo>=</mo>
<mrow><mo>{</mo><mtable columnalign="left left">
<mtr><mtd><mfrac><msup><mi>r</mi><mn>2</mn></msup><mn>2</mn></mfrac></mtd>
<mtd><mrow><mo>|</mo><mi>r</mi><mo>|</mo><mo>&#8804;</mo><mi>&#948;</mi></mrow></mtd></mtr>
<mtr><mtd><mrow><mi>&#948;</mi><mo>(</mo><mo>|</mo><mi>r</mi><mo>|</mo><mo>&#8722;</mo>
<mi>&#948;</mi><mo>/</mo><mn>2</mn><mo>)</mo></mrow></mtd>
<mtd><mrow><mo>|</mo><mi>r</mi><mo>|</mo><mo>&gt;</mo><mi>&#948;</mi></mrow></mtd></mtr>
</mtable></mrow></mrow></math>"""

M["normal"] = """<math display="block"><mrow>
<mo>(</mo><msup><mi>J</mi><mo>&#8868;</mo></msup><mi>W</mi><mi>J</mi>
<mo>+</mo><mi>&#955;</mi><mi>I</mi><mo>)</mo>
<mi>&#916;</mi><mi>&#952;</mi><mo>=</mo><mo>&#8722;</mo>
<msup><mi>J</mi><mo>&#8868;</mo></msup><mi>W</mi><mi>r</mi><mo>,</mo>
<mspace width="1.4em"/>
<msub><mi>W</mi><mrow><mi>i</mi><mi>i</mi></mrow></msub><mo>=</mo>
<mrow><mo>{</mo><mtable columnalign="left left">
<mtr><mtd><mn>1</mn></mtd><mtd><mrow><mo>|</mo><msub><mi>r</mi><mi>i</mi></msub><mo>|</mo>
<mo>&#8804;</mo><mi>&#948;</mi></mrow></mtd></mtr>
<mtr><mtd><mrow><mi>&#948;</mi><mo>/</mo><mo>|</mo><msub><mi>r</mi><mi>i</mi></msub><mo>|</mo></mrow></mtd>
<mtd><mrow><mo>|</mo><msub><mi>r</mi><mi>i</mi></msub><mo>|</mo><mo>&gt;</mo><mi>&#948;</mi></mrow></mtd></mtr>
</mtable></mrow></mrow></math>"""

M["cov"] = """<math display="block"><mrow>
<msub><mi mathvariant="normal">&#931;</mi><mi>&#952;</mi></msub><mo>=</mo>
<msup><mover><mi>&#963;</mi><mo>&#94;</mo></mover><mn>2</mn></msup>
<msup><mrow><mo>(</mo><msup><mi>J</mi><mo>&#8868;</mo></msup><mi>W</mi><mi>J</mi><mo>)</mo></mrow>
<mrow><mo>&#8722;</mo><mn>1</mn></mrow></msup><mo>,</mo><mspace width="1.4em"/>
<msup><mover><mi>&#963;</mi><mo>&#94;</mo></mover><mn>2</mn></msup><mo>=</mo>
<mfrac><mrow><munder><mo>&#8721;</mo><mi>i</mi></munder><msub><mi>w</mi><mi>i</mi></msub>
<msup><msub><mi>r</mi><mi>i</mi></msub><mn>2</mn></msup></mrow>
<mrow><mi>N</mi><mo>&#8722;</mo><mn>8</mn></mrow></mfrac>
</mrow></math>"""

M["implicit"] = """<math display="block"><mrow>
<msup><mi>J</mi><mo>&#8868;</mo></msup><mi>W</mi><mi>r</mi><mo>(</mo>
<mover><mi>&#952;</mi><mo>&#94;</mo></mover><mo>,</mo><mi>y</mi><mo>)</mo>
<mo>=</mo><mn>0</mn>
<mspace width="2.2em"/><mo>&#8658;</mo><mspace width="1.2em"/>
<msup><mi>J</mi><mo>&#8868;</mo></msup><mi>W</mi><mi>J</mi>
<mspace width="0.15em"/><mi>&#948;</mi><mi>&#952;</mi>
<mo>&#8722;</mo>
<msup><mi>J</mi><mo>&#8868;</mo></msup><mi>W</mi>
<mspace width="0.15em"/><mi>&#948;</mi><mi>y</mi>
<mo>=</mo><mn>0</mn>
</mrow></math>
<math display="block"><mrow>
<mfrac><mrow><mo>&#8706;</mo><mover><mi>&#952;</mi><mo>&#94;</mo></mover></mrow>
<mrow><mo>&#8706;</mo><mi>y</mi></mrow></mfrac>
<mo>=</mo>
<msup><mrow><mo>(</mo><msup><mi>J</mi><mo>&#8868;</mo></msup><mi>W</mi><mi>J</mi><mo>)</mo></mrow>
<mrow><mo>&#8722;</mo><mn>1</mn></mrow></msup>
<msup><mi>J</mi><mo>&#8868;</mo></msup><mi>W</mi>
</mrow></math>"""

M["neff"] = """<math display="block"><mrow>
<msub><mi>N</mi><mtext>eff</mtext></msub><mo>=</mo>
<mfrac><mi>N</mi><msub><mi>&#964;</mi><mtext>int</mtext></msub></mfrac><mo>,</mo>
<mspace width="1.4em"/>
<msub><mi>&#964;</mi><mtext>int</mtext></msub><mo>=</mo><mn>1</mn><mo>+</mo><mn>2</mn>
<munder><mo>&#8721;</mo><mrow><mi>k</mi><mo>&gt;</mo><mn>0</mn></mrow></munder>
<msub><mi>&#961;</mi><mi>k</mi></msub><mo>,</mo><mspace width="1.4em"/>
<msub><mi>&#961;</mi><mi>k</mi></msub><mo>=</mo><mtext>corr</mtext><mo>(</mo>
<msub><mi>r</mi><mi>i</mi></msub><mo>,</mo><msub><mi>r</mi><mrow><mi>i</mi><mo>+</mo><mi>k</mi></mrow></msub><mo>)</mo>
</mrow></math>"""

M["whiten"] = """<math display="block"><mrow>
<mi>A</mi><mo>(</mo><mi>&#966;</mi><mo>)</mo><mo>=</mo>
<msubsup><mi>W</mi><mtext>mask</mtext><mrow><mo>&#8722;</mo><mn>1</mn></mrow></msubsup>
<mspace width="0.2em"/><mi>R</mi><mo>(</mo><mi>&#966;</mi><mo>)</mo><mspace width="0.2em"/>
<msub><mi>W</mi><mtext>tpl</mtext></msub><mo>,</mo><mspace width="1.2em"/>
<msub><mi>W</mi><mo>&#8226;</mo></msub><mo>=</mo>
<msubsup><mi mathvariant="normal">&#931;</mi><mo>&#8226;</mo>
<mrow><mo>&#8722;</mo><mn>1</mn><mo>/</mo><mn>2</mn></mrow></msubsup>
</mrow></math>"""

M["score"] = """<math display="block"><mrow>
<mi>log</mi><mspace width="0.2em"/><mi>s</mi><mo>=</mo>
<mi>a</mi><mspace width="0.2em"/><mi>log</mi><mspace width="0.2em"/><mtext>IoU</mtext>
<mo>+</mo><mi>b</mi><mspace width="0.2em"/><mi>log</mi><mspace width="0.2em"/>
<mfrac><mrow><mn>1</mn><mo>+</mo><mtext>NCC</mtext></mrow><mn>2</mn></mfrac>
</mrow></math>"""

M["maha"] = """<math display="block"><mrow>
<msubsup><mi>d</mi><mi>k</mi><mn>2</mn></msubsup><mo>=</mo>
<msup><mrow><mo>(</mo><msub><mi>a</mi><mi>k</mi></msub><mo>&#8722;</mo>
<msub><mi>b</mi><mi>k</mi></msub><mo>)</mo></mrow><mo>&#8868;</mo></msup>
<msup><mrow><mo>(</mo><msub><mi mathvariant="normal">&#931;</mi><mrow><mi>a</mi><mo>,</mo><mi>k</mi></mrow></msub>
<mo>+</mo><msub><mi mathvariant="normal">&#931;</mi><mrow><mi>b</mi><mo>,</mo><mi>k</mi></mrow></msub>
<mo>)</mo></mrow><mrow><mo>&#8722;</mo><mn>1</mn></mrow></msup>
<mo>(</mo><msub><mi>a</mi><mi>k</mi></msub><mo>&#8722;</mo><msub><mi>b</mi><mi>k</mi></msub><mo>)</mo>
</mrow></math>"""

M["fuse"] = """<math display="block"><mrow>
<msub><mover><mi>c</mi><mo>&#94;</mo></mover><mi>k</mi></msub><mo>=</mo>
<msup><mrow><mo>(</mo><msubsup><mi mathvariant="normal">&#931;</mi><mrow><mi>a</mi><mo>,</mo><mi>k</mi></mrow><mrow><mo>&#8722;</mo><mn>1</mn></mrow></msubsup>
<mo>+</mo><msubsup><mi mathvariant="normal">&#931;</mi><mrow><mi>b</mi><mo>,</mo><mi>k</mi></mrow><mrow><mo>&#8722;</mo><mn>1</mn></mrow></msubsup><mo>)</mo></mrow>
<mrow><mo>&#8722;</mo><mn>1</mn></mrow></msup>
<mo>(</mo><msubsup><mi mathvariant="normal">&#931;</mi><mrow><mi>a</mi><mo>,</mo><mi>k</mi></mrow><mrow><mo>&#8722;</mo><mn>1</mn></mrow></msubsup>
<msub><mi>a</mi><mi>k</mi></msub><mo>+</mo>
<msubsup><mi mathvariant="normal">&#931;</mi><mrow><mi>b</mi><mo>,</mo><mi>k</mi></mrow><mrow><mo>&#8722;</mo><mn>1</mn></mrow></msubsup>
<msub><mi>b</mi><mi>k</mi></msub><mo>)</mo>
</mrow></math>"""


def build_page(m, facts, comp, bench, scope, fig_dir, fig_dirs) -> str:
    """Assemble the page from measurements, figures and the template."""
    cfg, case = facts["config"], facts["case"]
    seq, frame = case["seq"], case["frame_idx"]
    ver, fus, gn, cov = facts["verify"]["rows"], facts["fusion"], facts["gn"], facts["covariance"]
    jac = facts["jacobian"]

    pti = lambda s, k: m[s]["PTi"][k]        # noqa: E731
    pot = lambda s, k: m[s]["POT"][k]        # noqa: E731

    def bench_cell(method, regime, key="p15"):
        return fmt(bench[method][regime][key], 1)

    win = fig_dirs["win"]
    miss = fig_dirs["miss"]
    win_facts = json.loads((win / "figs_facts.json").read_text())["fusion"]
    miss_facts = json.loads((miss / "figs_facts.json").read_text())["fusion"]
    wd = win_facts["gt_corner_dist"]
    md = miss_facts["gt_corner_dist"]

    return TEMPLATE.format(
        M=M,
        name_rows="".join(f'<tr><td class="mono">{s}</td><td class="mono dim">{c}</td></tr>'
                          for s, c in NAMES.items()),
        seq=seq, frame=frame,
        n_angles=cfg["n_angles"], n_refine=cfg["n_refine"], n_full=cfg["n_full"],
        accept=cfg["accept_score"], margin=cfg["incumbent_margin"], huber=cfg["huber_delta_px"],
        nb=cfg["n_boundary_points"], max_iters=cfg["max_iters"],
        a_pow=cfg["overlap_power"], b_pow=cfg["appearance_power"],
        stop=cfg["sweep_stop_score"], roi=cfg["sdf_roi_margin"],
        n_boundary=facts["sdf"]["n_boundary"], med_r=f'{facts["sdf"]["median_abs_r"]:.1f}',
        perturb=f'{facts["perturb_px"]:.0f}',
        gn_iters=gn["n_iters"], gn_c0=f'{gn["cost0"]:.3g}', gn_c1=f'{gn["cost1"]:.3g}',
        n_prop=facts["propose"]["n_proposals"], best_iou=f'{facts["propose"]["best_iou"]:.3f}',
        v0_iou=f'{ver[0]["iou"]:.3f}', v1_iou=f'{ver[1]["iou"]:.3f}',
        v0_ncc=f'{ver[0]["ncc"]:+.3f}', v1_ncc=f'{ver[1]["ncc"]:+.3f}',
        v2_ncc=f'{ver[2]["ncc"]:+.3f}', v3_ncc=f'{ver[3]["ncc"]:+.3f}',
        v0_score=f'{ver[0]["score"]:.3f}', v1_score=f'{ver[1]["score"]:.3f}',
        v0_err=f'{ver[0]["gt_err_px"]:.0f}', v1_err=f'{ver[1]["gt_err_px"]:.1f}',
        n_solves=facts["verify"]["n_solves"], win_src=facts["verify"]["winner_source"],
        # covariance validation
        tau_int=f'{cov["residual_autocorr_time"]:.1f}', n_eff=f'{cov["n_eff"]:.0f}',
        n_bnd=cov["n_boundary"], block_len=cov["block_len"], n_boot=cov["n_boot"],
        sqrt_tau=f'{cov["residual_autocorr_time"] ** 0.5:.2f}',
        med_ratio=f'{cov["median_ratio"]:.2f}', med_block=f'{cov["median_block_ratio"]:.2f}',
        var_short=f'{cov["median_block_ratio"] ** 2:.1f}',
        cov_rows="".join(
            f'<tr><th class="rowname">corner {c["corner"]}</th>'
            f'<td>{c["analytic_std_px"]:.2f}</td><td>{c["bootstrap_std_px"]:.2f}</td>'
            f'<td>{c["block_bootstrap_std_px"]:.2f}</td><td>{c["block_ratio"]:.2f}</td></tr>'
            for c in cov["per_corner"]),
        # fusion calibration
        gate=f'{fus["gate"]:.0f}', loma_scale=f'{fus["loma_cov_scale"]:.0f}',
        samh_scale=f'{fus["samh_cov_scale"]:.0f}',
        d2raw=", ".join(f"{v:.0f}" for v in fus["d2_raw"]),
        d2dep=", ".join(f"{v:.1f}" for v in fus["d2_deployed"]),
        modes=", ".join(fus["modes"]),
        # complementarity
        c_frames=f'{comp["n_frames"]:,}', c_seqs=comp["n_sequences"], c_tau=f'{comp["tau_px"]:.0f}',
        c_both=f'{comp["shares_pct"]["both_good"]:.1f}',
        c_one=f'{comp["shares_pct"]["one_branch"]:.1f}',
        c_split=f'{comp["shares_pct"]["split"]:.2f}',
        c_neither=f'{comp["shares_pct"]["neither"]:.1f}',
        c_rescue=f'{comp["three_plus_one_pct"]:.2f}',
        c_kp=f'{comp["p_at_15_by_pose"]["keypoint_branch"]:.2f}',
        c_mask=f'{comp["p_at_15_by_pose"]["mask_branch"]:.2f}',
        c_branch_oracle=f'{comp["p_at_15_by_pose"]["better_branch_oracle"]:.2f}',
        c_corner_oracle=f'{comp["p_at_15_by_pose"]["per_corner_oracle"]:.2f}',
        c_fusion=f'{comp["p_at_15_by_pose"]["deployed_fusion"]:.2f}',
        c_gain=f'{comp["p_at_15_by_pose"]["per_corner_oracle"] - comp["p_at_15_by_pose"]["better_branch_oracle"]:+.2f}',
        c_bg_kp=f'{comp["both_good_median_err_px"]["keypoint_branch"]:.2f}',
        c_bg_mask=f'{comp["both_good_median_err_px"]["mask_branch"]:.2f}',
        c_bg_fused=f'{comp["both_good_median_err_px"]["deployed_fusion"]:.2f}',
        c_bg_oracle=f'{comp["both_good_median_err_px"]["better_branch_oracle"]:.2f}',
        c_beats=f'{comp["fused_beats_better_branch_pct"]:.1f}',
        # the two worked examples
        win_seq="Seq_00232", win_frame=491,
        win_kp=", ".join(f"{v:.0f}" for v in wd["kp_px"]),
        win_mask=", ".join(f"{v:.0f}" for v in wd["mask_px"]),
        win_fused=", ".join(f"{v:.0f}" for v in wd["fused_px"]),
        miss_seq="Seq_00350", miss_frame=275,
        miss_kp=", ".join(f"{v:.0f}" for v in md["kp_px"]),
        miss_mask=", ".join(f"{v:.0f}" for v in md["mask_px"]),
        miss_modes=", ".join(miss_facts["modes"]),
        drop_pct="4.06",
        # bench tables
        tbl_score=bench_table(bench, [("a=2, b=0.5 (deployed)", "abl_score_deployed"),
                                      ("a=2, b=1", "abl_score_w21"),
                                      ("a=1, b=1", "abl_score_w11")], extra_p90=True),
        tbl_recheck=bench_table(bench, [("no re-check (deployed)", "abl_score_deployed"),
                                        ("alias re-check", "abl_alias_recheck")], extra_p90=True),
        rc_good=bench_cell("abl_alias_recheck", "good"),
        rc_lost=bench_cell("abl_alias_recheck", "lost"),
        rc_lost_base=bench_cell("abl_score_deployed", "lost"),
        rc_p90=bench_cell("abl_alias_recheck", "lost", "p90"),
        rc_p90_base=bench_cell("abl_score_deployed", "lost", "p90"),
        # figures
        fig_inputs=img_tag(fig_dir, "fig_inputs.png", "Template and target frame with SAM3 masks."),
        fig_sdf=img_tag(fig_dir, "fig_sdf.png", "Signed distance field with warped boundary samples and residual histogram."),
        fig_gn=img_tag(fig_dir, "fig_gn.png", "Gauss-Newton iterates and objective curve."),
        fig_cov=img_tag(fig_dir, "fig_covariance.png", "Closed-form corner covariance against i.i.d. and block bootstrap."),
        fig_jac=img_tag(fig_dir, "fig_jacobian.png", "Boundary samples coloured by their sensitivity to each corner."),
        jac_eff_lo=min(jac["eff_support"]),
        jac_eff_hi=max(jac["eff_support"]),
        jac_rms=fmt(jac["residual_rms_px"], 2),
        jac_dof=jac["dof"],
        jac_sigmas=", ".join(fmt(s, 2) for s in jac["sigma_px"]),
        jac_gain=fmt(jac["ellipse_gain"], 0),
        tbl_scope=scope_table(scope),
        wf_p_base=fmt(pti("FINAL", "p15")), wf_p=fmt(pti("WHOLEFRAME", "p15")),
        wf_br_base=fmt(pti("FINAL", "br15")), wf_br=fmt(pti("WHOLEFRAME", "br15")),
        wf_gap_base=fmt(pti("FINAL", "br15") - pti("FINAL", "p15")),
        wf_gap=fmt(pti("WHOLEFRAME", "br15") - pti("WHOLEFRAME", "p15")),
        wf_br_gain=fmt(pti("WHOLEFRAME", "br15") - pti("FINAL", "br15")),
        ws_pot=fmt(_scope_gain(scope, "POT-210")[0]),
        ws_pti=fmt(_scope_gain(scope, "PlanarTrackInitfix")[0]),
        ws_pot_share=fmt(_scope_gain(scope, "POT-210")[1], 0),
        ws_pti_share=fmt(_scope_gain(scope, "PlanarTrackInitfix")[1], 0),
        fig_propose=img_tag(fig_dir, "fig_propose.png", "Second-moment proposals and their overlap scores per sweep angle."),
        fig_verify=img_tag(fig_dir, "fig_verify.png", "A pose and its three cyclic relabelings with IoU and NCC."),
        fig_fusion=img_tag(fig_dir, "fig_fusion.png", "Corner estimates with covariance ellipses and per-corner disagreement."),
        fig_win=img_tag(win, "fig_fusion.png", "A frame where each branch is wrong about a different corner and fusion recovers both."),
        fig_miss=img_tag(miss, "fig_fusion.png", "A frame where the keypoint branch is dropped for lacking a covariance."),
        # grid tables
        tbl_branches=metric_table(m, ["LMr", "SAM3H", "SAM2H", "WOFTSAM2", "SWITCH35", "ORACLE"],
                                  ["POT", "PTi"], baseline="SWITCH35"),
        tbl_thread=metric_table(m, ["SWITCH35", "ALIGN-PURE", "FUSE-ALIGN", "FUSE-REDETECT",
                                    "FUSE-REG", "FUSE-REG-KP", "FINAL", "FINAL+RECHECK"],
                                ["POT", "PTi"], baseline="SWITCH35"),
        tbl_noapp=metric_table(m, ["FUSE-REG-KP", "FUSE-REG-NOAPP"], ["POT", "PTi"],
                               baseline="FUSE-REG-KP"),
        tbl_gap=gap_table(m, ["SWITCH35", "FUSE-REDETECT", "FUSE-REG", "FUSE-REG-KP",
                              "FUSE-REG-NOAPP", "FINAL"], "PTi"),
        tbl_six=metric_table(m, ["WOFTSAM2", "SAM2H", "SWITCH35", "FINAL"],
                             ["POT", "PTi", "POIC", "TMT", "UCSB", "MPOT"], baseline="SWITCH35"),
        # prose numbers
        sw_pti15=fmt(pti("SWITCH35", "p15")), sw_pot15=fmt(pot("SWITCH35", "p15")),
        fin_pti15=fmt(pti("FINAL", "p15")), fin_pti5=fmt(pti("FINAL", "p5")),
        fin_pot15=fmt(pot("FINAL", "p15")),
        or_pti15=fmt(pti("ORACLE", "p15")),
        ap_drop=delta(pti("ALIGN-PURE", "p15"), pti("SWITCH35", "p15")),
        fa_pti15=fmt(pti("FUSE-ALIGN", "p15")), reg_pti15=fmt(pti("FUSE-REG", "p15")),
        reg_gain=delta(pti("FUSE-REG", "p15"), pti("FUSE-ALIGN", "p15")),
        kp_pti15=fmt(pti("FUSE-REG-KP", "p15")),
        noapp_pti15=fmt(pti("FUSE-REG-NOAPP", "p15")),
        noapp_drop=delta(pti("FUSE-REG-NOAPP", "p15"), pti("FUSE-REG-KP", "p15")),
        kp_br=fmt(m["FUSE-REG-KP"]["PTi"]["br15"]), noapp_br=fmt(m["FUSE-REG-NOAPP"]["PTi"]["br15"]),
        kp_gap=fmt(m["FUSE-REG-KP"]["PTi"]["gap"]), noapp_gap=fmt(m["FUSE-REG-NOAPP"]["PTi"]["gap"]),
        rd_pti15=fmt(pti("FUSE-REDETECT", "p15")), rd_gap=fmt(m["FUSE-REDETECT"]["PTi"]["gap"]),
        rd_gap5=fmt(m["FUSE-REDETECT"]["PTi"]["gap5"], 1),
        rd_br=fmt(m["FUSE-REDETECT"]["PTi"]["br15"]),
        fin_p5_delta=delta(pti("FINAL", "p5"), pti("SWITCH35", "p5")),
        fin_p15_delta=delta(pti("FINAL", "p15"), pti("SWITCH35", "p15")),
        ucsb_w=fmt(m["WOFTSAM2"]["UCSB"]["p15"]), ucsb_s=fmt(m["SWITCH35"]["UCSB"]["p15"]),
        ucsb_f=fmt(m["FINAL"]["UCSB"]["p15"]),
        pti_frames=f'{m["SWITCH35"]["PTi"]["frames"]:,}',
        pot_frames=f'{m["SWITCH35"]["POT"]["frames"]:,}',
        recheck_row=("" if "PTi" not in m.get("FINAL+RECHECK", {}) else
                     f'FINAL+RECHECK on the full grids: PTi {fmt(pti("FINAL+RECHECK", "p15"))} P@15 '
                     f'({delta(pti("FINAL+RECHECK", "p15"), pti("FINAL", "p15"))} against FINAL), '
                     f'POT {fmt(pot("FINAL+RECHECK", "p15"))}.'),
    )


TEMPLATE = r"""<title>FlatSAM2 Tracker Cards</title>
<style>
  :root {{
    --ground:#F2F4F6; --surface:#FFFFFF; --surface-2:#E8ECF0; --ink:#0E1318;
    --muted:#57646F; --rule:#D3DBE2;
    --kp:#8F5B10; --mask:#0E6675; --fused:#544A86; --bad:#94301F; --ok:#1F6B45;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ground:#0B0F12; --surface:#131A20; --surface-2:#1A232B; --ink:#E2E9EF;
      --muted:#8B98A5; --rule:#212B34;
      --kp:#DCA84A; --mask:#3FB2C4; --fused:#9E8FD8; --bad:#E4735E; --ok:#5CC08C;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground:#0B0F12; --surface:#131A20; --surface-2:#1A232B; --ink:#E2E9EF;
    --muted:#8B98A5; --rule:#212B34;
    --kp:#DCA84A; --mask:#3FB2C4; --fused:#9E8FD8; --bad:#E4735E; --ok:#5CC08C;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    background:var(--ground); color:var(--ink); margin:0; padding:0 20px 96px;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    font-size:16px; line-height:1.6; -webkit-font-smoothing:antialiased;
  }}
  .wrap {{ max-width:980px; margin:0 auto; }}
  .mono, code, table {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
  h1,h2,h3 {{ font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
             letter-spacing:-.01em; text-wrap:balance; }}
  h1 {{ font-size:clamp(32px,5.5vw,48px); line-height:1.05; margin:0; font-weight:600; }}
  h2 {{ font-size:27px; margin:0 0 4px; font-weight:600; }}
  h3 {{ font-size:19px; margin:26px 0 6px; font-weight:600; }}
  p, li {{ max-width:76ch; }}
  header.masthead {{ padding:60px 0 26px; border-bottom:2px solid var(--ink); display:grid; gap:16px; }}
  .eyebrow {{ font-size:11.5px; letter-spacing:.14em; text-transform:uppercase;
              color:var(--muted); font-weight:600; }}
  .standfirst {{ font-size:17.5px; color:var(--muted); margin:0; max-width:72ch; }}
  .standfirst strong {{ color:var(--ink); font-weight:600; }}
  section {{ padding-top:46px; }}
  .lede {{ color:var(--muted); margin:0 0 18px; max-width:76ch; }}
  math {{ font-size:1.05em; }}
  math[display="block"] {{ margin:14px 0; overflow-x:auto; }}
  .mathbox {{ background:var(--surface-2); border-radius:3px; padding:10px 16px; margin:14px 0;
              overflow-x:auto; }}
  .where {{ font-size:13.5px; color:var(--muted); margin:6px 0 0; max-width:76ch; }}
  ol.recipe {{ max-width:80ch; padding-left:1.35em; margin:16px 0; }}
  ol.recipe > li {{ margin:0 0 16px; padding-left:6px; }}
  ol.recipe > li::marker {{ color:var(--accent); font-variant-numeric:tabular-nums;
                            font-weight:700; }}
  ol.recipe ul {{ margin:8px 0 4px; padding-left:1.1em; }}
  ol.recipe ul > li {{ margin:0 0 8px; }}
  .why {{ display:block; font-size:13.5px; color:var(--muted); margin-top:5px;
          border-left:2px solid var(--rule); padding-left:11px; }}
  .card {{ background:var(--surface); border:1px solid var(--rule); border-radius:3px;
           padding:20px 22px; margin-top:18px; }}
  .card.kp {{ border-left:3px solid var(--kp); }}
  .card.mask {{ border-left:3px solid var(--mask); }}
  .card.fused {{ border-left:3px solid var(--fused); }}
  .card.dead {{ border-left:3px solid var(--bad); }}
  .card.good {{ border-left:3px solid var(--ok); }}
  .card h3 {{ margin-top:0; }}
  figure {{ margin:22px 0 0; background:var(--surface); border:1px solid var(--rule);
            border-radius:3px; padding:16px; }}
  figure img {{ display:block; width:100%; height:auto; border-radius:2px; }}
  figcaption {{ font-size:13.5px; color:var(--muted); margin-top:12px;
                border-left:2px solid var(--rule); padding-left:12px; max-width:80ch; }}
  figcaption strong {{ color:var(--ink); }}
  .tablewrap {{ overflow-x:auto; margin:14px 0; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }}
  th, td {{ text-align:right; padding:6px 8px; border-bottom:1px solid var(--rule); white-space:nowrap; }}
  thead th {{ color:var(--muted); font-weight:600; font-size:11.5px; }}
  th.rowname {{ text-align:left; font-weight:600; }}
  td .d {{ color:var(--muted); font-size:11px; margin-left:6px; }}
  tbody tr:last-child td {{ border-bottom:0; }}
  .dim {{ color:var(--muted); }}
  .missing {{ color:var(--bad); font-family:monospace; }}
  ul.tight {{ margin:8px 0; padding-left:20px; }}
  ul.tight li {{ margin-bottom:6px; max-width:76ch; }}
  footer {{ margin-top:56px; padding-top:16px; border-top:1px solid var(--rule);
            font-size:12.5px; color:var(--muted); }}
</style>

<div class="wrap">

<header class="masthead">
  <div class="eyebrow">FlatSAM2 · design record · generated by experiments/build_tracker_cards.py</div>
  <h1>Tracker Cards</h1>
  <p class="standfirst">The tracker between WOFTSAM/SAM-H and today. <strong>WOFT, WOFTSAM and SAM-H are
  assumed known; everything else is defined here.</strong> Each mechanism is given as equations plus the
  function that implements it, each figure is that function running on real frames, and each claim carries
  the measurement that supports it — including the ones that came out against the design.</p>
</header>

<section>
  <h2>0 · Conventions</h2>

  <h3>0.1 Names</h3>
  <p class="lede">Short names below denote exactly one config, which is also its results directory.
  <b>FUSE-REG-KP and FINAL are different configs</b> — see §7.</p>
  <div class="tablewrap"><table>
    <thead><tr><th class="rowname">short name</th>
    <th class="rowname" style="text-align:left">config stem</th></tr></thead>
    <tbody>{name_rows}</tbody></table></div>

  <h3>0.2 Symbols</h3>
  <ul class="tight">
    <li><b>C</b> = [c₁ c₂ c₃ c₄], the 2×4 <b>control points</b>: image positions of the template's four
      corners in fixed cyclic order. <b>θ</b> = vec(C) ∈ ℝ⁸ is what every solver here optimizes.</li>
    <li><b>H(C)</b>: the homography taking the template corners to C (the 4-point parameterization). No
      solver touches raw 3×3 coefficients — that choice is what makes §3.3 work.</li>
    <li><b>M_t</b>: the SAM3 mask of frame t. <b>D_t</b>: its signed distance field, negative inside,
      in pixels. <b>B</b> = {{b₁ … b_N}}: N points sampled on the template silhouette's contour
      (N = {nb}).</li>
  </ul>

  <h3>0.3 Metric and aggregation</h3>
  {M[error]}
  {M[patk]}
  <p class="where">V = all valid annotated frames of all sequences, first frame excluded. This
  <b>pooled</b> form — every frame counts once — is the aggregation the benchmark success plots use and
  equals <code>overall_p@k</code> in <code>evaluate.py</code>. PTi pools {pti_frames} frames over 300
  sequences; POT-210 pools {pot_frames} over 210 (its GT is sparse).</p>
  <div class="card dead">
    <p><b>Warning.</b> <code>docs/state-of-the-tracker.md</code> and older journal entries quote
    <em>per-sequence means</em> instead, which weight a 14-frame sequence like a 2 500-frame one. Numbers
    here differ from those documents; on MPOT-3K the two even disagree in sign. The pooled number is the
    one to quote.</p>
  </div>
  <p><b>br-P@15</b> is P@15 after choosing, per frame, the cyclic relabeling of the predicted corners that
  minimizes the error; <b>gap = br-P@15 − P@15</b> separates corner <em>naming</em> from
  <em>localization</em>.</p>
</section>

<section>
  <h2>1 · Two branches, one question</h2>
  <p class="lede">Both branches consume the same SAM3 mask stream and output four corners with a 2×2
  covariance each. The keypoint branch (LMr) matches texture and fits a homography by MAGSAC on the
  correspondences inside the mask dilated by 10 px. The mask branch estimates the pose from the silhouette
  alone. Everything in §3–§6 is about the mask branch's estimator and how the two are combined.</p>
  {tbl_branches}
  <p class="where">ORACLE is not a tracker: it picks, per frame, whichever branch is closer to the ground
  truth. Its PTi P@15 of {or_pti15} against SWITCH35's {sw_pti15} bounds what any combination rule can buy.</p>
  <figure>
    {fig_inputs}
    <figcaption>The case used in every mechanism figure below: <b>{seq}, frame {frame}</b> (PTi). Left the
    template and its silhouette, right the frame to register and its SAM3 mask.</figcaption>
  </figure>
</section>

<section>
  <h2>2 · What the mask branch optimizes</h2>
  <p class="lede">Implementation: <code>flatsam2/mask_align.py::_gn_align</code>. One objective, stated
  once.</p>
  {M[residual]}
  {M[objective]}
  <p class="where">In words: warp the template outline by the candidate pose, and measure how far each of
  its sampled points lands from the current mask's boundary, where D_t = 0. δ = {huber} px. Only the
  outline enters, so the objective is defined for any shape and is blind to texture — which is exactly the
  weakness §5 has to repair.</p>

  <h3>2.1 The solver</h3>
  {M[normal]}
  <p class="where">J = ∂r/∂θ by forward differences (0.5 px step), λ Levenberg damping starting at 1,
  halved on an accepted step and quadrupled on a rejected one; stop at max|Δθ| &lt; 0.05 px or
  {max_iters} iterations. W is the IRLS weight of the Huber loss, so a sample beyond δ is down-weighted,
  not discarded.</p>
  <figure>
    {fig_sdf}
    <figcaption>Left: D_t, cropped to the mask's bounding box grown by <code>sdf_roi_margin = {roi}</code>
    of its longer side, with the {n_boundary} warped boundary samples on it. Right: their residuals,
    median |r| = {med_r} px at a pose perturbed by {perturb} px, with ±δ marked.</figcaption>
  </figure>
  <figure>
    {fig_gn}
    <figcaption>A real solve from that perturbed start: control-point quads per iteration, and the
    objective falling {gn_c0} → {gn_c1} over {gn_iters} iterations.</figcaption>
  </figure>
</section>

<section>
  <h2>3 · Corner covariance — derivation and validation</h2>
  <p class="lede">Both branches must speak the same language for §6 to be possible. This is the whole
  construction, and then the check that it is true.</p>

  <h3>3.1 The recipe, step by step</h3>
  <p>Both branches run the <em>same seven steps</em>; they differ only in step 1 and step 2. Nothing below
  is specific to masks or to keypoints, which is the whole point — it is what lets the two report
  uncertainty in one currency.</p>

  <ol class="recipe">
    <li><b>Fix what is being estimated.</b> θ = (x₁,y₁,x₂,y₂,x₃,y₃,x₄,y₄) ∈ ℝ⁸ — the four corner
    positions in the current frame, and nothing else. Any θ determines a homography H(θ): the unique
    projective map taking the four <em>template</em> corners to those four points
    (<code>cv2.getPerspectiveTransform</code>, an 8×8 solve). <span class="why">Why this parameterization
    and not the 8 entries of H: the answer we want <em>is</em> the corners, so their covariance will fall
    out with no change of variables. Parameterize by H and you must afterwards push the covariance through
    the warp Jacobian — an extra approximation for nothing.</span></li>

    <li><b>Take the evidence, and write one residual per piece of it.</b> A residual answers: <em>if the
    corners were θ, how wrong would this one piece of evidence be?</em>
      <ul>
        <li><b>Mask branch</b> — take the <b>N = {n_bnd} template boundary points</b> b_i, sampled evenly
        along the outline of the template silhouette, and the signed distance field D_t of this frame's
        SAM3 mask (§2). Warp each b_i by H(θ) and read the field there: r_i(θ) = D_t(H(θ)·b_i), one number
        per point. <span class="why">Zero means that boundary sample landed exactly on the mask's edge,
        which is what a correct pose does.</span></li>
        <li><b>Keypoint branch</b> — take the <b>MAGSAC inlier correspondences</b> 𝓘 = {{(p_i, q_i)}}: p_i a
        template-frame keypoint, q_i its match in this frame. Warp p_i and compare:
        r_i(θ) = H(θ)·p_i − q_i, <em>two</em> numbers per correspondence, so the stacked vector is 2|𝓘|
        long. <span class="why">Zero means that match is reprojected exactly onto its partner.</span></li>
      </ul>
    Both stack into one vector r(θ) with as many rows as there are pieces of evidence. Both are already
    the objective the branch minimized — this is not a new model fitted afterwards.</li>

    <li><b>Differentiate the residuals with respect to the corners.</b> For each of the 8 coordinates in
    turn: move that one coordinate by ±0.5 px, recompute the <em>entire</em> residual vector both times,
    and take the central difference. That fills one column of J, so J is (rows = evidence) × 8 and costs
    16 residual evaluations.
    <span class="why">Why by finite differences rather than analytically: r goes through
    <code>getPerspectiveTransform</code> and a bilinear field lookup; ±0.5 px is well inside the smooth
    regime, matches the aligner's own step, and smaller steps lose to cancellation rather than gaining
    accuracy. <b>J is where all the geometry lives</b>: entry (i, p) says how strongly evidence i responds
    to moving corner coordinate p. Evidence that does not respond does not constrain that corner — which
    is exactly what a large uncertainty ellipse is reporting.</span></li>

    <li><b>Weight the rows the way the fit did.</b> W is diagonal: 1 for the keypoint branch, the fit's own
    Huber weights w_i = min(1, δ/|r_i|) for the mask branch. <span class="why">The covariance must describe
    the estimator that actually ran; a row the solve down-weighted did not get a full vote in θ̂ and must
    not get one here.</span></li>

    <li><b>Get the noise scale from the fit's own residuals.</b> σ̂² = Σ w_i r_i² / (N − 8).
    <span class="why">N − 8, not N, because eight parameters were fitted to this same data — the residuals
    are that much smaller than the true noise. There is no external noise model anywhere: the fit's own
    misfit is the estimate of how noisy its evidence was.</span></li>

    <li><b>Invert the normal matrix.</b> This is the standard first-order covariance of a weighted
    nonlinear least-squares estimate at its converged point:</li>
  </ol>
  {M[cov]}

  <div class="card">
    <h3 style="margin-top:0">Why an inverse, when J points the wrong way</h3>
    <p>Step 3 measures <b>how wrong the evidence goes when a corner moves</b> — ∂r/∂θ. What we want is the
    opposite: <b>how far a corner moves when the evidence moves</b> — ∂θ/∂y. The two are connected, and
    <b>the inversion in the formula above is that connection</b>, not a bookkeeping detail.</p>
    <p>θ̂ is not an explicit function of the evidence — it is whatever minimizes the fit, defined
    <em>implicitly</em>. So it cannot be differentiated directly. Instead differentiate the condition that
    defines it. At the optimum the gradient vanishes, and r = f(θ) − y, so ∂r/∂y = −I:</p>
    {M[implicit]}
    <p class="where">Solving for δθ gives the sensitivity we actually wanted,
    <b>∂θ̂/∂y = (JᵀWJ)⁻¹JᵀW</b>. Propagating evidence noise through it,
    Cov(θ̂) = (∂θ̂/∂y) Cov(y) (∂θ̂/∂y)ᵀ, collapses to σ̂²(JᵀWJ)⁻¹ whenever Cov(y) = σ²W⁻¹. The forward
    Jacobian is the only differentiable object on the path, which is why it is computed "backwards" and
    then inverted.</p>
    <p><b>The scalar case makes it obvious.</b> With one parameter and rᵢ = c·θ − yᵢ, where c is how
    strongly that corner drives the residual: JᵀJ = Nc², so Σ = σ²/(Nc²) — and directly, δθ = δy/c.
    <b>∂θ/∂y is one over ∂r/∂θ.</b> A corner whose movement barely disturbs the residuals is a corner that
    evidence noise can shove a long way before the fit objects; a corner whose movement wrecks the fit
    immediately is pinned. J is the <em>stiffness</em> of the fit against moving each corner; the
    covariance is the <em>compliance</em>, stiffness inverted and scaled by how noisy the evidence is.</p>
    <p>Which is also why the two readings are the same statement: "the evidence goes badly wrong when this
    corner moves" is precisely what it means for that evidence to be <em>informative</em> about the corner.
    JᵀWJ is the Fisher information; covariance is inverse information. Sensitivity, information and
    1/variance are one quantity read three ways.</p>
    <p><b>What "the evidence moves" is, concretely.</b> For the mask branch y is where SAM3's boundary
    actually sits, so the covariance answers: <em>had SAM3 drawn this silhouette slightly differently,
    where would this corner have landed?</em> For the keypoint branch y is the matched point positions, and
    the same question is asked of match jitter.</p>
  </div>
  <div class="card dead">
    <p><b>Caveat this derivation exposes.</b> The collapse to σ̂²(JᵀWJ)⁻¹ needs Cov(y) = σ²W⁻¹, i.e. weights
    that <em>are</em> inverse variances. The mask branch's W holds <b>Huber</b> weights, chosen for
    robustness against a boundary that is partly wrong, not because they are inverse variances. Strictly
    the sandwich (∂θ̂/∂y) Cov(y) (∂θ̂/∂y)ᵀ therefore does not collapse and we use the convenient form
    anyway. That is a second, independent source of miscalibration alongside the correlated residuals of
    §3.3, and part of why the deployed scale corrections (§6.2) are as large as they are.</p>
  </div>
  <ol class="recipe" start="7">
    <li><b>Read the corners off the diagonal.</b> Because θ <em>is</em> the corner positions,
    <b>Σ_k is simply the 2×2 diagonal block of Σ_θ for corner k</b> — no propagation step. The off-diagonal
    blocks, i.e. how corner 1's error correlates with corner 3's, are computed and then <em>discarded</em>;
    §6 treats the four corners as independent. <span class="why">That is a real approximation and the
    honest place to flag it: the corners of one fitted quad are correlated, so the fusion is not using the
    full joint covariance it could.</span> Finally each Σ_k's eigenvalues are clamped from below at
    (0.25 px)². <span class="why">A near-singular normal matrix — a boundary arc too short to pin an axis —
    otherwise produces an eigenvalue near zero, i.e. a claim of near-certainty, and any Mahalanobis
    distance computed against it explodes; values near 10⁸ were observed in the field.</span></li>
  </ol>

  <p>Implementations, line for line the same formula:
  <code>flatsam2/mask_align.py::_corner_cov_from_boundary_fit</code> and
  <code>flatsam2/corner_covariance.py::homography_corner_covs</code>.</p>

  <h3>3.2 What J looks like on a real frame</h3>
  <p>Since every claim about which corner is uncertain is a claim about J, here is J itself, computed by
  the deployed code on real data — each boundary sample coloured by ‖∂r_i/∂c_k‖, its sensitivity to moving
  corner k one pixel.</p>
  <figure>
    {fig_jac}
    <figcaption>{n_bnd} boundary samples on frame {frame} of {seq}, one panel per corner. <strong>The
    samples that constrain a corner are the ones near it</strong>, falling off along the two edges that
    meet there: {jac_eff_lo}–{jac_eff_hi} samples carry 80% of each corner's sensitivity, out of {n_bnd}.
    That is the whole reason a partly-occluded or foreshortened side produces a large ellipse — the rows of
    J that would have pinned those corners are gone. Residual RMS {jac_rms} px over {jac_dof} degrees of
    freedom gives the σ̂ that scales every ellipse; the per-corner σ here are {jac_sigmas} px, so the
    drawn 1σ ellipses are magnified {jac_gain}× to be visible at all.</figcaption>
  </figure>
  <p><b>What the covariance means, and what it does not.</b> It is how far this corner would move if the
  evidence this branch used were resampled — a <em>sampling-noise</em> model. It cannot know the evidence
  is the wrong object: a confident fit to a distractor gives a tight, wrong ellipse. Gross-error rejection
  is a separate mechanism (§6's gate), not something the covariance can provide.</p>

  <h3>3.3 Validation on real data, and where it breaks</h3>
  <p>The direct check is to resample the evidence and see where the corners actually land. Two resamplings:
  an i.i.d. bootstrap over the boundary samples (the assumption the closed form makes) and a moving-block
  bootstrap with block length L = {block_len} (which preserves runs of correlated residuals).</p>
  <figure>
    {fig_cov}
    <figcaption>{n_boot} re-solves per corner on frame {frame} of {seq}. The closed form (solid) matches the
    i.i.d. cloud in scale, but the block bootstrap (dotted) is <strong>systematically wider</strong>: the
    boundary residuals are not independent.</figcaption>
  </figure>
  <div class="tablewrap"><table>
    <thead><tr><th></th><th>closed form σ (px)</th><th>i.i.d. bootstrap σ</th>
    <th>block bootstrap σ</th><th>block / closed form</th></tr></thead>
    <tbody>{cov_rows}</tbody></table></div>
  <p>The reason is physical: the samples are ordered along the contour, and a mask boundary error is
  spatially smooth, so neighbouring residuals err together. The measured autocorrelation of r along the
  contour gives the effective sample size the independence assumption over-counts:</p>
  {M[neff]}
  <p class="where">Measured on this frame: τ_int = {tau_int} samples, so N_eff = {n_eff} of {n_bnd}
  boundary points. That predicts a σ inflation of √τ_int = {sqrt_tau}×; the block bootstrap measures
  {med_block}× (median over corners), against {med_ratio}× for the i.i.d. bootstrap. <b>So most of the
  closed form's optimism is explained from first principles by correlated residuals</b>, and the correction
  is computable per frame at no cost — τ_int comes from the residual vector the solver already has.</p>
</section>

<section>
  <h2>4 · Propose → refine → verify</h2>
  <p class="lede">A single solve from the last pose only works when the last pose is nearly right, which is
  when the branch is not needed. So the pose is searched. Implementation:
  <code>flatsam2/mask_register.py::register_mask</code>.</p>
  <h3>4.1 Propose</h3>
  <p>Whitening a silhouette by the inverse square root of its second-moment matrix leaves it defined up to
  a rotation, so two whitened silhouettes agree up to exactly one angle:</p>
  {M[whiten]}
  <p class="where">Sweeping φ over {n_angles} steps enumerates every affine pose the two shapes' second
  moments are consistent with — no corners, lines or convexity anywhere, so it is defined for any shape.
  Reflections are excluded: a planar object seen from the front cannot mirror. The last-frame pose and any
  externally supplied pose enter as proposals alongside the sweep.</p>
  <h3>4.2 Refine</h3>
  <p class="where">The {n_refine} best proposals get a coarse GN solve (long side 256 px), the best
  {n_full} of those a full-resolution solve. The solve is §2, unchanged.</p>
  <figure>
    {fig_propose}
    <figcaption>All {n_prop} proposals and their scores against sweep angle. <strong>The four near-equal
    peaks 90° apart are the aliasing problem, visible in the ranking itself</strong> — this silhouette maps
    onto itself under quarter turns, so overlap cannot separate those four poses. Best proposal IoU
    {best_iou}.</figcaption>
  </figure>
</section>

<section>
  <h2>5 · The verification score, from first principles</h2>
  <p class="lede">This is the part that most deserved the accusation of being convoluted. Here is what it
  is, what justifies it, and what the data says about its parameters.</p>

  <h3>5.1 What is being asked</h3>
  <p>Given a candidate pose, two <em>independent</em> pieces of evidence are available: the warped template
  silhouette should coincide with the mask, and the surface under the warped template should look like the
  template. Writing the two agreements as likelihood terms and taking their log gives a log-linear pool —
  a weighted sum of log-evidence, i.e. a weighted geometric mean of the two agreements:</p>
  {M[score]}
  <p class="where">with IoU the raster intersection-over-union of the warped silhouette against M_t, and
  NCC the normalized cross-correlation of the warped template's grey values against the frame, computed
  <b>only inside warp ∧ M_t</b> so occluders and background never enter the sum. (1 + NCC)/2 maps
  correlation to [0, 1]. The deployed setting is a = {a_pow}, b = {b_pow}; ranking uses s, so any monotone
  rescaling is irrelevant and only the <em>ratio</em> a : b matters.</p>
  <p>Two things follow immediately and are worth stating plainly: the exponents are <b>weights on two
  log-evidence terms</b>, not tuning knobs with independent meaning; and a third factor that used to sit in
  this product — a sharpness term in the fit residual — is <b>disabled</b> (<code>residual_power = 0</code>)
  because the residual was measured to separate right from wrong poses only weakly.</p>

  <h3>5.2 Do the weights matter? (offline benchmark, PTi test, 961 cases per regime)</h3>
  {tbl_score}
  <p class="where">P@15 by initialization regime; the last two columns are the 90th-percentile corner error
  in pixels. Within a : b from 2 : 0.5 to 1 : 1 the metric moves by at most 0.7 pp anywhere, so
  <strong>the weights are not identifiable from this data and should not be presented as tuned
  parameters</strong>. The simple choice a = 2, b = 1 is as good or better everywhere except 0.1 pp in
  <code>good</code>, and cuts the p90 error substantially when the pose is lost. What is <em>not</em>
  optional is the second term at all: dropping it (b = 0) costs 13 pp — see §7's ablation.</p>
  <figure>
    {fig_verify}
    <figcaption>Why the texture term exists, on one frame: a pose and its three cyclic relabelings — same
    warped quadrilateral, corners renamed. IoU moves only between {v0_iou} and {v1_iou}; NCC moves from
    {v0_ncc} to {v1_ncc}, {v2_ncc}, {v3_ncc}. <strong>Silhouette overlap cannot name corners; texture
    can.</strong></figcaption>
  </figure>

  <h3>5.3 A defect this figure exposed, and its fix</h3>
  <p>In that panel the returned pose (left, {v0_err} px from GT, score {v0_score}) is beaten by its own +1
  relabeling ({v1_err} px, score {v1_score}). Cause: <code>alias_shortcut</code> compares relabelings of
  <em>coarsely refined proposals</em>, and the full-resolution solve then moves the pose with nothing
  re-examining its naming.</p>
  <p>The obvious fix — re-score the winner's three relabelings — <b>loses 3.0 pp P@15 in the
  <code>good</code> regime</b> while gaining 0.9 when lost, because a naming inherited from the previous
  frame carries temporal evidence the single-frame score does not, and a noisy score then flips correct
  namings. Requiring an inherited naming to be beaten by the same margin a challenger pose must clear
  restores that:</p>
  {tbl_recheck}
  <p class="where">Corrected rule: <code>good</code> unchanged at {rc_good}, <code>lost</code>
  {rc_lost_base} → {rc_lost}, and the p90 error when lost {rc_p90_base} → {rc_p90} px. Shipped as
  <code>register_alias_recheck</code> (config <code>corner_fusion_register_recheck</code>). {recheck_row}</p>
</section>

<section>
  <h2>6 · Fusing the branches — and whether per-corner is justified</h2>
  <p class="lede">Implementation: <code>flatsam2/tracking/corner_fusion.py::fuse_corners</code>.</p>

  <h3>6.1 The rule</h3>
  {M[maha]}
  {M[fuse]}
  <p class="where">a_k, b_k are the two branches' corner k, with covariances Σ_a,k and Σ_b,k. When the two
  agree the corner is the precision-weighted average above; when they disagree the rule takes the branch
  with the smaller covariance trace. A missing branch is zero precision, so it drops out with no special
  case. G = {gate}.</p>
  <p><b>The decision is taken once for the whole quad, not four times.</b> The original rule compared
  d²_k ≤ G per corner, so its output could take two corners from one branch and two from the other — a
  pose neither branch ever proposed. §6.3 measures what that freedom is worth (almost nothing) and §6.5
  measures what removing it is worth (a real gain). The shipped rule
  (<code>fusion.decision_scope='whole_frame'</code>) tests the <em>mean</em> of the four d²_k against the
  same G — under the same model the sum over four corners is χ²₈, so the mean keeps G in its original
  χ²₂ units — and then either fuses every corner or hands the whole pose to the more confident branch.</p>

  <div class="card dead">
    <h3 style="margin-top:0">6.2 Calibration — this is not probabilistic fusion</h3>
    <p>Under a correct Gaussian model d²_k is χ²₂ (median 1.39, 95% at 5.99). Deployed: covariances are
    multiplied by <b>{loma_scale}</b> (keypoint) and <b>{samh_scale}</b> (mask) before the gate, and
    G = {gate} sits above the χ² quantile on top of that. §3.3 explains part of it from first principles —
    correlated residuals cost a factor ≈{var_short} in variance — but not a factor of {samh_scale}. The
    remainder is model error: bias in the mask, and fits to the wrong object, which no resampling
    covariance can see. <b>Honest description: a relative precision model with an empirical scale
    correction, and a tuned gate.</b> The per-frame τ_int correction of §3.3 is the obvious way to replace
    part of the fudge with something derived; it is not yet implemented.</p>
  </div>

  <h3>6.3 Is per-corner fusion doing anything? (PTi, {c_frames} frames, {c_seqs} sequences)</h3>
  <p>Classify every frame where both branches produced a pose, by whether each branch's corners are within
  τ = {c_tau} px of the ground truth:</p>
  <div class="tablewrap"><table>
    <thead><tr><th class="rowname">case</th><th>share of frames</th><th>meaning</th></tr></thead>
    <tbody>
      <tr><th class="rowname">both branches all 4 good</th><td>{c_both}%</td>
        <td style="text-align:left">nothing to rescue; averaging can only reduce noise</td></tr>
      <tr><th class="rowname">exactly one branch all 4 good</th><td>{c_one}%</td>
        <td style="text-align:left">a <em>selection</em> problem, not a per-corner one</td></tr>
      <tr><th class="rowname">complementary (neither alone, together yes)</th><td>{c_split}%</td>
        <td style="text-align:left">the case per-corner fusion exists for</td></tr>
      <tr><th class="rowname">neither, even per corner</th><td>{c_neither}%</td>
        <td style="text-align:left">out of reach of any combination rule</td></tr>
    </tbody></table></div>
  <p>And the ceiling, as pose-level P@15 on the same frames:</p>
  <div class="tablewrap"><table>
    <thead><tr><th class="rowname"></th><th>P@15</th></tr></thead><tbody>
      <tr><th class="rowname">keypoint branch alone</th><td>{c_kp}</td></tr>
      <tr><th class="rowname">mask branch alone</th><td>{c_mask}</td></tr>
      <tr><th class="rowname">oracle: better <em>whole branch</em> per frame</th><td>{c_branch_oracle}</td></tr>
      <tr><th class="rowname">oracle: better <em>branch per corner</em></th><td>{c_corner_oracle}</td></tr>
      <tr><th class="rowname">deployed fusion</th><td>{c_fusion}</td></tr>
    </tbody></table></div>
  <div class="card dead">
    <p><b>The per-corner argument is weak, and here is the number.</b> Going from choosing the better whole
    branch to choosing the better branch <em>per corner</em> is worth <b>{c_gain} pp</b> of P@15 at the
    oracle level, and the frames it can rescue are <b>{c_split}%</b> of all frames ({c_rescue}% for the
    "three good corners plus one borrowed" pattern the rule is usually sold with). Meanwhile the deployed
    fusion sits {c_fusion} against the whole-branch oracle's {c_branch_oracle}. <b>Almost all the available
    headroom is in choosing the right branch, not in mixing corners.</b></p>
    <p>What per-corner fusion <em>does</em> buy is the ordinary variance reduction of averaging two
    estimates of the same quantity, and that is real: on the {c_both}% of frames where both branches are
    good, the median error is {c_bg_kp} px (keypoint), {c_bg_mask} px (mask), and <b>{c_bg_fused} px
    fused</b> — better than either input and close to the {c_bg_oracle} px oracle. Across all frames the
    fused pose beats the better branch on {c_beats}% of them. That is the honest justification: averaging
    when they agree, selecting when they do not — <em>not</em> because complementary corners are common.
    Note that this justification says nothing about deciding per corner: averaging down noise works just
    as well under one decision for the whole quad.</p>
  </div>

  <h3>6.5 Removing the per-corner freedom — measured, not argued</h3>
  <p>Because neither branch reads the fused output, a run's stored per-frame branch estimates are exactly
  what a rerun under a different fusion rule would see. So the rule can be changed and re-scored offline
  against a completed grid with no GPU and no approximation
  (<code>experiments/fusion_rule_replay.py</code>; its <code>baseline</code> row reproduces the run's own
  eval to five decimals). Pooled P@15, every GT frame counted once:</p>
  {tbl_scope}
  <p class="where">Deciding once per frame is <b>better</b>, not merely simpler: +{ws_pot} pp on POT and
  +{ws_pti} pp on PTi, which is {ws_pot_share}% and {ws_pti_share}% of the entire gap to the
  whole-branch oracle. Mixing corners was not buying accuracy; it was spending it, by emitting quads no
  branch had proposed and no verifier had scored.</p>
  <div class="card good">
    <p><b>Confirmed at grid scale, to the decimal.</b> <code>WHOLEFRAME</code>
    (<code>configs/tracker/corner_fusion_wholeframe.py</code>) changes only
    <code>fusion.decision_scope</code> against <code>FINAL</code>. The replay above predicted
    <b>98.80 POT</b> and <b>88.32 PTi</b> pooled P@15; the grids returned <b>98.80</b> and <b>88.32</b>.
    It is the best config in this thread on every cell of both benchmarks, and it is the shipped tracker.
    Two consequences worth keeping: the offline replay is an <em>exact</em> instrument for a fusion-rule
    change, so such a rule never needs a GPU to decide; and the gain is larger than P@15 shows on PTi,
    where br-P@15 rises {wf_br_gain} pp — part of it is currently absorbed by corner naming (§6.6).</p>
  </div>

  <h3>6.6 What the whole-frame rule does <em>not</em> fix</h3>
  <p>On POT the localization gain lands undiluted, because POT's br-gap is zero — it has no
  corner-naming problem. On PTi it does not: br-P@15 goes {wf_br_base} → {wf_br}, while P@15 goes
  {wf_p_base} → {wf_p}, so the br-gap widens {wf_gap_base} → {wf_gap}. <b>The whole remaining PTi
  deficit is corner naming</b>, and one plausible mechanism is this rule itself: handing the entire pose
  to the more confident branch also adopts that branch's <em>naming</em>, and the mask branch cannot name
  corners. The per-corner rule hedged that accidentally. Not yet tested.</p>

  <h3>6.4 The two cases, in real frames</h3>
  <figure>
    {fig_win}
    <figcaption><b>{win_seq} frame {win_frame}: complementarity, working.</b> Per-corner distance to GT —
    keypoint branch {win_kp} px, mask branch {win_mask} px. Each branch is catastrophically wrong about a
    <em>different</em> corner, and the mask branch's covariance says so, so the weighted average keeps the
    good ones: fused {win_fused} px. This is the {c_split}% case.</figcaption>
  </figure>
  <figure>
    {fig_miss}
    <figcaption><b>{miss_seq} frame {miss_frame}: a defect, not a design.</b> Per-corner distance —
    keypoint {miss_kp} px, mask {miss_mask} px: the keypoint branch has the two corners the mask branch
    loses. But the fusion modes are ({miss_modes}) — the keypoint branch produced a pose and
    <em>no covariance</em>, and the rule treats a missing covariance as no opinion and discards the branch
    entirely. That happens on <b>{drop_pct}% of PTi frames</b>. The obvious fallback — let such a branch
    compete under a fixed wide isotropic prior instead of dropping it — was replayed with the rules of
    §6.5 and <b>loses on both datasets</b> (σ = 10 px: POT −0.09, PTi −0.25 pp pooled), so it is measured
    and rejected rather than pending. A branch that cannot say how sure it is turns out to be a branch
    worth ignoring.</figcaption>
  </figure>
  <figure>
    {fig_fusion}
    <figcaption>The rule on frame {frame} of {seq}, with the covariances the run actually fused. Raw
    covariances would give d² = {d2raw} — every corner above the gate; the deployed inflation gives
    {d2dep}, hence modes ({modes}).</figcaption>
  </figure>
</section>

<section>
  <h2>7 · The variants</h2>
  {tbl_thread}
  <ul class="tight">
    <li><b>SWITCH35</b> — take the keypoint pose if RANSAC inliers &gt; 35, else the mask pose. PTi
      {sw_pti15}, POT {sw_pot15}.</li>
    <li><b>ALIGN-PURE</b> (dead end) — quad replaced by a single GN solve anchored at the last pose:
      {ap_drop} pp PTi. The aligner's problem was never its objective, only that it cannot recover.</li>
    <li><b>FUSE-ALIGN</b> — §6's rule with that weak estimator: PTi {fa_pti15}.</li>
    <li><b>FUSE-REDETECT</b> (dead end) — moment re-detection ranked without texture: PTi {rd_pti15},
      br-P@15 {rd_br}, i.e. a naming gap of {rd_gap} pp across {rd_gap5}% of sequences.</li>
    <li><b>FUSE-REG</b> — §4 replaces the estimator, rule unchanged: PTi {reg_pti15} ({reg_gain} pp over
      FUSE-ALIGN).</li>
    <li><b>FUSE-REG-KP</b> — the keypoint homography offered to the registrar as one more proposal,
      verified like any other: PTi {kp_pti15}.</li>
    <li><b>FINAL</b> = FUSE-REG-KP + three search-path settings: <code>sdf_roi_margin = {roi}</code>,
      <code>alias_shortcut</code>, <code>sweep_stop_score = {stop}</code>. PTi {fin_pti15} / {fin_pti5}
      P@5, POT {fin_pot15}.</li>
    <li><b>FINAL+RECHECK</b> = FINAL + §5.3's corrected alias re-check.</li>
  </ul>

  <h3>The texture ablation, at grid scale</h3>
  {tbl_noapp}
  <p class="where">FUSE-REG-NOAPP is FUSE-REG-KP with b = 0. PTi P@15 falls {noapp_drop} pp
  ({kp_pti15} → {noapp_pti15}) while br-P@15 falls only {kp_br} → {noapp_br}: the naming gap opens
  {kp_gap} → {noapp_gap} pp. The object is still found; its corners stop being named.</p>
  {tbl_gap}
</section>

<section>
  <h2>8 · Six benchmarks</h2>
  {tbl_six}
  <div class="card dead">
    <p><b>UCSB is a regression.</b> WOFTSAM2 {ucsb_w} P@15 against SWITCH35 {ucsb_s} and FINAL {ucsb_f},
    with a naming gap of ≈0 for all three — the newer trackers lose the pose, on short close-up sequences
    (≤100 frames, median 50).</p>
  </div>
  <p>Against SWITCH35, FINAL is {fin_p5_delta} pp PTi P@5 and {fin_p15_delta} pp P@15: more accurate where
  it succeeds rather than more robust.</p>
</section>

<section>
  <h2>9 · Open, with the evidence</h2>
  <ul class="tight">
    <li><b>Replace the covariance fudge with the τ_int correction (§3.3).</b> Correlated residuals explain
      a factor ≈{var_short} in variance from first principles; the deployed ×{samh_scale} does not come
      from anywhere.</li>
    <li><b>Drop the branch-drop defect (§6.4).</b> {drop_pct}% of PTi frames discard a keypoint pose that
      exists, because no covariance was produced.</li>
    <li><b>Simplify the score weights (§5.2)</b> to a = 2, b = 1 — as good or better and one fewer tuned
      number.</li>
    <li><b>Consider dropping per-corner granularity (§6.3)</b> unless it stays free: its oracle advantage
      over per-pose selection is {c_gain} pp.</li>
    <li><b>UCSB (§8).</b></li>
  </ul>
</section>

<footer>
  Generated by <code>experiments/build_tracker_cards.py</code>. Sources: <code>eval_results.json</code>
  under <code>OUTPUT_DIRECTORY</code>; <code>figs_facts.json</code> from
  <code>experiments/register_visuals.py</code>; <code>complementarity.json</code> from
  <code>experiments/corner_complementarity.py</code>; offline benchmark CSVs from
  <code>experiments/maskh_bench.py</code>. Code: <code>flatsam2/mask_align.py</code>,
  <code>flatsam2/mask_register.py</code>, <code>flatsam2/corner_covariance.py</code>,
  <code>flatsam2/tracking/corner_fusion.py</code>. Metric: pooled P@τ over all valid annotated frames.
</footer>

</div>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=paths.OUTPUT_DIRECTORY)
    parser.add_argument("--fig-dir", type=Path,
                        default=paths.COMPARISONS_DIRECTORY / "reports" / "figs_pti")
    parser.add_argument("--fig-dir-win", type=Path,
                        default=paths.COMPARISONS_DIRECTORY / "reports" / "figs_comp_win")
    parser.add_argument("--fig-dir-miss", type=Path,
                        default=paths.COMPARISONS_DIRECTORY / "reports" / "figs_comp_miss")
    parser.add_argument("--complementarity", type=Path,
                        default=paths.COMPARISONS_DIRECTORY / "reports" / "complementarity.json")
    parser.add_argument("--bench-csv", type=Path,
                        default=paths.OUTPUT_DIRECTORY / "experiments" / "maskh_bench" / "recheck_v3.csv")
    parser.add_argument("--scope-pot", type=Path,
                        default=paths.COMPARISONS_DIRECTORY / "reports" / "fusion_scope_pot.json")
    parser.add_argument("--scope-pti", type=Path,
                        default=paths.COMPARISONS_DIRECTORY / "reports" / "fusion_scope_pti.json")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent.parent / "docs" / "tracker-cards.html")
    args = parser.parse_args()

    facts = json.loads((args.fig_dir / "figs_facts.json").read_text())
    comp = json.loads(args.complementarity.read_text())
    bench = load_bench(args.bench_csv)
    scope = {
        json.loads(path.read_text())["dataset"]: json.loads(path.read_text())
        for path in (args.scope_pot, args.scope_pti)
    }
    metrics = load_metrics(args.results_dir.expanduser())
    html = build_page(metrics, facts, comp, bench, scope, args.fig_dir,
                      {"win": args.fig_dir_win, "miss": args.fig_dir_miss})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"Wrote {args.out} ({len(html) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
