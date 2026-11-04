"""Collect evaluation results across experiments and print an org-mode table.

Scans OUTPUT_DIRECTORY plus any extra --results-dir roots for eval_results.json
files produced by evaluate.py and summarises p@5 / p@15 per experiment.

Directory layout expected::

    <results_dir>/<config_name>/<dataset>/eval_results.json

Usage examples::

    python collect_results.py
    python collect_results.py --results-dir ~/data/flatsam2/results/
    python collect_results.py --results-dir ~/data/flatsam2/results/ --results-dir ./results
    python collect_results.py --original-pt   # show PT instead of PTi
    python collect_results.py --alias-config configs/aliases/cvpr.py
    python collect_results.py --html-out results_summary.html
    python collect_results.py --table-report-out results_table_report.html
    python collect_results.py --table-report-out out.html --views-config configs/reports/example_views.py
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from flatsam2.config_aliases import (
    get_config_display_name,
    load_config_aliases_from_file,
)
from flatsam2.path_config import paths
from flatsam2.results_table_report import (
    build_base_table_rows,
    build_metric_columns,
    build_report_payload,
    load_report_spec_from_file,
    write_table_report_html,
)
from flatsam2.registry import DATASETS
from flatsam2.sequences import get_sequences

# Derive label strings and short-label map directly from the registry so that
# adding a new dataset variant here requires no changes to this file.
_LABEL: dict[str, str] = {ds.key: ds.label for ds in set(DATASETS.values())}

# Default: POT-210 + PlanarTrackInitfix (improved GT)
_DEFAULT_DATASETS = (_LABEL["pot"], _LABEL["planartrack-initfix"])
# Alternative: POT-210 + original PlanarTrack
_ORIGINAL_PT_DATASETS = (_LABEL["pot"], _LABEL["planartrack"])


def find_eval_files(
    results_dirs: tuple[Path, ...],
    dataset_dir_names: tuple[str, ...],
) -> list[tuple[str, str, Path]]:
    """Find all eval_results.json files under result roots.

    Args:
        results_dirs: Top-level results directories.
            Each root is expected to follow
            ``<results_dir>/<config>/<dataset>/eval_results.json``.
        dataset_dir_names: Subdirectory names to scan for each dataset.

    Returns:
        List of (config_name, dataset_label, json_path) tuples.
    """
    found = []
    for results_dir in results_dirs:
        for config_dir in sorted(results_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            config_name = config_dir.name
            for ds_dir_name in dataset_dir_names:
                eval_file = config_dir / ds_dir_name / "eval_results.json"
                if eval_file.exists():
                    found.append((config_name, ds_dir_name, eval_file))

    return found


def load_metrics(
    json_path: Path,
    expected_sequences: Optional[set[str]] = None,
) -> tuple[dict, Optional[int]]:
    """Load aggregate metrics and failed-sequence count from eval_results.json.

    Args:
        json_path: Path to eval_results.json.

    Returns:
        Tuple ``(aggregate_metrics, num_failed)`` where ``num_failed`` is
        ``None`` when failure information cannot be inferred.

    Raises:
        ValueError: if the file is missing expected keys.
    """
    with json_path.open() as f:
        data = json.load(f)

    agg = data.get("aggregate")
    if agg is None:
        raise ValueError(f"No 'aggregate' key in {json_path}")

    num_failed: Optional[int] = None
    sequence_entries = data.get("sequences")
    if isinstance(sequence_entries, list) and expected_sequences is not None:
        evaluated_sequences = {
            entry.get("sequence")
            for entry in sequence_entries
            if isinstance(entry, dict) and isinstance(entry.get("sequence"), str)
        }
        num_failed = len(expected_sequences - evaluated_sequences)
    elif isinstance(failed_entries := data.get("failed"), list):
        num_failed = len(failed_entries)
    elif isinstance(data.get("failed_sequences"), list):
        num_failed = len(data["failed_sequences"])
    elif isinstance(agg.get("num_failed"), (int, float)):
        num_failed = int(agg["num_failed"])
    elif isinstance(data.get("num_failed"), (int, float)):
        num_failed = int(data["num_failed"])
    elif isinstance(agg.get("num_total_sequences"), (int, float)) and isinstance(
        agg.get("num_sequences"), (int, float)
    ):
        total = int(agg["num_total_sequences"])
        done = int(agg["num_sequences"])
        if total >= done:
            num_failed = total - done

    return agg, num_failed


# Map label → short_label, derived from the registry
_DATASET_SHORT_LABELS: dict[str, str] = {
    ds.label: ds.short_label for ds in set(DATASETS.values())
}

# Map label -> canonical dataset key
_DATASET_KEY_BY_LABEL: dict[str, str] = {
    ds.label: ds.key for ds in set(DATASETS.values())
}


def resolve_expected_sequences(
    dataset_dir_names: tuple[str, ...],
) -> dict[str, Optional[set[str]]]:
    """Resolve expected sequence sets for each dataset label.

    Args:
        dataset_dir_names: Ordered dataset directory names used in results folders.

    Returns:
        Mapping ``dataset_label -> expected sequence set``. Values are ``None``
        when the expected sequence set cannot be determined.
    """
    expected: dict[str, Optional[set[str]]] = {}

    for ds_label in dataset_dir_names:
        ds_key = _DATASET_KEY_BY_LABEL.get(ds_label)
        if ds_key is None or ds_key not in DATASETS:
            expected[ds_label] = None
            continue

        ds_info = DATASETS[ds_key]
        base_dir = getattr(paths, ds_info.base_dir_attr)
        if not base_dir.exists():
            expected[ds_label] = None
            continue

        try:
            expected[ds_label] = set(get_sequences(ds_key, base_dir))
        except Exception:
            expected[ds_label] = None

    return expected


def pivot_rows(
    rows: list[tuple[str, str, str, float, float, Optional[int]]],
    dataset_dir_names: tuple[str, ...],
) -> list[list[Optional[float] | Optional[int] | str]]:
    """Pivot rows into one row per real config.

    Args:
        rows: List of
            (method_display, real_config, dataset_label, p@5, p@15, num_failed)
            tuples.
        dataset_dir_names: Ordered dataset directory names that define the columns.

    Returns:
        List of rows, each being
        [method_display, ds1_p5, ds1_p15, ds1_failed,
         ds2_p5, ds2_p15, ds2_failed, ..., real_config],
        sorted by method and real config name. Missing values are represented as
        None.
    """
    data: dict[str, dict[str, object]] = {}
    for method_display, real_config, dataset, p5, p15, num_failed in rows:
        entry = data.setdefault(real_config, {"method": method_display, "scores": {}})
        entry["method"] = method_display
        entry["scores"][dataset] = (p5, p15, num_failed)

    pivoted = []
    for real_config, entry in sorted(
        data.items(),
        key=lambda item: (str(item[1]["method"]), item[0]),
    ):
        row: list = [str(entry["method"])]
        for ds in dataset_dir_names:
            score = entry["scores"].get(ds)
            row.append(score[0] if score else None)
            row.append(score[1] if score else None)
            row.append(score[2] if score else None)
        row.append(real_config)
        pivoted.append(row)
    return pivoted


def print_org_table(
    rows: list[tuple[str, str, str, float, float, Optional[int]]],
    dataset_dir_names: tuple[str, ...],
) -> None:
    """Print results as an org-mode table with one row per method.

    Columns: method | <DS1> p@5 | <DS1> p@15 | ... | config | failed (joined)

    Args:
        rows: List of
            (method_display, real_config, dataset_label, p@5, p@15, num_failed)
            tuples.
        dataset_dir_names: Ordered dataset directory names that define the columns.
    """
    if not rows:
        print("No results found.")
        return

    pivoted = pivot_rows(rows, dataset_dir_names)

    def fmt_val(v: Optional[float]) -> str:
        return f"{v:.1f}" if v is not None else "-"

    def fmt_failed(v: Optional[int]) -> str:
        return str(v) if v is not None else "?"

    # Build header labels: p@5 / p@15 per dataset, then combined failed counts.
    headers = ["method"]
    for ds in dataset_dir_names:
        label = _DATASET_SHORT_LABELS.get(ds, ds)
        headers += [f"{label} p@5", f"{label} p@15"]
    headers.append("config")
    headers.append("failed")

    # Compute column widths
    col_w = [len(h) for h in headers]
    for prow in pivoted:
        metric_cells = []
        failed_cells = []
        for ds_idx in range(len(dataset_dir_names)):
            start = 1 + 3 * ds_idx
            metric_cells.append(fmt_val(prow[start]))
            metric_cells.append(fmt_val(prow[start + 1]))
            failed_cells.append(fmt_failed(prow[start + 2]))

        combined_failed = " / ".join(failed_cells)
        cells = [str(prow[0])] + metric_cells + [str(prow[-1]), combined_failed]
        for i, c in enumerate(cells):
            col_w[i] = max(col_w[i], len(c))

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for idx, (cell, w) in enumerate(zip(cells, col_w)):
            if idx == 0:
                parts.append(f" {cell:<{w}} ")
            else:
                parts.append(f" {cell:>{w}} ")
        return "|" + "|".join(parts) + "|"

    sep = "|-" + "-+-".join("-" * w for w in col_w) + "-|"

    print(fmt_row(headers))
    print(sep)
    for prow in pivoted:
        metric_cells = []
        failed_cells = []
        for ds_idx in range(len(dataset_dir_names)):
            start = 1 + 3 * ds_idx
            metric_cells.append(fmt_val(prow[start]))
            metric_cells.append(fmt_val(prow[start + 1]))
            failed_cells.append(fmt_failed(prow[start + 2]))

        combined_failed = " / ".join(failed_cells)
        cells = [str(prow[0])] + metric_cells + [str(prow[-1]), combined_failed]
        print(fmt_row(cells))


def build_segment_plot_rows(
    rows: list[tuple[str, str, str, float, float, Optional[int]]],
    dataset_dir_names: tuple[str, ...],
) -> list[dict[str, object]]:
    """Build plot rows for a 2-dataset segment plot.

    Each method is represented by a line segment where the segment endpoints are:
      - p@5:  (x = ds1 p@5,  y = ds2 p@5)
      - p@15: (x = ds1 p@15, y = ds2 p@15)

    Args:
        rows: List of
            (method_display, real_config, dataset_label, p@5, p@15, num_failed)
            tuples.
        dataset_dir_names: Ordered dataset labels. The first two define x/y axes.

    Returns:
        List of dictionaries ready for JSON serialization.
    """
    if len(dataset_dir_names) < 2:
        return []

    pivoted = pivot_rows(rows, dataset_dir_names)
    records: list[dict[str, object]] = []

    for prow in pivoted:
        ds1_p5 = prow[1]
        ds1_p15 = prow[2]
        ds1_failed = prow[3]
        ds2_p5 = prow[4]
        ds2_p15 = prow[5]
        ds2_failed = prow[6]

        if not all(
            isinstance(v, (int, float)) for v in (ds1_p5, ds1_p15, ds2_p5, ds2_p15)
        ):
            continue

        records.append(
            {
                "method": str(prow[0]),
                "config": str(prow[-1]),
                "x_p5": float(ds1_p5),
                "x_p15": float(ds1_p15),
                "y_p5": float(ds2_p5),
                "y_p15": float(ds2_p15),
                "x_failed": int(ds1_failed) if isinstance(ds1_failed, int) else None,
                "y_failed": int(ds2_failed) if isinstance(ds2_failed, int) else None,
            }
        )

    return records


def write_interactive_plot_html(
    rows: list[tuple[str, str, str, float, float, Optional[int]]],
    dataset_dir_names: tuple[str, ...],
    output_path: Path,
) -> None:
    """Write a self-contained interactive HTML segment plot.

    Args:
        rows: List of
            (method_display, real_config, dataset_label, p@5, p@15, num_failed)
            tuples.
        dataset_dir_names: Ordered dataset labels. First is x-axis, second y-axis.
        output_path: Output HTML file path.

    Raises:
        ValueError: If fewer than two datasets are available.
    """
    if len(dataset_dir_names) < 2:
        raise ValueError(
            f"Need at least two datasets for plotting, got {len(dataset_dir_names)}"
        )

    plot_rows = build_segment_plot_rows(rows, dataset_dir_names)
    x_label = _DATASET_SHORT_LABELS.get(dataset_dir_names[0], dataset_dir_names[0])
    y_label = _DATASET_SHORT_LABELS.get(dataset_dir_names[1], dataset_dir_names[1])
    plot_rows_json = json.dumps(plot_rows, ensure_ascii=True)

    html_content = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Evaluation Segment Plot</title>
  <style>
    :root {{
      --bg: #f7fafc;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --grid: #cbd5e1;
      --axis: #334155;
      --border: #e2e8f0;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top right, #e0f2fe 0%, rgba(224, 242, 254, 0) 45%),
        radial-gradient(circle at bottom left, #dcfce7 0%, rgba(220, 252, 231, 0) 35%),
        var(--bg);
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      box-shadow: 0 8px 22px rgba(2, 6, 23, 0.06);
    }}
    h1 {{
      font-size: 18px;
      margin: 0 0 10px 0;
      letter-spacing: 0.02em;
    }}
    p {{
      margin: 8px 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin: 10px 0 12px;
    }}
    button {{
      border: 1px solid #94a3b8;
      background: #f8fafc;
      color: var(--text);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 13px;
      cursor: pointer;
    }}
    button:hover {{
      background: #eef2ff;
    }}
    label {{
      font-size: 13px;
      color: var(--muted);
      display: inline-flex;
      gap: 6px;
      align-items: center;
    }}
    #methodList {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 6px 10px;
      margin-top: 8px;
    }}
    .plot-wrap {{
      width: 100%;
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: white;
    }}
    svg {{
      width: 100%;
      min-width: 760px;
      height: 620px;
      display: block;
    }}
    .footnote {{
      font-size: 12px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class=\"container\">
    <div class=\"panel\">
      <h1>Experiment Comparison Segment Plot</h1>
      <p>
        Segment start is p@5, segment end is p@15. X axis: {x_label}, Y axis: {y_label}.
      </p>
      <div class=\"controls\">
        <button id=\"allBtn\" type=\"button\">Select all</button>
        <button id=\"noneBtn\" type=\"button\">Select none</button>
        <label><input id=\"labelsToggle\" type=\"checkbox\" checked /> Show labels</label>
      </div>
      <div id=\"methodList\"></div>
      <div class=\"plot-wrap\">
        <svg id=\"plot\" viewBox=\"0 0 980 620\" role=\"img\" aria-label=\"segment plot\"></svg>
      </div>
      <p class=\"footnote\">
        Hover a segment or endpoint to inspect values. Endpoint circles: hollow = p@5, filled = p@15.
      </p>
    </div>
  </div>
  <script>
    const rows = {plot_rows_json};
    const state = {{
      visibleConfigs: new Set(rows.map((r) => r.config)),
      showLabels: true,
    }};

    const svg = document.getElementById('plot');
    const methodList = document.getElementById('methodList');

    function colorForIndex(index) {{
      const hue = (index * 137.508) % 360;
      return `hsl(${{hue.toFixed(1)}}, 70%, 42%)`;
    }}

    function createMethodControls() {{
      methodList.innerHTML = '';
      rows.forEach((row) => {{
        const wrapper = document.createElement('label');
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = true;
        cb.dataset.config = row.config;
        cb.addEventListener('change', () => {{
          if (cb.checked) {{
            state.visibleConfigs.add(row.config);
          }} else {{
            state.visibleConfigs.delete(row.config);
          }}
          render();
        }});

        const txt = document.createElement('span');
        txt.textContent = `${{row.method}} (${{row.config}})`;
        wrapper.appendChild(cb);
        wrapper.appendChild(txt);
        methodList.appendChild(wrapper);
      }});
    }}

    function line(x1, y1, x2, y2, stroke, width, dash = '') {{
      const el = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      el.setAttribute('x1', x1);
      el.setAttribute('y1', y1);
      el.setAttribute('x2', x2);
      el.setAttribute('y2', y2);
      el.setAttribute('stroke', stroke);
      el.setAttribute('stroke-width', width);
      if (dash) {{
        el.setAttribute('stroke-dasharray', dash);
      }}
      return el;
    }}

    function text(x, y, label, size = 12, anchor = 'middle', fill = '#334155') {{
      const el = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      el.setAttribute('x', x);
      el.setAttribute('y', y);
      el.setAttribute('font-size', size);
      el.setAttribute('text-anchor', anchor);
      el.setAttribute('fill', fill);
      el.textContent = label;
      return el;
    }}

    function circle(x, y, r, stroke, fill) {{
      const el = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      el.setAttribute('cx', x);
      el.setAttribute('cy', y);
      el.setAttribute('r', r);
      el.setAttribute('stroke', stroke);
      el.setAttribute('stroke-width', 2);
      el.setAttribute('fill', fill);
      return el;
    }}

    function render() {{
      svg.innerHTML = '';

      const visible = rows.filter((r) => state.visibleConfigs.has(r.config));
      const width = 980;
      const height = 620;
      const margin = {{ top: 30, right: 24, bottom: 56, left: 72 }};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;

      const xToPx = (x) => margin.left + (x / 100.0) * innerW;
      const yToPx = (y) => margin.top + (1.0 - y / 100.0) * innerH;

      for (let tick = 0; tick <= 100; tick += 10) {{
        const x = xToPx(tick);
        const y = yToPx(tick);
        svg.appendChild(line(x, margin.top, x, height - margin.bottom, '#e2e8f0', 1));
        svg.appendChild(line(margin.left, y, width - margin.right, y, '#e2e8f0', 1));
        svg.appendChild(text(x, height - margin.bottom + 18, String(tick), 11, 'middle', '#64748b'));
        svg.appendChild(text(margin.left - 12, y + 4, String(tick), 11, 'end', '#64748b'));
      }}

      svg.appendChild(
        line(
          margin.left,
          height - margin.bottom,
          width - margin.right,
          height - margin.bottom,
          '#0f172a',
          1.5
        )
      );
      svg.appendChild(line(margin.left, margin.top, margin.left, height - margin.bottom, '#0f172a', 1.5));

      svg.appendChild(text(width / 2, height - 16, '{x_label} (score)', 13));

      const yAxisLabel = text(18, height / 2, '{y_label} (score)', 13, 'middle');
      yAxisLabel.setAttribute('transform', `rotate(-90, 18, ${{height / 2}})`);
      svg.appendChild(yAxisLabel);

      visible.forEach((row, index) => {{
        const color = colorForIndex(index);
        const x1 = xToPx(row.x_p5);
        const y1 = yToPx(row.y_p5);
        const x2 = xToPx(row.x_p15);
        const y2 = yToPx(row.y_p15);

        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        const tip = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        tip.textContent = [
          `Method: ${{row.method}}`,
          `Config: ${{row.config}}`,
          '{x_label} p@5: ' + row.x_p5.toFixed(1),
          '{y_label} p@5: ' + row.y_p5.toFixed(1),
          '{x_label} p@15: ' + row.x_p15.toFixed(1),
          '{y_label} p@15: ' + row.y_p15.toFixed(1),
          '{x_label} failed: ' + (row.x_failed ?? '?'),
          '{y_label} failed: ' + (row.y_failed ?? '?'),
        ].join('\\n');

        g.appendChild(line(x1, y1, x2, y2, color, 2.5));
        g.appendChild(circle(x1, y1, 5, color, '#ffffff'));
        g.appendChild(circle(x2, y2, 5, color, color));
        g.appendChild(tip);

        if (state.showLabels) {{
          g.appendChild(text(x2 + 7, y2 - 7, row.method, 11, 'start', color));
        }}

        svg.appendChild(g);
      }});

      if (visible.length === 0) {{
        svg.appendChild(text(width / 2, height / 2, 'No methods selected', 18, 'middle', '#64748b'));
      }}
    }}

    document.getElementById('allBtn').addEventListener('click', () => {{
      state.visibleConfigs = new Set(rows.map((r) => r.config));
      methodList.querySelectorAll('input[type=checkbox]').forEach((cb) => {{
        cb.checked = true;
      }});
      render();
    }});

    document.getElementById('noneBtn').addEventListener('click', () => {{
      state.visibleConfigs = new Set();
      methodList.querySelectorAll('input[type=checkbox]').forEach((cb) => {{
        cb.checked = false;
      }});
      render();
    }});

    document.getElementById('labelsToggle').addEventListener('change', (event) => {{
      state.showLabels = event.target.checked;
      render();
    }});

    createMethodControls();
    render();
  </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")


def find_eval_file_for_config(
    config_name: str,
    dataset_label: str,
    results_dirs: tuple[Path, ...],
) -> Optional[Path]:
    """Return first matching eval_results.json for a config+dataset pair.

    Args:
        config_name: Config directory name (e.g. ``fast_kitti``).
        dataset_label: Dataset subdirectory label (e.g. ``PTi``).
        results_dirs: Ordered results root directories to search.

    Returns:
        Path to eval_results.json or ``None`` if not found.
    """
    for root in results_dirs:
        candidate = root / config_name / dataset_label / "eval_results.json"
        if candidate.exists():
            return candidate
    return None


def print_attribute_comparison(
    config_a: str,
    config_b: str,
    path_a: Path,
    path_b: Path,
) -> None:
    """Print per-attribute p@5/p@15 comparison between two configs.

    Args:
        config_a: Display name for first config.
        config_b: Display name for second config.
        path_a: Path to eval_results.json for first config.
        path_b: Path to eval_results.json for second config.
    """
    data_a = json.loads(path_a.read_text(encoding="utf-8"))
    data_b = json.loads(path_b.read_text(encoding="utf-8"))

    agg_a: dict = data_a.get("attribute_aggregates", {})
    agg_b: dict = data_b.get("attribute_aggregates", {})

    if not agg_a and not agg_b:
        print(
            "Neither eval_results.json contains 'attribute_aggregates'. "
            "Re-run evaluate.py with --attributes-file."
        )
        return
    if not agg_a:
        print(f"WARNING: {path_a} has no 'attribute_aggregates'.")
    if not agg_b:
        print(f"WARNING: {path_b} has no 'attribute_aggregates'.")

    all_attrs = list(dict.fromkeys(list(agg_a.keys()) + list(agg_b.keys())))

    # Build display rows: one per (attr, group)
    # Columns: attribute, group, n, a_p5, b_p5, Δp5, a_p15, b_p15, Δp15
    def fmt_val(v: Optional[float]) -> str:
        return f"{v:.1f}" if v is not None else "-"

    def fmt_delta(d: Optional[float]) -> str:
        if d is None:
            return "-"
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}"

    rows = []
    for attr in all_attrs:
        entry_a = agg_a.get(attr, {})
        entry_b = agg_b.get(attr, {})
        for group in ("with", "without"):
            ga = entry_a.get(group)
            gb = entry_b.get(group)
            if ga is None and gb is None:
                continue
            n = str(ga["num_sequences"]) if ga else (str(gb["num_sequences"]) if gb else "-")
            a5  = ga["p@5"]  if ga else None
            a15 = ga["p@15"] if ga else None
            b5  = gb["p@5"]  if gb else None
            b15 = gb["p@15"] if gb else None
            d5  = (b5  - a5)  if (a5  is not None and b5  is not None) else None
            d15 = (b15 - a15) if (a15 is not None and b15 is not None) else None
            rows.append([
                attr, group, n,
                fmt_val(a5), fmt_val(b5), fmt_delta(d5),
                fmt_val(a15), fmt_val(b15), fmt_delta(d15),
            ])

    if not rows:
        print("No attribute data to compare.")
        return

    # Truncate long config names for column headers
    max_cfg_len = 14
    def short(name: str) -> str:
        return name if len(name) <= max_cfg_len else name[:max_cfg_len - 1] + "…"

    sa, sb = short(config_a), short(config_b)
    headers = [
        "attribute", "group", "n",
        f"{sa} p@5", f"{sb} p@5", "Δp@5",
        f"{sa} p@15", f"{sb} p@15", "Δp@15",
    ]

    col_w = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_w[i] = max(col_w[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for i, (cell, w) in enumerate(zip(cells, col_w)):
            if i <= 1:
                parts.append(f" {cell:<{w}} ")
            else:
                parts.append(f" {cell:>{w}} ")
        return "|" + "|".join(parts) + "|"

    sep = "|-" + "-+-".join("-" * w for w in col_w) + "-|"

    print(f"\nAttribute comparison: {config_a}  vs  {config_b}")
    print(fmt_row(headers))
    print(sep)
    prev_attr = None
    for row in rows:
        if prev_attr is not None and row[0] != prev_attr:
            print(sep)
        print(fmt_row(row))
        prev_attr = row[0]


def main(argv=None) -> int:
    default_alias_config = Path("configs/aliases/default.py")
    default_alias_config = (
        default_alias_config if default_alias_config.exists() else None
    )

    parser = argparse.ArgumentParser(
        description="Collect evaluate.py results and print an org-mode summary table"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Extra top-level results directory. May be passed multiple times. "
            "Default search always includes path_config.py OUTPUT_DIRECTORY."
        ),
    )
    parser.add_argument(
        "--best-rotation",
        action="store_true",
        help=(
            "Report rotation-invariant (best-rotation) p@5/p@15 instead of the "
            "standard metrics. Ignores the planar symmetry/flip ambiguity and "
            "scores accuracy only. Requires eval_results.json produced with "
            "best-rotation metrics (re-run evaluate.py if missing)."
        ),
    )
    parser.add_argument(
        "--original-pt",
        action="store_true",
        help=(
            "Show original PlanarTrack (PT) results instead of the improved "
            "initfix variant (PTi). Default is to show POT + PTi."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="KEY",
        default=None,
        help=(
            "Dataset keys to show as column groups, in order (e.g. "
            "'pot pti poic tmt ucsb mpot'). Overrides --original-pt. "
            f"Accepted: {', '.join(sorted(DATASETS))}."
        ),
    )
    parser.add_argument(
        "--alias-config",
        type=Path,
        default=default_alias_config,
        help=(
            "Python file defining config-name display aliases (expects 'aliases' "
            "mapping). Defaults to configs/aliases/default.py when present."
        ),
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help=(
            "Optional path for a self-contained interactive HTML segment plot. "
            "The first dataset is used for x-axis and the second for y-axis."
        ),
    )
    parser.add_argument(
        "--table-report-out",
        type=Path,
        default=None,
        help=(
            "Optional path for standalone interactive HTML table report with "
            "named views/slides and live editing controls."
        ),
    )
    parser.add_argument(
        "--views-config",
        type=Path,
        default=None,
        help=(
            "Optional Python file defining ReportSpec for --table-report-out. "
            "If omitted, report contains one 'All results' view."
        ),
    )
    parser.add_argument(
        "--compare-configs",
        nargs=2,
        metavar=("CONFIG_A", "CONFIG_B"),
        default=None,
        help=(
            "Print per-attribute p@5/p@15 comparison between two configs. "
            "Values must be config directory names (e.g. fast_kitti baseline). "
            "Reads pre-computed 'attribute_aggregates' from eval_results.json."
        ),
    )
    parser.add_argument(
        "--compare-dataset",
        type=str,
        default=None,
        help=(
            "Dataset label to use for --compare-configs (e.g. 'PTi'). "
            "Defaults to the first PlanarTrack-initfix dataset in the active set."
        ),
    )
    args = parser.parse_args(argv)

    if args.views_config is not None and args.table_report_out is None:
        print("WARNING: --views-config ignored without --table-report-out")

    if args.datasets is not None:
        unknown = [key for key in args.datasets if key not in DATASETS]
        if unknown:
            print(
                f"ERROR: unknown dataset key(s): {', '.join(unknown)}. "
                f"Expected any of: {', '.join(sorted(DATASETS))}."
            )
            return 1
        # dict.fromkeys keeps the requested order while dropping alias duplicates
        # (e.g. 'pti' and 'planartrack-initfix' both resolve to one label).
        dataset_dir_names = tuple(
            dict.fromkeys(DATASETS[key].label for key in args.datasets)
        )
    else:
        dataset_dir_names = (
            _ORIGINAL_PT_DATASETS if args.original_pt else _DEFAULT_DATASETS
        )

    requested_results_dirs = [paths.OUTPUT_DIRECTORY.expanduser()]
    if args.results_dir is not None:
        requested_results_dirs.extend(path.expanduser() for path in args.results_dir)

    results_dirs = tuple(dict.fromkeys(requested_results_dirs))
    existing_results_dirs = tuple(path for path in results_dirs if path.is_dir())

    for path in results_dirs:
        if not path.is_dir():
            print(f"WARNING: results directory not found, skipping: {path}")

    if not existing_results_dirs:
        joined = ", ".join(str(path) for path in results_dirs)
        print(f"ERROR: No results directories found: {joined}")
        return 1

    try:
        config_aliases = load_config_aliases_from_file(args.alias_config)
    except Exception as e:
        print(f"ERROR: Could not load alias config: {e}")
        return 1

    eval_files = find_eval_files(existing_results_dirs, dataset_dir_names)
    expected_sequences_by_dataset = resolve_expected_sequences(dataset_dir_names)

    if not eval_files:
        joined = ", ".join(str(path) for path in existing_results_dirs)
        print(f"No eval_results.json files found under {joined}")
        return 0

    # Collect standard and best-rotation rows in one pass; the report carries
    # both (toggle button), while stdout/segment-plot honour --best-rotation.
    rows_std: list = []
    rows_br: list = []
    missing_br = 0
    for config_name, dataset_label, json_path in eval_files:
        try:
            agg, num_failed = load_metrics(
                json_path,
                expected_sequences=expected_sequences_by_dataset.get(dataset_label),
            )
            method_name = get_config_display_name(config_name, config_aliases)
            rows_std.append((
                method_name, config_name, dataset_label,
                float(agg["overall_p@5"]), float(agg["overall_p@15"]), num_failed,
            ))
            if "overall_br_p@5" in agg and "overall_br_p@15" in agg:
                rows_br.append((
                    method_name, config_name, dataset_label,
                    float(agg["overall_br_p@5"]), float(agg["overall_br_p@15"]),
                    num_failed,
                ))
            else:
                missing_br += 1
        except Exception as e:
            print(f"WARNING: could not load {json_path}: {e}")

    if args.best_rotation:
        if not rows_br:
            print(
                "ERROR: --best-rotation requested but no eval_results.json has "
                "best-rotation metrics. Re-run evaluate.py."
            )
            return 1
        if missing_br:
            print(
                f"WARNING: {missing_br} result file(s) lack best-rotation metrics "
                "and are omitted; re-run evaluate.py for them."
            )
        print("Metric: best-rotation (rotation-invariant) p@5 / p@15\n")

    rows = rows_br if args.best_rotation else rows_std

    print_org_table(rows, dataset_dir_names)

    if args.html_out is not None:
        # The segment plot is inherently 2-dimensional (one dataset per axis).
        if len(dataset_dir_names) != 2:
            print(
                "ERROR: --html-out needs exactly two datasets (x and y axis); "
                f"got {len(dataset_dir_names)}: {', '.join(dataset_dir_names)}."
            )
            return 1
        try:
            write_interactive_plot_html(rows, dataset_dir_names, args.html_out)
            print(f"Wrote interactive plot: {args.html_out}")
        except Exception as e:
            print(f"ERROR: could not write HTML plot: {e}")
            return 1

    if args.table_report_out is not None:
        try:
            report_spec = load_report_spec_from_file(args.views_config)
            pivoted_rows = pivot_rows(rows_std, dataset_dir_names)
            br_pivoted_rows = (
                pivot_rows(rows_br, dataset_dir_names) if rows_br else None
            )
            base_columns, base_rows = build_base_table_rows(
                pivoted_rows,
                dataset_dir_names,
                _DATASET_SHORT_LABELS,
                br_pivoted_rows=br_pivoted_rows,
            )
            metric_columns = build_metric_columns(
                dataset_dir_names, _DATASET_SHORT_LABELS
            )
            payload = build_report_payload(
                report_spec, base_columns, base_rows, metric_columns=metric_columns
            )
            write_table_report_html(payload, args.table_report_out)
            print(f"Wrote standalone table report: {args.table_report_out}")
        except Exception as e:
            print(f"ERROR: could not write table report: {e}")
            return 1

    if args.compare_configs is not None:
        config_a, config_b = args.compare_configs

        # Pick dataset: explicit > first PTi in active set > first in active set
        if args.compare_dataset is not None:
            compare_ds = args.compare_dataset
        else:
            pti_label = _LABEL.get("planartrack-initfix", "")
            compare_ds = pti_label if pti_label in dataset_dir_names else dataset_dir_names[0]

        path_a = find_eval_file_for_config(config_a, compare_ds, existing_results_dirs)
        path_b = find_eval_file_for_config(config_b, compare_ds, existing_results_dirs)

        ok = True
        for cfg, path in ((config_a, path_a), (config_b, path_b)):
            if path is None:
                print(
                    f"ERROR: eval_results.json not found for config={cfg!r} "
                    f"dataset={compare_ds!r} under {existing_results_dirs}"
                )
                ok = False
        if ok:
            name_a = get_config_display_name(config_a, config_aliases)
            name_b = get_config_display_name(config_b, config_aliases)
            print_attribute_comparison(name_a, name_b, path_a, path_b)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
