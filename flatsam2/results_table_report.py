"""Standalone HTML report generation for collect_results table output."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional


RowFilter = Callable[[dict[str, object]], bool]


@dataclass
class SortRule:
    """Sort rule for one table column.

    Args:
        column: Column name to sort by.
        descending: Whether to sort in descending order.
    """

    column: str
    descending: bool = False


def asc(column: str) -> SortRule:
    """Create ascending sort rule for ``column``."""
    return SortRule(column=column, descending=False)


def desc(column: str) -> SortRule:
    """Create descending sort rule for ``column``."""
    return SortRule(column=column, descending=True)


@dataclass
class ViewSpec:
    """One named table view shown in report.

    Args:
        key: Stable identifier for view.
        title: Human-readable title shown in report.
        note: Optional short note rendered under title.
        include_configs: Optional allow-list of config directory names.
        exclude_configs: Optional deny-list of config directory names.
        include_methods: Optional allow-list of method display names.
        exclude_methods: Optional deny-list of method display names.
        filters: Optional custom row predicates.
        sort_by: Optional ordered sort rules.
        row_order: Optional explicit config order to place first.
        rename_configs: Optional map ``config -> new method label``.
        rename_methods: Optional map ``method label -> new method label``.
        columns: Optional ordered subset of columns to show.
        diff_mode: Whether diff mode starts enabled for this view.
        baseline_config: Config directory name used as initial diff baseline.
    """

    key: str
    title: str
    note: str = ""
    include_configs: Optional[list[str]] = None
    exclude_configs: Optional[list[str]] = None
    include_methods: Optional[list[str]] = None
    exclude_methods: Optional[list[str]] = None
    filters: list[RowFilter] = field(default_factory=list)
    sort_by: list[SortRule] = field(default_factory=list)
    row_order: Optional[list[str]] = None
    rename_configs: dict[str, str] = field(default_factory=dict)
    rename_methods: dict[str, str] = field(default_factory=dict)
    columns: Optional[list[str]] = None
    diff_mode: bool = False
    baseline_config: Optional[str] = None


@dataclass
class ReportSpec:
    """Top-level report specification loaded from Python file.

    Args:
        title: Report page title.
        subtitle: Optional subtitle displayed below title.
        views: Ordered list of named views.
    """

    title: str = "Evaluation Table Report"
    subtitle: str = ""
    views: list[ViewSpec] = field(default_factory=list)


def load_report_spec_from_file(spec_path: Optional[Path]) -> ReportSpec:
    """Load report spec from Python file.

    Args:
        spec_path: Path to Python file defining ``report`` or ``spec`` variable.

    Returns:
        Parsed report specification. If ``spec_path`` is ``None``, returns
        empty default specification.

    Raises:
        FileNotFoundError: If ``spec_path`` does not exist.
        ValueError: If file cannot be loaded or spec value is invalid.
    """
    if spec_path is None:
        return ReportSpec()

    spec_path = Path(spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Report spec file not found: {spec_path}")

    spec = importlib.util.spec_from_file_location("results_report_spec_module", spec_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load report spec from {spec_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    value = getattr(module, "report", None)
    if value is None:
        value = getattr(module, "spec", None)
    if value is None:
        raise ValueError(
            f"Report spec file {spec_path} must define 'report' (or legacy 'spec')"
        )
    if not isinstance(value, ReportSpec):
        raise ValueError(
            f"Report spec in {spec_path} must be ReportSpec, got {type(value)}"
        )

    _validate_report_spec(value)
    return value


def _validate_report_spec(report: ReportSpec) -> None:
    """Validate report spec fields.

    Args:
        report: Report specification.

    Raises:
        ValueError: If report contains invalid values.
    """
    keys: set[str] = set()
    for view in report.views:
        if not view.key:
            raise ValueError("Each ViewSpec.key must be non-empty")
        if view.key in keys:
            raise ValueError(f"Duplicate ViewSpec.key: {view.key}")
        keys.add(view.key)
        if not view.title:
            raise ValueError(f"ViewSpec.title must be non-empty for key={view.key}")
        for func in view.filters:
            if not callable(func):
                raise ValueError(
                    f"Every filter in view={view.key} must be callable, got {type(func)}"
                )


def build_base_table_rows(
    pivoted_rows: list[list[object]],
    dataset_dir_names: tuple[str, ...],
    dataset_short_labels: Mapping[str, str],
    br_pivoted_rows: Optional[list[list[object]]] = None,
) -> tuple[list[str], list[dict[str, object]]]:
    """Build row dictionaries used by HTML report.

    Args:
        pivoted_rows: Output rows from ``pivot_rows`` (standard metrics).
        dataset_dir_names: Ordered dataset labels used in summary table.
        dataset_short_labels: Mapping ``dataset label -> short display label``.
        br_pivoted_rows: Optional best-rotation ``pivot_rows`` output. When
            given, extra ``<ds> br-p@5`` / ``br-p@15`` columns are appended and
            populated (matched to standard rows by config), so the report can
            toggle between standard and rotation-invariant metrics.

    Returns:
        Tuple of ``(columns, rows)`` where each row is JSON-serializable. Any
        best-rotation columns are appended after the standard metric columns and
        before ``config`` / ``failed``.
    """
    columns = ["method"]
    for ds in dataset_dir_names:
        ds_label = dataset_short_labels.get(ds, ds)
        columns.extend([f"{ds_label} p@5", f"{ds_label} p@15"])

    # Best-rotation values keyed by config, then dataset label.
    br_by_config: dict[str, dict[str, tuple[object, object]]] = {}
    if br_pivoted_rows is not None:
        for ds_label in (dataset_short_labels.get(ds, ds) for ds in dataset_dir_names):
            columns.extend([f"{ds_label} br-p@5", f"{ds_label} br-p@15"])
        for prow in br_pivoted_rows:
            config = str(prow[-1])
            per_ds: dict[str, tuple[object, object]] = {}
            for ds_idx, ds in enumerate(dataset_dir_names):
                base = 1 + 3 * ds_idx
                per_ds[dataset_short_labels.get(ds, ds)] = (prow[base], prow[base + 1])
            br_by_config[config] = per_ds

    columns.append("config")
    columns.append("failed")

    rows: list[dict[str, object]] = []
    for prow in pivoted_rows:
        config = str(prow[-1])
        row: dict[str, object] = {
            "method": prow[0],
            "config": prow[-1],
            "__original_method": prow[0],
        }
        failed_parts: list[dict[str, object]] = []
        for ds_idx, ds in enumerate(dataset_dir_names):
            base = 1 + 3 * ds_idx
            ds_label = dataset_short_labels.get(ds, ds)
            row[f"{ds_label} p@5"] = prow[base]
            row[f"{ds_label} p@15"] = prow[base + 1]
            if br_pivoted_rows is not None:
                br5, br15 = br_by_config.get(config, {}).get(ds_label, (None, None))
                row[f"{ds_label} br-p@5"] = br5
                row[f"{ds_label} br-p@15"] = br15
            raw_failed = prow[base + 2]
            failed_val = int(raw_failed) if isinstance(raw_failed, int) else None
            failed_parts.append({"dataset": ds_label, "value": failed_val})

        any_failed = any(
            part["value"] is not None and int(part["value"]) > 0 for part in failed_parts
        )
        row["failed"] = failed_parts if any_failed else []
        rows.append(row)

    return columns, rows


def build_metric_columns(
    dataset_dir_names: tuple[str, ...],
    dataset_short_labels: Mapping[str, str],
) -> dict[str, list[str]]:
    """Return the standard vs best-rotation metric column names for the toggle.

    Args:
        dataset_dir_names: Ordered dataset labels.
        dataset_short_labels: Mapping ``dataset label -> short display label``.

    Returns:
        Mapping with ``"standard"`` and ``"best_rotation"`` keys, each listing
        the p@5/p@15 column names for that metric variant.
    """
    standard: list[str] = []
    best_rotation: list[str] = []
    for ds in dataset_dir_names:
        ds_label = dataset_short_labels.get(ds, ds)
        standard.extend([f"{ds_label} p@5", f"{ds_label} p@15"])
        best_rotation.extend([f"{ds_label} br-p@5", f"{ds_label} br-p@15"])
    return {"standard": standard, "best_rotation": best_rotation}


def build_report_payload(
    report_spec: ReportSpec,
    base_columns: list[str],
    base_rows: list[dict[str, object]],
    metric_columns: Optional[dict[str, list[str]]] = None,
) -> dict[str, object]:
    """Create JSON payload consumed by standalone report HTML.

    Args:
        metric_columns: Optional ``{"standard": [...], "best_rotation": [...]}``
            column grouping enabling the report's standard/best-rotation toggle.
    """
    views = [
        _materialize_view(view, base_columns, base_rows, metric_columns)
        for view in report_spec.views
    ]
    if not views:
        views = [
            {
                "key": "all",
                "title": "All results",
                "note": "",
                "columns": list(base_columns),
                "rows": list(base_rows),
                "diff_mode": False,
                "baseline_config": None,
            }
        ]
    br_cols = (metric_columns or {}).get("best_rotation", [])
    has_best_rotation = any(col in base_columns for col in br_cols)
    return {
        "title": report_spec.title,
        "subtitle": report_spec.subtitle,
        "base_columns": base_columns,
        "base_rows": base_rows,
        "views": views,
        "metric_columns": metric_columns or {"standard": [], "best_rotation": []},
        "has_best_rotation": has_best_rotation,
    }


def _expand_metric_columns(
    cols: list[str],
    base_columns: list[str],
    metric_columns: Optional[dict[str, list[str]]],
) -> list[str]:
    """Add the best-rotation counterpart next to each standard metric column.

    A view that lists only standard metric columns (e.g. ``POT p@5``) would have
    nothing to show once the report's metric toggle switches to best-rotation,
    because the matching ``br-`` columns were filtered out. This inserts each
    counterpart (in either direction) right after its sibling so the toggle can
    swap variants. Counterparts absent from ``base_columns`` (no best-rotation
    data) are skipped.

    Args:
        cols: Ordered columns requested by the view.
        base_columns: All available columns; counterparts must be present here.
        metric_columns: ``{"standard": [...], "best_rotation": [...]}`` pairing.

    Returns:
        Columns with metric counterparts interleaved, order otherwise preserved.
    """
    if not metric_columns:
        return cols
    standard = metric_columns.get("standard", [])
    best_rotation = metric_columns.get("best_rotation", [])
    counterpart = dict(zip(standard, best_rotation))
    counterpart.update(zip(best_rotation, standard))

    base_set = set(base_columns)
    expanded: list[str] = []
    seen: set[str] = set()
    for col in cols:
        for candidate in (col, counterpart.get(col)):
            if candidate is None or candidate in seen or candidate not in base_set:
                continue
            expanded.append(candidate)
            seen.add(candidate)
    return expanded


def _materialize_view(
    view: ViewSpec,
    base_columns: list[str],
    base_rows: list[dict[str, object]],
    metric_columns: Optional[dict[str, list[str]]] = None,
) -> dict[str, object]:
    rows = [dict(row) for row in base_rows]

    if view.include_configs is not None:
        allowed = set(view.include_configs)
        rows = [row for row in rows if str(row.get("config")) in allowed]
    if view.exclude_configs is not None:
        denied = set(view.exclude_configs)
        rows = [row for row in rows if str(row.get("config")) not in denied]

    if view.include_methods is not None:
        allowed = set(view.include_methods)
        rows = [row for row in rows if str(row.get("method")) in allowed]
    if view.exclude_methods is not None:
        denied = set(view.exclude_methods)
        rows = [row for row in rows if str(row.get("method")) not in denied]

    for filter_fn in view.filters:
        rows = [row for row in rows if filter_fn(row)]

    for row in rows:
        config_name = str(row.get("config"))
        method_name = str(row.get("method"))
        if config_name in view.rename_configs:
            row["method"] = view.rename_configs[config_name]
        if method_name in view.rename_methods:
            row["method"] = view.rename_methods[method_name]

    rows = _apply_explicit_row_order(rows, view.row_order)
    rows = _apply_sort_rules(rows, view.sort_by)

    if view.columns is not None:
        ordered_cols = [col for col in view.columns if col in base_columns]
        ordered_cols = _expand_metric_columns(ordered_cols, base_columns, metric_columns)
    else:
        ordered_cols = list(base_columns)

    if "failed" in ordered_cols:
        ordered_cols = [col for col in ordered_cols if col != "failed"] + ["failed"]

    return {
        "key": view.key,
        "title": view.title,
        "note": view.note,
        "columns": ordered_cols,
        "rows": rows,
        "diff_mode": bool(view.diff_mode),
        "baseline_config": view.baseline_config,
    }


def _apply_explicit_row_order(
    rows: list[dict[str, object]],
    row_order: Optional[list[str]],
) -> list[dict[str, object]]:
    if row_order is None:
        return rows

    priority = {config: idx for idx, config in enumerate(row_order)}
    indexed_rows = list(enumerate(rows))
    indexed_rows.sort(
        key=lambda item: (
            priority.get(str(item[1].get("config")), len(priority)),
            item[0],
        )
    )
    return [row for _, row in indexed_rows]


def _apply_sort_rules(
    rows: list[dict[str, object]],
    rules: list[SortRule],
) -> list[dict[str, object]]:
    if not rules:
        return rows

    sorted_rows = list(rows)
    for rule in reversed(rules):
        sorted_rows.sort(
            key=lambda row: _sort_value(row.get(rule.column)),
            reverse=rule.descending,
        )
    return sorted_rows


def _sort_value(value: object) -> tuple[int, object]:
    if value is None:
        return (1, "")
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (0, float(value))
    return (0, str(value))


def write_table_report_html(payload: dict[str, object], output_path: Path) -> None:
    """Write fully standalone interactive table report HTML.

    Args:
        payload: JSON payload from :func:`build_report_payload`.
        output_path: Target HTML file path.
    """
    payload_json = json.dumps(payload, ensure_ascii=True)
    html_content = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{payload.get("title", "Evaluation Table Report")}</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --line: #d8dee6;
      --accent: #334155;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: ui-sans-serif, -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
      background: var(--bg);
    }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 16px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 6px;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
      padding: 14px;
    }}
    h1 {{ margin: 0; font-size: 28px; line-height: 1.2; }}
    .report-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .report-brand {{
      margin: 0 0 4px 0;
      font-size: 12px;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: #64748b;
    }}
    .report-subtitle {{ margin-top: 4px; color: var(--muted); font-size: 12px; }}
    .slide-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: flex-end;
    }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 12px; }}
    button, select, input {{
      border: 1px solid #c6ced8;
      background: #ffffff;
      color: var(--text);
      border-radius: 4px;
      font-size: 13px;
      padding: 6px 10px;
    }}
    button {{ cursor: pointer; }}
    button:hover {{ background: #f8fafc; }}
    .accent {{ border-color: #94a3b8; background: #f8fafc; }}
    .note {{
      margin-top: 8px;
      margin-bottom: 12px;
      color: #1e293b;
      font-size: 15px;
      font-weight: 500;
      background: #f8fafc;
      border: 1px solid #dbe2ea;
      border-left: 2px solid #94a3b8;
      border-radius: 2px;
      padding: 8px 10px;
    }}
    .muted {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .slide-nav {{ display: inline-flex; gap: 6px; align-items: center; }}
    .slide-name {{ font-size: 12px; color: #334155; min-width: 170px; text-align: right; }}
    .local-controls {{ margin-top: 10px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }}
    .export-card {{ margin-top: 12px; border: 1px solid var(--line); border-radius: 4px; padding: 8px; }}
    .export-controls {{ display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }}
    .export-controls input {{ min-width: 180px; }}
    .export-area {{
      width: 100%;
      min-height: 170px;
      margin-top: 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      border: 1px solid #9fb2c9;
      border-radius: 4px;
      background: #ffffff;
      color: #0f172a;
      padding: 8px;
    }}
    .cols {{ margin-top: 10px; display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 6px; }}
    .cols label {{ font-size: 12px; color: #334155; display: flex; gap: 6px; align-items: center; }}
    .table-scroll {{
      margin-top: 12px;
      margin-bottom: 26px;
      overflow-x: auto;
      overflow-y: hidden;
    }}
    .table-wrap {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
    }}
    .presentation-gap {{ display: flex; justify-content: flex-end; gap: 6px; }}
    .collapsible-panel {{
      margin-top: 12px;
      border-top: 1px solid var(--line);
      padding-top: 10px;
      display: none;
    }}
    .collapsible-panel.open {{ display: block; }}
    table {{ border-collapse: collapse; width: fit-content; max-width: 100%; min-width: 0; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 6px 6px; font-size: 13px; white-space: nowrap; }}
    th {{ background: #f7fafc; position: sticky; top: 0; z-index: 2; cursor: pointer; }}
    th:first-child, td:first-child {{ position: sticky; left: 0; z-index: 1; background: #fbfdff; }}
    tr:hover td {{ background: #f8fafc; }}
    .cell-method {{ cursor: pointer; }}
    .row-controls {{ display: inline-flex; gap: 4px; align-items: center; }}
    .drag-handle {{ cursor: grab; font-size: 15px; color: #475569; user-select: none; }}
    tr.dragging td {{ opacity: 0.45; }}
    tr.baseline-row td {{ background: #f1f5f9; }}
    .failed-ok {{ color: #94a3b8; }}
    .failed-warn {{ color: #b45309; font-weight: 600; }}
    .row-btn {{
      border: 1px solid #c6ced8;
      border-radius: 4px;
      padding: 2px 6px;
      font-size: 11px;
      cursor: pointer;
      background: #ffffff;
      color: #0f172a;
    }}
    .row-btn-visible {{ background: #e8f7ee; border-color: #2f9d56; color: #166534; }}
    .row-btn-hidden {{ background: #fde8e8; border-color: #c83b3b; color: #991b1b; }}
    .row-btn-baseline {{ background: #fff7e8; border-color: #d18d22; color: #7a4b00; }}
    .cell-main {{ display: inline; }}
    .cell-delta {{ margin-left: 6px; font-size: 11px; font-weight: 600; }}
    .delta-pos {{ color: #15803d; }}
    .delta-neg {{ color: #b91c1c; }}
    .delta-zero {{ color: #64748b; }}
    .dataset-sep {{ border-right: 1px solid #cbd5e1; }}
    @media (max-width: 920px) {{
      .report-header {{
        flex-direction: column;
        align-items: stretch;
      }}
      .slide-controls {{ justify-content: flex-start; }}
      .slide-name {{ text-align: left; }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"card\">
      <div class=\"report-header\">
        <div>
          <div id=\"reportTitle\" class=\"report-brand\"></div>
          <h1 id=\"slideTitle\"></h1>
          <div class=\"report-subtitle\" id=\"reportSubtitle\"></div>
        </div>
        <div class=\"slide-controls\">
          <label>View <select id=\"viewSelect\"></select></label>
          <span class=\"slide-nav\">
            <button id=\"prevSlideBtn\" type=\"button\">◀</button>
            <button id=\"nextSlideBtn\" type=\"button\">▶</button>
            <span id=\"slideName\" class=\"slide-name\"></span>
          </span>
        </div>
      </div>
      <div class=\"note\" id=\"viewNote\"></div>
      <div class=\"table-scroll\">
        <div class=\"table-wrap\">
          <table>
            <thead id=\"thead\"></thead>
            <tbody id=\"tbody\"></tbody>
          </table>
        </div>
      </div>
      <div class=\"presentation-gap\">
        <button id=\"metricToggleBtn\" type=\"button\" class=\"accent\" style=\"display:none\">Metric: standard P@k</button>
        <button id=\"copyOrgBtn\" type=\"button\">Copy org table</button>
        <button id=\"copyLatexBtn\" type=\"button\">Copy LaTeX table</button>
        <button id=\"toggleControlsBtn\" type=\"button\">Show controls</button>
      </div>
      <div id=\"advancedPanel\" class=\"collapsible-panel\">
        <div class=\"muted\" id=\"stats\"></div>
        <div class=\"controls\">
          <button id=\"fullBtn\" class=\"accent\" type=\"button\">Show full table</button>
          <button id=\"backBtn\" type=\"button\" style=\"display:none\">Back to view</button>
          <button id=\"resetBtn\" type=\"button\">Reset live edits</button>
        </div>
        <div class=\"local-controls\">
          <button id=\"selectAllRowsBtn\" type=\"button\">Select all rows</button>
          <button id=\"selectNoRowsBtn\" type=\"button\">Select no rows</button>
          <button id=\"toggleSelectedBtn\" type=\"button\" class=\"accent\">Show only selected</button>
        </div>
        <div class=\"cols\" id=\"columnToggles\"></div>
        <div class=\"export-card\">
          <div class=\"export-controls\">
            <input id=\"exportKeyInput\" type=\"text\" placeholder=\"View key\" />
            <input id=\"exportTitleInput\" type=\"text\" placeholder=\"View title\" />
            <button id=\"refreshSpecBtn\" type=\"button\">Refresh ViewSpec</button>
            <button id=\"copySpecBtn\" type=\"button\" class=\"accent\">Copy ViewSpec</button>
          </div>
          <textarea id=\"exportSpecArea\" class=\"export-area\" spellcheck=\"false\"></textarea>
        </div>
        <div class=\"muted\">Tip: use shown/hidden and set base buttons on each row. Click method to rename. Drag by handle to reorder.</div>
      </div>
    </div>
  </div>
  <script>
    const payload = {payload_json};
    const state = {{
      showFull: false,
      selectedViewKey: null,
      previousViewKey: null,
      sortColumn: null,
      sortDescending: false,
      onlySelected: false,
      diffMode: false,
      baselineRowKey: null,
      controlsExpanded: false,
      activeColumns: [],
      rows: [],
      columns: [],
      rowSelected: {{}},
      draggedRowKey: null,
      metric: 'standard',
    }};

    const els = {{
      title: document.getElementById('reportTitle'),
      slideTitle: document.getElementById('slideTitle'),
      subtitle: document.getElementById('reportSubtitle'),
      viewSelect: document.getElementById('viewSelect'),
      prevSlideBtn: document.getElementById('prevSlideBtn'),
      nextSlideBtn: document.getElementById('nextSlideBtn'),
      slideName: document.getElementById('slideName'),
      fullBtn: document.getElementById('fullBtn'),
      backBtn: document.getElementById('backBtn'),
      resetBtn: document.getElementById('resetBtn'),
      toggleControlsBtn: document.getElementById('toggleControlsBtn'),
      advancedPanel: document.getElementById('advancedPanel'),
      selectAllRowsBtn: document.getElementById('selectAllRowsBtn'),
      selectNoRowsBtn: document.getElementById('selectNoRowsBtn'),
      toggleSelectedBtn: document.getElementById('toggleSelectedBtn'),
      exportKeyInput: document.getElementById('exportKeyInput'),
      exportTitleInput: document.getElementById('exportTitleInput'),
      refreshSpecBtn: document.getElementById('refreshSpecBtn'),
      copySpecBtn: document.getElementById('copySpecBtn'),
      exportSpecArea: document.getElementById('exportSpecArea'),
      copyOrgBtn: document.getElementById('copyOrgBtn'),
      copyLatexBtn: document.getElementById('copyLatexBtn'),
      metricToggleBtn: document.getElementById('metricToggleBtn'),
      note: document.getElementById('viewNote'),
      stats: document.getElementById('stats'),
      toggles: document.getElementById('columnToggles'),
      thead: document.getElementById('thead'),
      tbody: document.getElementById('tbody'),
    }};

    function clone(value) {{
      return JSON.parse(JSON.stringify(value));
    }}

    function sourceData() {{
      if (state.showFull) {{
        return {{
          title: 'Full table',
          note: '',
          columns: payload.base_columns,
          rows: payload.base_rows,
          diff_mode: false,
          baseline_config: null,
        }};
      }}
      return payload.views.find((view) => view.key === state.selectedViewKey)
        || {{
          title: 'Full table',
          note: '',
          columns: payload.base_columns,
          rows: payload.base_rows,
          diff_mode: false,
          baseline_config: null,
        }};
    }}

    function hasFailedEntry(value) {{
      if (Array.isArray(value)) {{
        return value.some((part) => typeof part.value === 'number' && part.value > 0);
      }}
      if (typeof value === 'number') {{
        return value > 0;
      }}
      return false;
    }}

    function hasAnyFailedRows(rows) {{
      return rows.some((row) => hasFailedEntry(row.failed));
    }}

    function resetLiveState() {{
      const source = sourceData();
      state.rows = clone(source.rows);
      let columns = clone(source.columns);
      if (!hasAnyFailedRows(state.rows)) {{
        columns = columns.filter((column) => column !== 'failed');
      }}
      state.columns = columns;
      state.activeColumns = applyMetricToColumns(clone(columns));
      state.sortColumn = null;
      state.sortDescending = false;
      state.onlySelected = false;
      state.diffMode = !!source.diff_mode;
      state.rowSelected = {{}};
      state.rows.forEach((row) => {{
        state.rowSelected[rowKey(row)] = true;
      }});
      state.baselineRowKey = source.baseline_config ? String(source.baseline_config) : null;

      if (state.selectedViewKey !== null) {{
        const selectedView = payload.views.find((view) => view.key === state.selectedViewKey);
        if (selectedView) {{
          if (!els.exportKeyInput.value || els.exportKeyInput.value === selectedView.key) {{
            els.exportKeyInput.value = `${{selectedView.key}}_manual`;
          }}
          if (!els.exportTitleInput.value || els.exportTitleInput.value === selectedView.title) {{
            els.exportTitleInput.value = `${{selectedView.title}} (manual)`;
          }}
        }}
      }}
      render();
    }}

    function rowKey(row) {{
      return String(row.config ?? row.method ?? Math.random());
    }}

    function metricColumnsToHide(variant) {{
      const mc = payload.metric_columns || {{ standard: [], best_rotation: [] }};
      // In a given metric mode, hide the *other* variant's metric columns.
      return variant === 'best_rotation' ? (mc.standard || []) : (mc.best_rotation || []);
    }}

    function applyMetricToColumns(cols) {{
      const hide = new Set(metricColumnsToHide(state.metric));
      return cols.filter((col) => !hide.has(col));
    }}

    function updateMetricButton() {{
      if (!payload.has_best_rotation) {{
        els.metricToggleBtn.style.display = 'none';
        return;
      }}
      els.metricToggleBtn.style.display = 'inline-block';
      els.metricToggleBtn.textContent = state.metric === 'best_rotation'
        ? 'Metric: best-rotation P@k'
        : 'Metric: standard P@k';
    }}

    function renderColumnToggles() {{
      els.toggles.innerHTML = '';
      state.columns.forEach((column) => {{
        const label = document.createElement('label');
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = state.activeColumns.includes(column);
        cb.addEventListener('change', () => {{
          if (cb.checked) {{
            if (!state.activeColumns.includes(column)) state.activeColumns.push(column);
          }} else {{
            state.activeColumns = state.activeColumns.filter((col) => col !== column);
          }}
          render();
        }});
        label.appendChild(cb);
        label.appendChild(document.createTextNode(column));
        els.toggles.appendChild(label);
      }});
    }}

    function formatValue(value) {{
      if (value === null || value === undefined) return '-';
      if (Array.isArray(value)) {{
        const nonZero = value.filter((part) => typeof part.value === 'number' && part.value > 0);
        if (nonZero.length === 0) return '';
        const body = nonZero.map((part) => `${{part.dataset}}:${{part.value}}`).join(' / ');
        return `⚠ ${{body}}`;
      }}
      if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(1);
      return String(value);
    }}

    function numericFailedTotal(value) {{
      if (!Array.isArray(value)) return null;
      return value.reduce((acc, part) => acc + (typeof part.value === 'number' ? part.value : 0), 0);
    }}

    function baselineRow() {{
      if (!state.baselineRowKey) return null;
      const row = state.rows.find((candidate) => rowKey(candidate) === state.baselineRowKey);
      return row || null;
    }}

    function deltaSuffix(column) {{
      return String(column).includes('p@') ? 'pp' : '';
    }}

    function formatDelta(value, baseValue, column) {{
      if (typeof value !== 'number' || typeof baseValue !== 'number') return null;
      const delta = value - baseValue;
      const sign = delta > 0 ? '+' : '';
      const suffix = deltaSuffix(column);
      const cls = delta > 0 ? 'delta-pos' : (delta < 0 ? 'delta-neg' : 'delta-zero');
      return {{
        text: `${{sign}}${{delta.toFixed(1)}}${{suffix}}`,
        cls: cls,
      }};
    }}

    function sortRowsIfNeeded(rows) {{
      if (!state.sortColumn) return rows;
      const copy = [...rows];
      copy.sort((a, b) => {{
        const av = a[state.sortColumn];
        const bv = b[state.sortColumn];
        if (av == null && bv == null) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;
        if (state.sortColumn === 'failed') {{
          const af = numericFailedTotal(av) ?? -1;
          const bf = numericFailedTotal(bv) ?? -1;
          return state.sortDescending ? bf - af : af - bf;
        }}
        if (typeof av === 'number' && typeof bv === 'number') {{
          return state.sortDescending ? bv - av : av - bv;
        }}
        const cmp = String(av).localeCompare(String(bv));
        return state.sortDescending ? -cmp : cmp;
      }});
      return copy;
    }}

    function filteredRows() {{
      let rows = state.rows;
      if (state.onlySelected) {{
        rows = rows.filter((row) => state.rowSelected[rowKey(row)]);
      }}
      return sortRowsIfNeeded(rows);
    }}

    function renameMethod(row) {{
      const current = row.method == null ? '' : String(row.method);
      const next = window.prompt('Rename method label (live only):', current);
      if (next === null) return;
      row.method = next;
      render();
    }}

    function setAllRowSelection(value) {{
      state.rows.forEach((row) => {{
        state.rowSelected[rowKey(row)] = value;
      }});
      render();
    }}

    function ensureBaselineAvailable() {{
      if (state.rows.length === 0) {{
        state.baselineRowKey = null;
      }} else if (state.diffMode && !state.rows.some((row) => rowKey(row) === state.baselineRowKey)) {{
        state.baselineRowKey = rowKey(state.rows[0]);
      }}
    }}

    function pyString(value) {{
      return JSON.stringify(String(value));
    }}

    function pyList(values, indent = '    ') {{
      if (values.length === 0) return '[]';
      const lines = values.map((value) => `${{indent}}${{pyString(value)}},`);
      return `[\\n${{lines.join('\\n')}}\\n  ]`;
    }}

    function pyDict(mapping, indent = '    ') {{
      const entries = Object.entries(mapping);
      if (entries.length === 0) return '{{}}';
      const lines = entries.map(([key, value]) => `${{indent}}${{pyString(key)}}: ${{pyString(value)}},`);
      return `{{\\n${{lines.join('\\n')}}\\n  }}`;
    }}

    function currentSelectedRows() {{
      return state.rows.filter((row) => state.rowSelected[rowKey(row)]);
    }}

    function currentVisibleColumns() {{
      return state.columns.filter((column) => state.activeColumns.includes(column));
    }}

    function refreshExportSpec() {{
      const source = sourceData();
      const selectedRows = currentSelectedRows();
      const selectedConfigs = selectedRows.map((row) => String(row.config));
      const allConfigs = state.rows.map((row) => String(row.config));
      const visibleColumns = currentVisibleColumns();
      const renamedConfigs = {{}};

      state.rows.forEach((row) => {{
        const method = row.method == null ? '' : String(row.method);
        const originalMethod = row.__original_method == null ? '' : String(row.__original_method);
        if (method !== originalMethod) {{
          renamedConfigs[String(row.config)] = method;
        }}
      }});

      const keyValue = els.exportKeyInput.value.trim() || 'manual_view';
      const titleValue = els.exportTitleInput.value.trim() || (source.title ? `${{source.title}} (manual)` : 'Manual view');

      const lines = [];
      lines.push('ViewSpec(');
      lines.push(`  key=${{pyString(keyValue)}},`);
      lines.push(`  title=${{pyString(titleValue)}},`);
      if (source.note && String(source.note).trim()) {{
        lines.push(`  note=${{pyString(String(source.note))}},`);
      }}

      const includeConfigsNeeded = selectedConfigs.length !== allConfigs.length;
      if (includeConfigsNeeded) {{
        lines.push(`  include_configs=${{pyList(selectedConfigs)}},`);
      }}

      if (selectedConfigs.length > 0) {{
        lines.push(`  row_order=${{pyList(selectedConfigs)}},`);
      }}

      if (Object.keys(renamedConfigs).length > 0) {{
        lines.push(`  rename_configs=${{pyDict(renamedConfigs)}},`);
      }}

      if (visibleColumns.length > 0) {{
        lines.push(`  columns=${{pyList(visibleColumns)}},`);
      }}

      if (state.sortColumn) {{
        const sortFn = state.sortDescending ? 'desc' : 'asc';
        lines.push(`  sort_by=[${{sortFn}}(${{pyString(state.sortColumn)}})],`);
      }}

      lines.push(`  diff_mode=${{state.diffMode ? 'True' : 'False'}},`);
      if (state.baselineRowKey !== null) {{
        lines.push(`  baseline_config=${{pyString(state.baselineRowKey)}},`);
      }}
      lines.push(')');

      const preface = [
        '# Copy into report spec file',
        '# imports: from flatsam2.results_table_report import ViewSpec, asc, desc',
        '',
      ].join('\\n');
      els.exportSpecArea.value = preface + lines.join('\\n');
    }}

    async function copyExportSpec() {{
      const text = els.exportSpecArea.value;
      if (!text) return;
      if (navigator.clipboard && navigator.clipboard.writeText) {{
        try {{
          await navigator.clipboard.writeText(text);
          return;
        }} catch (error) {{
        }}
      }}
      els.exportSpecArea.focus();
      els.exportSpecArea.select();
      document.execCommand('copy');
    }}

    function reorderRows(dragKey, dropKey) {{
      const fromIdx = state.rows.findIndex((row) => rowKey(row) === dragKey);
      const toIdx = state.rows.findIndex((row) => rowKey(row) === dropKey);
      if (fromIdx < 0 || toIdx < 0 || fromIdx === toIdx) return;
      const [moved] = state.rows.splice(fromIdx, 1);
      state.rows.splice(toIdx, 0, moved);
    }}

    function hasDatasetSeparator(column) {{
      return column === 'POT p@15';
    }}

    function renderTable() {{
      const visibleCols = state.columns.filter((column) => state.activeColumns.includes(column));
      ensureBaselineAvailable();
      const rows = filteredRows();
      const base = baselineRow();

      els.thead.innerHTML = '';
      const headTr = document.createElement('tr');
      const ctrlTh = document.createElement('th');
      ctrlTh.textContent = '↕';
      headTr.appendChild(ctrlTh);
      visibleCols.forEach((column) => {{
        const th = document.createElement('th');
        const marker = state.sortColumn === column ? (state.sortDescending ? ' ▼' : ' ▲') : '';
        th.textContent = column + marker;
        if (hasDatasetSeparator(column)) {{
          th.classList.add('dataset-sep');
        }}
        th.addEventListener('click', () => {{
          if (state.sortColumn === column) {{
            state.sortDescending = !state.sortDescending;
          }} else {{
            state.sortColumn = column;
            state.sortDescending = false;
          }}
          render();
        }});
        headTr.appendChild(th);
      }});
      els.thead.appendChild(headTr);

      els.tbody.innerHTML = '';
      rows.forEach((row) => {{
        const currentKey = rowKey(row);
        const baselineActive = state.diffMode && state.baselineRowKey === currentKey;
        const tr = document.createElement('tr');
        if (baselineActive) {{
          tr.classList.add('baseline-row');
        }}
        tr.draggable = true;
        tr.dataset.rowKey = currentKey;
        tr.addEventListener('dragstart', (event) => {{
          state.draggedRowKey = currentKey;
          state.sortColumn = null;
          state.sortDescending = false;
          tr.classList.add('dragging');
          if (event.dataTransfer) {{
            event.dataTransfer.effectAllowed = 'move';
            event.dataTransfer.setData('text/plain', currentKey);
          }}
        }});
        tr.addEventListener('dragend', () => {{
          tr.classList.remove('dragging');
        }});
        tr.addEventListener('dragover', (event) => {{
          event.preventDefault();
        }});
        tr.addEventListener('drop', (event) => {{
          event.preventDefault();
          const dropKey = tr.dataset.rowKey;
          if (!dropKey || !state.draggedRowKey) return;
          reorderRows(state.draggedRowKey, dropKey);
          state.draggedRowKey = null;
          render();
        }});

        const ctrlTd = document.createElement('td');
        ctrlTd.className = 'row-controls';

        if (state.controlsExpanded) {{
          const visBtn = document.createElement('button');
          visBtn.type = 'button';
          visBtn.className = `row-btn ${{state.rowSelected[currentKey] ? 'row-btn-visible' : 'row-btn-hidden'}}`;
          visBtn.textContent = state.rowSelected[currentKey] ? 'shown' : 'hidden';
          visBtn.title = 'Toggle selected/hidden for Show only selected mode';
          visBtn.addEventListener('click', (event) => {{
            event.stopPropagation();
            state.rowSelected[currentKey] = !state.rowSelected[currentKey];
            render();
          }});
          ctrlTd.appendChild(visBtn);
        }}

        const baseBtn = document.createElement('button');
        baseBtn.type = 'button';
        baseBtn.className = `row-btn ${{baselineActive ? 'row-btn-baseline' : ''}}`;
        baseBtn.textContent = baselineActive ? 'baseline' : 'set base';
        baseBtn.title = 'Set this row as diff baseline';
        baseBtn.addEventListener('click', (event) => {{
          event.stopPropagation();
          const isActive = state.diffMode && state.baselineRowKey === currentKey;
          if (isActive) {{
            state.baselineRowKey = null;
            state.diffMode = false;
          }} else {{
            state.baselineRowKey = currentKey;
            state.diffMode = true;
          }}
          render();
        }});
        ctrlTd.appendChild(baseBtn);

        const handle = document.createElement('span');
        handle.className = 'drag-handle';
        handle.textContent = '⋮⋮';
        ctrlTd.appendChild(handle);
        tr.appendChild(ctrlTd);

        visibleCols.forEach((column) => {{
          const td = document.createElement('td');
          const value = row[column];
          if (hasDatasetSeparator(column)) {{
            td.classList.add('dataset-sep');
          }}
          const mainSpan = document.createElement('span');
          mainSpan.className = 'cell-main';
          mainSpan.textContent = formatValue(value);
          td.appendChild(mainSpan);

          if (state.diffMode && base !== null && column !== 'method' && column !== 'config' && column !== 'failed') {{
            const delta = formatDelta(value, base[column], column);
            if (delta !== null) {{
              const deltaSpan = document.createElement('span');
              deltaSpan.className = `cell-delta ${{delta.cls}}`;
              deltaSpan.textContent = delta.text;
              td.appendChild(deltaSpan);
            }}
          }}

          if (column === 'failed') {{
            td.className = Array.isArray(value) && value.length > 0 ? 'failed-warn' : 'failed-ok';
          }}
          if (column === 'method') {{
            td.className = 'cell-method';
            td.title = 'Click to rename label';
            td.addEventListener('click', () => renameMethod(row));
          }}
          tr.appendChild(td);
        }});
        els.tbody.appendChild(tr);
      }});

      const selectedCount = state.rows.filter((row) => state.rowSelected[rowKey(row)]).length;
      const filterMode = state.onlySelected ? 'only selected' : 'all';
      const diffLabel = state.diffMode && base !== null
        ? `diff: on (baseline=${{base.method}} / ${{base.config}})`
        : 'diff: off';
      els.stats.textContent =
        `rows shown: ${{rows.length}} / ${{state.rows.length}} | selected: ${{selectedCount}} | mode: ${{filterMode}} | ${{diffLabel}}`;
    }}

    function renderHeader() {{
      const source = sourceData();
      els.title.textContent = payload.title || 'Evaluation Table Report';
      els.subtitle.textContent = payload.subtitle || '';
      els.slideTitle.textContent = source.title || 'Slide';
      els.note.textContent = source.note || 'No interpretation note for this slide.';
      els.backBtn.style.display = state.showFull ? 'inline-block' : 'none';
      const viewTitles = payload.views.map((view) => view.title);
      const idx = payload.views.findIndex((view) => view.key === state.selectedViewKey);
      const hasView = idx >= 0;
      const label = state.showFull
        ? 'Full table'
        : (hasView ? `Slide ${{idx + 1}} / ${{viewTitles.length}}` : 'Slide');
      els.slideName.textContent = label;
    }}

    function renderControlsPanel() {{
      els.advancedPanel.classList.toggle('open', state.controlsExpanded);
      els.toggleControlsBtn.textContent = state.controlsExpanded ? 'Hide controls' : 'Show controls';
    }}

    function render() {{
      renderHeader();
      updateMetricButton();
      renderColumnToggles();
      renderTable();
      renderControlsPanel();
      els.toggleSelectedBtn.textContent = state.onlySelected ? 'Show all rows' : 'Show only selected';
      refreshExportSpec();
    }}

    function setSlideByIndex(index) {{
      if (payload.views.length === 0) return;
      const wrapped = (index + payload.views.length) % payload.views.length;
      state.selectedViewKey = payload.views[wrapped].key;
      state.showFull = false;
      els.viewSelect.value = state.selectedViewKey;
      resetLiveState();
    }}

    function setupViews() {{
      els.viewSelect.innerHTML = '';
      payload.views.forEach((view) => {{
        const option = document.createElement('option');
        option.value = view.key;
        option.textContent = view.title;
        els.viewSelect.appendChild(option);
      }});

      if (payload.views.length > 0) {{
        state.selectedViewKey = payload.views[0].key;
        els.viewSelect.value = state.selectedViewKey;
      }}

      els.viewSelect.addEventListener('change', () => {{
        state.selectedViewKey = els.viewSelect.value;
        state.showFull = false;
        resetLiveState();
      }});

      els.prevSlideBtn.addEventListener('click', () => {{
        const idx = payload.views.findIndex((view) => view.key === state.selectedViewKey);
        setSlideByIndex(idx - 1);
      }});
      els.nextSlideBtn.addEventListener('click', () => {{
        const idx = payload.views.findIndex((view) => view.key === state.selectedViewKey);
        setSlideByIndex(idx + 1);
      }});
    }}

    els.fullBtn.addEventListener('click', () => {{
      state.previousViewKey = state.selectedViewKey;
      state.showFull = true;
      resetLiveState();
    }});

    els.backBtn.addEventListener('click', () => {{
      state.showFull = false;
      if (state.previousViewKey) {{
        state.selectedViewKey = state.previousViewKey;
        els.viewSelect.value = state.selectedViewKey;
      }}
      resetLiveState();
    }});

    els.resetBtn.addEventListener('click', () => resetLiveState());
    els.toggleControlsBtn.addEventListener('click', () => {{
      state.controlsExpanded = !state.controlsExpanded;
      render();
    }});
    els.selectAllRowsBtn.addEventListener('click', () => setAllRowSelection(true));
    els.selectNoRowsBtn.addEventListener('click', () => setAllRowSelection(false));
    els.toggleSelectedBtn.addEventListener('click', () => {{
      state.onlySelected = !state.onlySelected;
      render();
    }});
    els.refreshSpecBtn.addEventListener('click', () => refreshExportSpec());
    els.copySpecBtn.addEventListener('click', () => copyExportSpec());
    els.exportKeyInput.addEventListener('input', () => refreshExportSpec());
    els.exportTitleInput.addEventListener('input', () => refreshExportSpec());

    function exportVisibleCols() {{
      return currentVisibleColumns();
    }}

    function exportCellText(row, col, base) {{
      const value = row[col];
      const main = formatValue(value);
      if (!state.diffMode || base === null || col === 'method' || col === 'config' || col === 'failed') {{
        return main;
      }}
      const delta = formatDelta(value, base[col], col);
      return delta !== null ? `${{main}} (${{delta.text}})` : main;
    }}

    function formatOrgTable() {{
      const cols = exportVisibleCols();
      const rows = filteredRows();
      const base = baselineRow();
      const widths = cols.map((col) => col.length);
      const cells = rows.map((row) => cols.map((col) => exportCellText(row, col, base)));
      cells.forEach((rowCells) => {{
        rowCells.forEach((cell, i) => {{
          widths[i] = Math.max(widths[i], cell.length);
        }});
      }});
      const fmtRow = (rowCells) =>
        '| ' + rowCells.map((cell, i) => cell.padEnd(widths[i])).join(' | ') + ' |';
      const sep = '|-' + widths.map((w) => '-'.repeat(w)).join('-+-') + '-|';
      return [fmtRow(cols), sep, ...cells.map(fmtRow)].join('\\n');
    }}

    function latexEscape(str) {{
      return str
        .replace(/\\\\/g, '\\\\textbackslash{{}}')
        .replace(/%/g, '\\\\%')
        .replace(/#/g, '\\\\#')
        .replace(/\\$/g, '\\\\$')
        .replace(/_/g, '\\\\_')
        .replace(/\\^/g, '\\\\^{{}}')
        .replace(/~/g, '\\\\~{{}}')
        .replace(/\\{{/g, '\\\\{{')
        .replace(/\\}}/g, '\\\\}}');
    }}

    function formatLatexTable() {{
      const cols = exportVisibleCols();
      const rows = filteredRows();
      const base = baselineRow();
      const colSpec = (cols.length > 0 ? 'l' : '') + 'c'.repeat(Math.max(0, cols.length - 1));
      const lines = [];
      lines.push(`\\\\begin{{tabular}}{{${{colSpec}}}}`);
      lines.push('\\\\toprule');
      lines.push(cols.map(latexEscape).join(' & ') + ' \\\\\\\\');
      lines.push('\\\\midrule');
      rows.forEach((row) => {{
        const rowCells = cols.map((col) => latexEscape(exportCellText(row, col, base)));
        lines.push(rowCells.join(' & ') + ' \\\\\\\\');
      }});
      lines.push('\\\\bottomrule');
      lines.push('\\\\end{{tabular}}');
      return lines.join('\\n');
    }}

    async function copyTextWithFeedback(text, btn) {{
      const original = btn.textContent;
      try {{
        if (navigator.clipboard && navigator.clipboard.writeText) {{
          await navigator.clipboard.writeText(text);
        }} else {{
          const ta = document.createElement('textarea');
          ta.value = text;
          ta.style.position = 'fixed';
          ta.style.opacity = '0';
          document.body.appendChild(ta);
          ta.focus();
          ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
        }}
        btn.textContent = 'Copied!';
      }} catch (err) {{
        btn.textContent = 'Failed';
      }}
      setTimeout(() => {{ btn.textContent = original; }}, 1500);
    }}

    els.metricToggleBtn.addEventListener('click', () => {{
      state.metric = state.metric === 'best_rotation' ? 'standard' : 'best_rotation';
      state.activeColumns = applyMetricToColumns(state.columns);
      state.sortColumn = null;
      state.sortDescending = false;
      render();
    }});

    els.copyOrgBtn.addEventListener('click', () => {{
      copyTextWithFeedback(formatOrgTable(), els.copyOrgBtn);
    }});
    els.copyLatexBtn.addEventListener('click', () => {{
      copyTextWithFeedback(formatLatexTable(), els.copyLatexBtn);
    }});

    setupViews();
    resetLiveState();
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
