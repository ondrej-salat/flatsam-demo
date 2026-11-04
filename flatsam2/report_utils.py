"""Utilities for generating self-contained HTML reports."""

from pathlib import Path

PLOTLY_VERSION = "2.27.0"
PLOTLY_CDN_URL = f"https://cdn.plot.ly/plotly-{PLOTLY_VERSION}.min.js"
_CACHE_PATH = Path.home() / ".cache" / "flatsam2" / f"plotly-{PLOTLY_VERSION}.min.js"


def get_plotly_script_tag(cdn: bool = False) -> str:
    """Return a <script> tag for Plotly.

    Args:
        cdn: If True, return a CDN src tag. If False (default), download
            plotly once to ~/.cache/flatsam2/ and inline it in the tag.

    Returns:
        HTML <script> string.
    """
    if cdn:
        return f'<script src="{PLOTLY_CDN_URL}"></script>'
    if not _CACHE_PATH.exists():
        import urllib.request

        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Plotly {PLOTLY_VERSION} to {_CACHE_PATH} …")
        urllib.request.urlretrieve(PLOTLY_CDN_URL, _CACHE_PATH)
    js = _CACHE_PATH.read_text(encoding="utf-8")
    return f"<script>{js}</script>"
