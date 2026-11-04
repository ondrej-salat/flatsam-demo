"""Test configuration system."""
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from flatsam2.config import (
    TrackerConfig,
    FlowConfig,
    LoMaPathConfig,
    SegmenterConfig,
    load_config_from_file,
)


def test_default_config():
    """Default config creation."""
    config = TrackerConfig()
    assert config.flow.model == "raft"
    assert config.flow.weights == "sintel"
    assert config.segmenter.model == "sam3"
    # RANSAC belongs to whichever estimator runs it, not to the flow model.
    assert config.loma.ransac_threshold == 5.0


def test_custom_config():
    """Custom config values."""
    config = TrackerConfig(flow=FlowConfig(model="gma", weights="kitti"))
    assert config.flow.model == "gma"
    assert config.flow.weights == "kitti"


def test_load_config_from_file():
    """Loading a config from a Python file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
from flatsam2.config import TrackerConfig, FlowConfig

config = TrackerConfig(
    flow=FlowConfig(model="gma", weights="things"),
    loma={"ransac_threshold": 2.5},
)
""")
        config_path = Path(f.name)

    try:
        config = load_config_from_file(config_path)
        assert isinstance(config, TrackerConfig)
        assert config.flow.model == "gma"
        assert config.flow.weights == "things"
        assert config.loma.ransac_threshold == 2.5
    finally:
        config_path.unlink()


def test_config_validation():
    """Pydantic rejects a wrongly typed field."""
    config = LoMaPathConfig(ransac_threshold=5.0)
    assert config.ransac_threshold == 5.0

    with pytest.raises(ValidationError):
        LoMaPathConfig(ransac_threshold="invalid")


def test_correction_cfg_defaults_keep_existing_behavior():
    """New oracle fields default to the pre-oracle semantics (loma/loma, no dry run)."""
    config = TrackerConfig()
    assert config.correction.trigger_source == "loma"
    assert config.correction.content_source == "loma"
    assert config.correction.dry_run is False
    assert config.correction.period == 30


def test_oracle_grid_configs_load():
    """Each E-cell config file loads and carries the expected correction sources."""
    configs_dir = Path(__file__).resolve().parent.parent / "configs" / "tracker"
    expected = {
        "loma_samh_correct_decoupled_oracleTC.py": ("oracle", "oracle", "sam_excess_depth", False),
        "loma_samh_correct_decoupled_oracleT.py": ("oracle", "loma", "sam_excess_depth", False),
        "loma_samh_correct_decoupled_oracleC.py": ("loma", "oracle", "sam_excess_depth", False),
        "loma_samh_correct_decoupled_periodic30_oracleC.py": ("loma", "oracle", "periodic", False),
        "loma_samh_correct_decoupled_excessdepth_rerun.py": ("loma", "loma", "sam_excess_depth", False),
        "loma_samh_correct_decoupled_dryrun.py": ("loma", "loma", "sam_excess_depth", True),
    }
    for filename, (trigger_source, content_source, trigger, dry_run) in expected.items():
        config = load_config_from_file(configs_dir / filename)
        assert config.algorithm == "loma_samh", filename
        assert config.correction.enabled is True, filename
        assert config.correction.trigger_source == trigger_source, filename
        assert config.correction.content_source == content_source, filename
        assert config.correction.trigger == trigger, filename
        assert config.correction.dry_run is dry_run, filename
        assert config.loma.decouple_hough_state is True, filename
    periodic = load_config_from_file(
        configs_dir / "loma_samh_correct_decoupled_periodic30_oracleC.py"
    )
    assert periodic.correction.period == 30


def test_fusion_flag_resolves_to_corner_fusion_algorithm():
    """loma.fusion=True must actually select algorithm='corner_fusion'.

    Regression test: model_copy(update=...) — how every configs/tracker/*.py
    composes a base config — does not re-run the algorithm-derivation
    validator, so a config that flips loma.fusion via model_copy() without
    also passing algorithm="corner_fusion" explicitly silently keeps running
    the base config's algorithm instead (caught in the field 2026-08-10: a
    real config shipped this way and silently ran loma_samh, not the new
    fusion code, on every frame).
    """
    config = TrackerConfig(
        loma={"enabled": True, "decouple_hough_state": True, "fusion": True}
    )
    assert config.algorithm == "corner_fusion"


def test_corner_fusion_alignh_config_loads():
    """The shipped corner-fusion config resolves to the right algorithm."""
    configs_dir = Path(__file__).resolve().parent.parent / "configs" / "tracker"
    config = load_config_from_file(configs_dir / "corner_fusion_alignh.py")
    assert config.algorithm == "corner_fusion"
    assert config.loma.fusion is True
    assert config.loma.decouple_hough_state is True
    assert config.hough.method == "align"
