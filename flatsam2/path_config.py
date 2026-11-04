from dataclasses import dataclass
from pathlib import Path
import os
import socket

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(env_file: Path = REPO_ROOT / ".env") -> None:
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:  # empty values in .env.example mean "unset", not ""
            os.environ.setdefault(key.strip(), value)


load_dotenv()


def _env_path(name: str, default: str) -> Path:
    return REPO_ROOT / Path(os.environ.get(name, default)).expanduser()  # relative = repo root


@dataclass
class PathConfig:
    POT_BASE: Path
    PT_BASE: Path
    PT_INITFIX_DIR: Path
    POIC_BASE: Path
    TMT_BASE: Path
    UCSB_BASE: Path
    MPOT_BASE: Path
    SAM3_PRECOMPUTED_MASKS_POT: Path
    SAM3_PRECOMPUTED_MASKS_PTI: Path
    OUTPUT_DIRECTORY: Path
    VIS_DIRECTORY: Path
    COMPARISONS_DIRECTORY: Path


HOSTNAME = socket.gethostname()

PATH_CONFIGS = {
    # Default config; every field can be overridden by the same-named var in .env
    "default": PathConfig(
        POT_BASE=_env_path("POT_BASE", "/mnt/data/vrg/public_datasets/POT-210/"),
        PT_BASE=_env_path("PT_BASE", "/mnt/data/vrg/public_datasets/PlanarTrack/PlanarTrack-Data/"),
        PT_INITFIX_DIR=_env_path(
            "PT_INITFIX_DIR",
            "/mnt/data/vrg/public_datasets/PlanarTrack/PlanarTrack_init_reannotation/",
        ),
        POIC_BASE=_env_path("POIC_BASE", "/mnt/personal/serycjon/data/POIC/"),
        TMT_BASE=_env_path("TMT_BASE", "/mnt/personal/serycjon/data/TMT/"),
        UCSB_BASE=_env_path("UCSB_BASE", "/mnt/personal/serycjon/data/UCSB/"),
        MPOT_BASE=_env_path("MPOT_BASE", "/mnt/personal/serycjon/data/MPOT-3K/"),
        SAM3_PRECOMPUTED_MASKS_POT=_env_path(
            "SAM3_PRECOMPUTED_MASKS_POT",
            "/mnt/personal/serycjon/flatsam2/results/hough_symmetry_only/POT-210/masks/",
        ),
        SAM3_PRECOMPUTED_MASKS_PTI=_env_path(
            "SAM3_PRECOMPUTED_MASKS_PTI",
            "/mnt/personal/serycjon/flatsam2/results/hough_symmetry_only/PlanarTrackInitfix/masks/",
        ),
        OUTPUT_DIRECTORY=_env_path("OUTPUT_DIRECTORY", "/mnt/personal/serycjon/flatsam2/results/"),
        VIS_DIRECTORY=_env_path("VIS_DIRECTORY", "/mnt/personal/serycjon/flatsam2/vis/"),
        COMPARISONS_DIRECTORY=_env_path(
            "COMPARISONS_DIRECTORY", "/mnt/personal/serycjon/flatsam2/comparisons/"
        ),
    ),
    # Example for another hostname:
    # "your_other_host": PathConfig(...),
}

# Select config by hostname, fallback to 'default'
paths: PathConfig = PATH_CONFIGS.get(HOSTNAME, PATH_CONFIGS["default"])
