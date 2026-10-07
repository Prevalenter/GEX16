"""GEX Retarget settings and bundled model loading."""

import json
from pathlib import Path

from paths import COLLISION_URDF_PATH, DEFAULT_CHECKPOINT_DIR, RESOURCE_ROOT


def get_config(name="gx16"):
    if name != "gx16":
        raise ValueError("Only GX16 is supported")
    config = json.loads((RESOURCE_ROOT / "gx16/retargeting.json").read_text())
    config["urdf_path"] = str(COLLISION_URDF_PATH)
    return config


def load_model(checkpoint_dir=DEFAULT_CHECKPOINT_DIR, device="cuda"):
    directory = Path(checkpoint_dir).expanduser().resolve()
    for name in ("last.pth", "config.json"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Checkpoint requires {directory / name}")
    from _vendor.geort.export import load_model as load

    return load(checkpoint_dir, device=device)
