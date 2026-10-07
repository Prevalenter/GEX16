"""Read-only robot resources shipped with the package."""

from pathlib import Path

RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"
DEFAULT_CHECKPOINT_DIR = RESOURCE_ROOT / "retarget"
DEFAULT_CALIBRATION_PATH = DEFAULT_CHECKPOINT_DIR / "calibration.npz"
DEFAULT_REFERENCE_PATH = DEFAULT_CHECKPOINT_DIR / "reference.npy"
EX16_URDF_PATH = RESOURCE_ROOT / "ex16/urdf/glove4.urdf"
EX16_MESH_DIR = RESOURCE_ROOT / "ex16/meshes"
GX16_URDF_PATH = RESOURCE_ROOT / "gx16/urdf/gx4m.urdf"
GX16_MESH_DIR = RESOURCE_ROOT / "gx16/meshes"
COLLISION_URDF_PATH = RESOURCE_ROOT / "gx16/collision.urdf"
