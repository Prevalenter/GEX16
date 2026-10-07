"""Project EX16 joint states into calibrated retargeting landmarks.

EX16 has no little-finger sensing; those landmarks are synthetic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from paths import EX16_URDF_PATH

DEFAULT_URDF_PATH = EX16_URDF_PATH
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 17))
TIP_IDS = np.asarray((4, 8, 12, 16), dtype=np.int64)

# Four EX16 mechanisms correspond to thumb, index, middle and ring.  There
# are five URDF link frames per mechanism after the wrist; the fourth
# mechanical link frame is omitted so each chain follows MediaPipe's four
# post-wrist landmarks and still terminates at the physical fingertip link.
SOURCE_CHAINS = (
    ("Link1", "Link2", "Link3", "Link17"),
    ("Link5", "Link6", "Link7", "Link18"),
    ("Link9", "Link10", "Link11", "Link19"),
    ("link13", "link14", "link15", "link20"),
)
HUMAN_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
)


def load_reference_hand(path: Path) -> np.ndarray:
    """Load a finite ``[T, 21, 3]`` reference trajectory."""
    path = Path(path).expanduser().resolve()
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Failed to load reference hand data from {path}: {exc}"
        ) from exc
    if data.ndim != 3 or data.shape[1:] != (21, 3) or data.shape[0] == 0:
        raise ValueError(
            f"Expected reference shape [T, 21, 3], got {data.shape} from {path}"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"Reference hand data contains NaN or Inf: {path}")
    return np.asarray(data, dtype=np.float64)


class EX16HumanProjector:
    """Project EX16 URDF states into wrist-local hand coordinates."""

    def __init__(self, urdf_path: Path = DEFAULT_URDF_PATH) -> None:
        try:
            import yourdfpy
        except ImportError as exc:
            raise ImportError(
                "yourdfpy is required: python -m pip install yourdfpy"
            ) from exc

        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        self.urdf = yourdfpy.URDF.load(
            self.urdf_path,
            build_scene_graph=True,
            load_meshes=False,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
        if "plam_link" not in {link.name for link in self.urdf.robot.links}:
            raise ValueError(f"EX16 URDF is missing plam_link: {self.urdf_path}")
        available_joints = set(self.urdf.actuated_joint_names)
        missing_joints = set(JOINT_NAMES) - available_joints
        if missing_joints:
            raise ValueError(f"EX16 URDF is missing joints: {sorted(missing_joints)}")

        self.scale: float | None = None
        self.rotation: np.ndarray | None = None
        self.reference_frame: np.ndarray | None = None
        self.reference_frame_index: int | None = None
        self.open_projected: np.ndarray | None = None
        self.calibration_qpos_deg: np.ndarray | None = None
        self.calibration_rmse_m: float | None = None

    @staticmethod
    def _validate_qpos(qpos_deg: Sequence[float]) -> np.ndarray:
        qpos = np.asarray(qpos_deg, dtype=np.float64)
        if qpos.shape != (16,):
            raise ValueError(f"Expected 16 EX16 joint angles, got {qpos.shape}")
        if not np.isfinite(qpos).all():
            raise ValueError("EX16 joint angles contain NaN or Inf")
        return qpos

    def source_keypoints(self, qpos_deg: Sequence[float]) -> np.ndarray:
        """Return four EX16 chains relative to ``plam_link`` in metres."""
        qpos = self._validate_qpos(qpos_deg)
        self.urdf.update_cfg(
            {name: np.deg2rad(value) for name, value in zip(JOINT_NAMES, qpos)}
        )
        points = np.zeros((21, 3), dtype=np.float64)
        for source_chain, human_chain in zip(SOURCE_CHAINS, HUMAN_CHAINS):
            for link_name, human_id in zip(source_chain, human_chain):
                points[human_id] = self.urdf.get_transform(link_name, "plam_link")[
                    :3, 3
                ]
        return points

    def restore_calibration(
        self,
        calibration_qpos_deg: Sequence[float],
        scale: float,
        rotation: np.ndarray,
        reference: np.ndarray,
        reference_frame_index: int,
    ) -> None:
        """Restore a calibration saved in an EX16 raw recording."""
        calibration_qpos = self._validate_qpos(calibration_qpos_deg)
        scale = float(scale)
        rotation = np.asarray(rotation, dtype=np.float64)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"Calibration scale must be positive and finite, got {scale}"
            )
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("Calibration rotation must be a finite [3, 3] matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("Calibration rotation must be orthonormal")
        if np.linalg.det(rotation) < 0.999:
            raise ValueError("Calibration rotation must be a proper rotation")
        reference = np.asarray(reference, dtype=np.float64)
        if reference.ndim != 3 or reference.shape[1:] != (21, 3):
            raise ValueError(
                f"Expected reference shape [T, 21, 3], got {reference.shape}"
            )
        if not 0 <= reference_frame_index < len(reference):
            raise ValueError(
                f"Reference frame {reference_frame_index} is outside [0, {len(reference)})"
            )

        reference_frame = (
            reference[reference_frame_index] - reference[reference_frame_index, 0]
        )
        source_open = self.source_keypoints(calibration_qpos)
        open_projected = source_open @ rotation * scale
        errors = open_projected[TIP_IDS] - reference_frame[TIP_IDS]

        self.scale = scale
        self.rotation = rotation
        self.reference_frame = reference_frame
        self.reference_frame_index = int(reference_frame_index)
        self.open_projected = open_projected
        self.calibration_qpos_deg = calibration_qpos
        self.calibration_rmse_m = float(np.sqrt(np.mean(errors * errors)))

    def project(self, qpos_deg: Sequence[float]) -> np.ndarray:
        """Return one finite ``[21, 3]`` frame in retargeting coordinates."""
        if self.scale is None or self.rotation is None:
            raise RuntimeError(
                "EX16HumanProjector must be calibrated before projection"
            )
        points = self.source_keypoints(qpos_deg) @ self.rotation * self.scale
        points[0] = 0.0

        # EX16 does not measure a pinky. Keep its four points fixed in the
        # calibrated wrist-local reference pose. Allegro and GX16 training
        # only select IDs 4/8/12/16.
        points[17:21] = self.reference_frame[17:21]
        return points.astype(np.float32)


def restore_projector(
    calibration_path: Path,
    reference_path: Path,
) -> tuple[EX16HumanProjector, dict[str, Any]]:
    """Restore the exact EX16-to-human transform used to create training data."""
    calibration_path = Path(calibration_path).expanduser().resolve()
    reference_path = Path(reference_path).expanduser().resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    reference = load_reference_hand(reference_path)
    try:
        with np.load(calibration_path, allow_pickle=False) as archive:
            required = {
                "calibration_qpos_deg",
                "calibration_scale",
                "calibration_rotation",
                "metadata_json",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"missing calibration arrays: {sorted(missing)}")
            qpos_deg = np.asarray(archive["calibration_qpos_deg"], dtype=np.float64)
            scale = float(archive["calibration_scale"].item())
            rotation = np.asarray(archive["calibration_rotation"], dtype=np.float64)
            metadata = json.loads(str(archive["metadata_json"].item()))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Failed to load calibration from {calibration_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict) or "reference_frame" not in metadata:
        raise ValueError("Calibration metadata must contain reference_frame")

    projector = EX16HumanProjector()
    projector.restore_calibration(
        qpos_deg,
        scale,
        rotation,
        reference,
        int(metadata["reference_frame"]),
    )
    return projector, metadata
