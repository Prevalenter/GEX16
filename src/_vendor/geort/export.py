# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import torch

from _vendor.geort.formatter import HandFormatter
from _vendor.geort.model import IKModel
from _vendor.geort.utils.config_utils import (
    load_json,
    parse_config_joint_limit,
    parse_config_keypoint_info,
)


class GeoRTRetargetingModel:
    """
    Used by external programs.
    """

    def __init__(self, model_path, config_path, device="cuda"):
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "GEX Retarget requires a CUDA-capable PyTorch installation"
            )
        self.device = torch.device(device)
        config = load_json(config_path)
        keypoint_info = parse_config_keypoint_info(config)
        joint_lower_limit, joint_upper_limit = parse_config_joint_limit(config)
        if (
            joint_lower_limit.shape != (16,)
            or joint_upper_limit.shape != (16,)
            or not np.isfinite([joint_lower_limit, joint_upper_limit]).all()
            or np.any(joint_lower_limit >= joint_upper_limit)
        ):
            raise ValueError("Checkpoint must contain 16 finite ordered joint limits")
        self.human_ids = keypoint_info["human_id"]
        self.model = IKModel(keypoint_joints=keypoint_info["joint"]).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        self.qpos_normalizer = HandFormatter(
            joint_lower_limit, joint_upper_limit
        )  # GeoRT will do normalization.

    @torch.inference_mode()
    def forward(self, keypoints):
        keypoints = np.asarray(keypoints)
        if keypoints.shape != (21, 3) or not np.isfinite(keypoints).all():
            raise ValueError("Expected finite keypoints with shape [21, 3]")
        # keypoints: [N, 3]
        keypoints = keypoints[self.human_ids]  # extract.
        joint_normalized = self.model.forward(
            torch.from_numpy(keypoints)
            .unsqueeze(0)
            .reshape(1, -1, 3)
            .float()
            .to(self.device)
        )
        joint_raw = self.qpos_normalizer.unnormalize(
            joint_normalized.detach().cpu().numpy()
        )
        return joint_raw[0]


def load_model(checkpoint_dir, device="cuda"):
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    model_path = checkpoint_root / "last.pth"
    config_path = checkpoint_root / "config.json"
    for path in (model_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint requires {path}")
    return GeoRTRetargetingModel(
        model_path=model_path, config_path=config_path, device=device
    )
