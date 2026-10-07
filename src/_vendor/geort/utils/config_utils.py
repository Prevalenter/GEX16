# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json

import numpy as np


def load_json(filename):
    """
    Load data from a JSON file into a Python dictionary.

    Parameters:
    - filename (str): Path to the JSON file to be loaded.

    Returns:
    - dict: The loaded data.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_config_keypoint_info(config):
    keypoint_links = []
    keypoint_offsets = []
    keypoint_joints = []
    keypoint_human_ids = []

    joint_order = config["joint_order"]

    for info in config["fingertip_link"]:
        keypoint_links.append(info["link"])
        keypoint_offsets.append(info["center_offset"])
        keypoint_human_ids.append(info["human_hand_id"])

        keypoint_joint = []
        for joint in info["joint"]:
            keypoint_joint.append(joint_order.index(joint))

        keypoint_joints.append(keypoint_joint)

    out = {
        "link": keypoint_links,
        "offset": keypoint_offsets,
        "joint": keypoint_joints,
        "human_id": keypoint_human_ids,
    }
    return out


def parse_config_joint_limit(config):
    lower_limit = config["joint"]["lower"]
    upper_limit = config["joint"]["upper"]
    return np.array(lower_limit), np.array(upper_limit)
