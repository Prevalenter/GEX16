# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch


# This is just a simple joint position normalizer.
class HandFormatter:
    def __init__(self, joint_lower_limit, joint_upper_limit):
        self.joint_lower_limit = np.array(joint_lower_limit)
        self.joint_upper_limit = np.array(joint_upper_limit)
        return

    def normalize(self, x):
        return (
            (x - self.joint_lower_limit)
            / (self.joint_upper_limit - self.joint_lower_limit)
            - 0.5
        ) * 2

    def unnormalize(self, x):
        return (x / 2 + 0.5) * (
            self.joint_upper_limit - self.joint_lower_limit
        ) + self.joint_lower_limit

    def normalize_torch(self, x):
        return (
            (x - torch.from_numpy(self.joint_lower_limit).cuda())
            / (
                torch.from_numpy(self.joint_upper_limit).cuda()
                - torch.from_numpy(self.joint_lower_limit).cuda()
            )
            - 0.5
        ) * 2
