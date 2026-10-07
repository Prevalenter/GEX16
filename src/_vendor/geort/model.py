# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn


def get_finger_ik(n_joint=4, hidden=128):
    return nn.Sequential(
        nn.Linear(3, hidden),
        nn.LeakyReLU(),
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, hidden),
        nn.LeakyReLU(),
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, n_joint),
        nn.Tanh(),  # Normalize.
    )


class IKModel(nn.Module):
    def __init__(self, keypoint_joints):
        # keypoint_joints: a list of list.
        # keypoint[i] is the indices of joints that drive the i-th keypoint.
        # Example: [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]

        super().__init__()
        self.n_total_joint = 0
        self.nets = []

        for joint in keypoint_joints:
            net = get_finger_ik(n_joint=len(joint))
            self.nets.append(net)
            self.n_total_joint += len(joint)

        self.nets = nn.ModuleList(self.nets)
        self.keypoint_joints = keypoint_joints

    def forward(self, x):
        # x:   [B, N, 3], sequence of keypoint.
        # out: [B, DOF], joint values. normalized to [-1, 1].
        batch_size = x.size(0)
        n_points = x.size(1)
        out = torch.zeros((batch_size, self.n_total_joint)).to(x.device)
        for i in range(n_points):
            joint = self.nets[i](x[:, i])
            out[:, self.keypoint_joints[i]] = joint
        return out
