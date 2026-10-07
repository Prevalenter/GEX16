# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import sapien.core as sapien

from _vendor.geort.utils.hand_utils import (
    get_active_joint_indices,
    get_active_joints,
    get_entity_by_name,
)


class HandKinematicModel:
    def __init__(
        self,
        scene=None,
        render=False,
        hand=None,
        hand_urdf="",
        n_hand_dof=16,
        base_link="base_link",
        joint_names=[],
        # Ideally, these two guys (PD controller args) shouldn't be here.
        # -- There should be a controller class. I leave them here for code simplicity (maybe truth: or because I am lazy).
        # If you see your hand model doing something weird (in the simulation viewer below), tune them.
        kp=400.0,
        kd=10,
    ):

        self.engine = None
        renderer = None
        if scene is None:
            engine = sapien.Engine()

            if render:
                renderer = sapien.VulkanRenderer()
                engine.set_renderer(renderer)
                print("Enable Render Mode.")
            else:
                renderer = None
            scene_config = sapien.SceneConfig()
            scene_config.default_dynamic_friction = 1.0
            scene_config.default_static_friction = 1.0
            scene_config.default_restitution = 0.00
            scene_config.contact_offset = 0.02
            scene_config.enable_pcm = False
            scene_config.solver_iterations = 25
            scene_config.solver_velocity_iterations = 1
            scene = engine.create_scene(scene_config)
            self.engine = engine

        self.scene = scene
        self.renderer = renderer

        if hand is not None:
            self.hand = hand

        else:
            loader = scene.create_urdf_loader()
            self.hand = loader.load(hand_urdf)
            self.hand.set_root_pose(sapien.Pose([0, 0, 0.35], [0.695, 0, -0.718, 0]))

        root_pose = self.hand.get_root_pose()
        self.collision_check_root_pose = sapien.Pose(root_pose.p, root_pose.q)
        self.pmodel = self.hand.create_pinocchio_model()

        # Setup hand base link.
        self.base_link = get_entity_by_name(self.hand.get_links(), base_link)
        self.base_link_idx = self.hand.get_links().index(self.base_link)

        # Setup hand dofs.
        self.simulation_joints = self.hand.get_active_joints()
        self.all_joints = get_active_joints(self.hand, joint_names)
        all_limits = [joint.get_limits() for joint in self.all_joints]

        self.joint_names = joint_names
        self.user_idx_to_sim_idx = get_active_joint_indices(self.hand, joint_names)
        print("User-to-Sim Joint", self.user_idx_to_sim_idx)
        self.sim_idx_to_user_idx = {
            sim_idx: user_idx
            for user_idx, sim_idx in enumerate(self.user_idx_to_sim_idx)
        }
        print("Sim-to-User Joint", self.sim_idx_to_user_idx)

        # URDFs may expose wrist joints even when GeoRT is configured in a
        # palm-local frame. Keep omitted joints fixed at a valid zero pose.
        simulation_limits = np.asarray(
            [joint.get_limits()[0] for joint in self.simulation_joints],
            dtype=np.float64,
        )
        self.fixed_sim_qpos = np.clip(
            np.zeros(len(self.simulation_joints)),
            simulation_limits[:, 0],
            simulation_limits[:, 1],
        )

        self.joint_lower_limit = np.array(
            [l[0][0] for l in all_limits]
        )  # this is in user specified "joint_name" order
        self.joint_upper_limit = np.array(
            [l[0][1] for l in all_limits]
        )  # this is in user specified "joint_name" order
        print(self.joint_lower_limit, self.joint_upper_limit)

        init_qpos = self.convert_user_order_to_sim_order(
            (self.joint_lower_limit + self.joint_upper_limit) / 2
        )
        self.hand.set_qpos(init_qpos)
        self.hand.set_qvel(0.0 * init_qpos)
        self.qpos_target = init_qpos

        for i, joint in enumerate(self.all_joints):
            print(
                i,
                self.joint_names[i],
                joint,
                self.joint_lower_limit[i],
                self.joint_upper_limit[i],
            )
            joint.set_drive_property(kp, kd, force_limit=10)

    def __del__(self):
        del self.engine
        del self.scene

    def get_n_dof(self):
        """
        number of dof.
        """
        return len(self.joint_lower_limit)

    def get_joint_limit(self):
        """
        Get the hand joint limit.
        """
        return self.joint_lower_limit, self.joint_upper_limit

    def convert_user_order_to_sim_order(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (len(self.user_idx_to_sim_idx),):
            raise ValueError(
                f"Expected {len(self.user_idx_to_sim_idx)} configured joints, "
                f"got {qpos.shape}"
            )
        simulation_qpos = self.fixed_sim_qpos.copy()
        simulation_qpos[self.user_idx_to_sim_idx] = qpos
        return simulation_qpos

    def self_collision_penetrations(self, qpos):
        """Return maximum penetration depth for every contacting self-link pair."""
        qpos = np.asarray(qpos)
        if qpos.shape != (self.get_n_dof(),):
            raise ValueError(
                f"Expected qpos with shape ({self.get_n_dof()},), got {qpos.shape}"
            )
        if not np.isfinite(qpos).all():
            raise ValueError("qpos contains NaN or Inf")

        simulation_qpos = self.convert_user_order_to_sim_order(qpos)
        self.hand.set_root_pose(self.collision_check_root_pose)
        self.hand.set_root_velocity(np.zeros(3))
        self.hand.set_root_angular_velocity(np.zeros(3))
        self.hand.set_qpos(simulation_qpos)
        self.hand.set_qvel(np.zeros_like(simulation_qpos))
        self.scene.step()

        link_ids = {link.get_id() for link in self.hand.get_links()}
        penetrations = {}
        for contact in self.scene.get_contacts():
            if (
                contact.actor0.get_id() not in link_ids
                or contact.actor1.get_id() not in link_ids
            ):
                continue
            depths = [
                -float(point.separation)
                for point in contact.points
                if point.separation < 0
            ]
            if not depths:
                continue
            pair = tuple(sorted((contact.actor0.get_name(), contact.actor1.get_name())))
            penetrations[pair] = max(penetrations.get(pair, 0.0), max(depths))
        return penetrations

    def has_self_collision(
        self,
        qpos,
        penetration_threshold=5e-4,
        allowed_link_pairs=None,
        allowed_penetration_threshold=None,
    ):
        """Return whether a pose has self-penetration beyond a tolerance.

        SAPIEN reports contact points within its contact offset even when two
        shapes are still separated. Only negative separation deeper than the
        configured threshold is treated as a collision. Directly adjacent
        articulation links are already excluded by PhysX.
        """
        if penetration_threshold < 0:
            raise ValueError("penetration_threshold must be non-negative")

        if allowed_penetration_threshold is None:
            allowed_penetration_threshold = penetration_threshold
        if allowed_penetration_threshold < 0:
            raise ValueError("allowed_penetration_threshold must be non-negative")
        allowed = {tuple(sorted(pair)) for pair in (allowed_link_pairs or ())}
        for pair, depth in self.self_collision_penetrations(qpos).items():
            threshold = (
                allowed_penetration_threshold
                if pair in allowed
                else penetration_threshold
            )
            if depth > threshold:
                return True
        return False

    @staticmethod
    def build_from_config(config, **kwargs):
        """
        Build a kinematic model from user config.
        """
        render = kwargs.get("render", False)
        urdf_path = config["urdf_path"]
        n_hand_dof = len(config["joint_order"])
        base_link = config["base_link"]
        joint_order = config["joint_order"]

        model = HandKinematicModel(
            hand_urdf=urdf_path,
            render=render,
            n_hand_dof=n_hand_dof,
            base_link=base_link,
            joint_names=joint_order,
        )
        return model
