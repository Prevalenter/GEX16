# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

# Many of these are stealed from Sapien or Maniskill repo.
# Thank you!! sapien developers


# This API is used in collision classifier training.
# -- pending updates.
def check_contact(scene, actors1, actors2, impulse_threshold=1e-10):

    actor_set1 = set(actors1)
    actor_set2 = set(actors2)
    for contact in scene.get_contacts():
        contact_actors = {contact.actor0, contact.actor1}
        if (
            len(actor_set1 & contact_actors) > 0
            and len(actor_set2 & contact_actors) > 0
        ):
            impulse = [point.impulse for point in contact.points]
            if np.sum(np.abs(impulse)) < impulse_threshold:
                continue
            return True
    return False


def get_active_joint_indices(articulation, joint_names):
    all_joint_names = [x.name for x in articulation.get_active_joints()]
    joint_indices = [all_joint_names.index(x) for x in joint_names]
    return joint_indices


def get_active_joints(articulation, joint_names):
    joints = articulation.get_active_joints()
    joint_indices = get_active_joint_indices(articulation, joint_names)
    return [joints[idx] for idx in joint_indices]


def get_entity_by_name(entities, name: str, is_unique=True):
    matched_entities = [x for x in entities if x.get_name() == name]
    if len(matched_entities) > 1:
        if not is_unique:
            return matched_entities
        else:
            raise RuntimeError(f"Multiple entities with the same name {name}.")
    elif len(matched_entities) == 1:
        return matched_entities[0]
    else:
        return None
