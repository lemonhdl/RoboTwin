from __future__ import annotations

import sapien

from ._base_task import Base_Task
from .utils import *

LEFT_HOME_STATE = [-0.30, 0.20, 0.55, -2.20, 0.0, 2.55, 0.785398]
RIGHT_HOME_STATE = [0.30, 0.20, -0.55, -2.20, 0.0, 2.55, 0.785398]


def _loose(actor, mass=0.03):
    actor.set_mass(mass)
    for component in actor.actor.get_components():
        if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
            component.max_depenetration_velocity = 0.5


class laboratory_scene(Base_Task):
    """Laboratory layout using the restaurant double-Franka camera/embodiment setup."""

    def setup_demo(self, **kwargs):
        kwargs = kwargs.copy()
        left = kwargs["left_embodiment_config"].copy()
        right = kwargs["right_embodiment_config"].copy()
        left["homestate"] = [LEFT_HOME_STATE.copy(), RIGHT_HOME_STATE.copy()]
        right["homestate"] = [LEFT_HOME_STATE.copy(), RIGHT_HOME_STATE.copy()]
        kwargs["left_embodiment_config"] = left
        kwargs["right_embodiment_config"] = right
        kwargs["table_color_override"] = [0.025, 0.12, 0.055]
        # The widened bench depth is 0.9217 m. Move its center back by 7 cm
        # so its rear edge stops about 9 mm before the wall front plane (y=.4).
        kwargs["table_xy_bias"] = [0.0, -0.07]
        super()._init_task_env_(**kwargs)

    def create_table_and_wall(
        self, table_xy_bias=[0, 0], table_height=0.74,
        table_texture_override=None, table_color_override=None,
    ):
        """Use the supplied laboratory bench asset instead of the generic box table."""
        del table_height, table_texture_override, table_color_override
        self.table_xy_bias = table_xy_bias
        self.wall_texture = None
        self.table_texture = None
        self.wall = create_box(
            self.scene, sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5], color=(1, 0.9, 0.9),
            name="wall", is_static=True,
        )
        # The converted bench visual/collision package has its support plane at
        # local z=0 and its tabletop at semantic z=0.74 m.  preprocess() adds
        # table_z_bias, so cancel it here to keep the feet on the ground.
        self.table = create_actor(
            self, sapien.Pose([table_xy_bias[0], table_xy_bias[1], -self.table_z_bias]),
            "lab_experiment_bench", convex=False, is_static=True,
        )

    def load_actors(self):
        # RoboTwin create_actor poses are actor-root poses.  The converted GLBs
        # keep their semantic local Z origin, so placing every root at table_z
        # makes the visible mesh float.  These offsets are the verified local
        # mesh minima from conversion_report.json.
        table_top = 0.74
        rack_root = table_top - 0.16
        # The lower shelf is the support surface.  The upper perforated
        # shelf is a guide; it should not carry the tube.  In the converted
        # rack these surfaces are at local z~=0.184 m and z~=0.285 m, while
        # the tube local bottom is z~=0.12 m.
        tube_root = rack_root + 0.184 - 0.12
        balance_scale = 0.70
        stand_scale = 1.25
        balance_root = table_top - 0.10 * balance_scale
        beaker_root = table_top - 0.10
        stand_root = table_top - 0.40 * stand_scale
        flask_root = table_top - 0.18
        # The normal reset keeps the lamp beside the stand. The on-stand
        # support pose is used only by the separate placement probe.
        lamp_side_root = table_top

        # Back/left: rack with four upright tubes. Each tube bottom rests
        # on the lower shelf; the upper perforated shelf only constrains tilt.
        self.test_tube_rack = create_actor(
            self, sapien.Pose([-0.38, 0.18, rack_root]),
            "lab_test_tube_rack", convex=False, is_static=True
        )
        self.test_tubes = []
        # The supplied rack has four holes, centered at these local x values.
        # Keep the tube axes on the holes; x=0 is not a hole center.
        for i, xoff in enumerate((-0.106, -0.036, 0.036, 0.106)):
            tube = create_actor(
                self, sapien.Pose([-0.38 + xoff, 0.18, tube_root]),
                "lab_test_tube", convex=True, is_static=False
            )
            tube.set_name(f"lab_test_tube_{i}")
            _loose(tube, 0.02)
            self.test_tubes.append(tube)

        # Front: analytical balance and a beaker beside it, both resting on
        # the tabletop rather than using the actor-root height as the contact.
        self.analytical_balance = create_actor(
            self, sapien.Pose([-0.08, 0.18, balance_root]),
            "lab_analytical_balance", convex=False, is_static=True,
            scale_multiplier=balance_scale
        )
        self.beaker = create_actor(
            self, sapien.Pose([-0.15, -0.22, beaker_root]),
            "lab_beaker", convex=True, is_static=False
        )
        _loose(self.beaker, 0.03)

        # Right/back: stand, flask, and alcohol lamp with a separately movable
        # cap. The initial lamp remains beside the stand; a separate probe
        # places it on the base and tests the collision relationship.
        self.iron_stand = create_actor(
            self, sapien.Pose([0.22, 0.18, stand_root]),
            "lab_iron_stand", convex=False, is_static=True,
            scale_multiplier=stand_scale
        )
        self.alcohol_lamp_body = create_actor(
            self, sapien.Pose([0.48, 0.18, lamp_side_root]),
            "lab_alcohol_lamp_body", convex=True, is_static=False
        )
        self.alcohol_lamp_cap = create_actor(
            self, sapien.Pose([0.48, 0.18, lamp_side_root]),
            "lab_alcohol_lamp_cap", convex=True, is_static=False
        )
        self.alcohol_lamp_cap.set_name("lab_alcohol_lamp_cap")
        _loose(self.alcohol_lamp_body, 0.05)
        _loose(self.alcohol_lamp_cap, 0.01)
        self.conical_flask = create_actor(
            self, sapien.Pose([0.18, -0.20, flask_root]),
            "lab_conical_flask", convex=False, is_static=True
        )

    def play_once(self):
        # Task proposed by the independent Astra reviewer: pick the beaker and
        # place it on the analytical balance pan.
        arm = ArmTag("left")
        self.move(self.grasp_actor(self.beaker, arm_tag=arm, pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm_tag=arm, z=0.06))
        self.move(self.place_actor(
            self.beaker, arm_tag=arm, target_pose=self.analytical_balance.get_pose(),
            constrain="free", pre_dis=0.08
        ))
        self.info["info"] = {
            "task": "pick up the white beaker and place it on the analytical balance pan",
            "arm": "left",
        }
        return self.info

    def check_success(self):
        # Scene construction is supported; a validated laboratory task predicate
        # is not implemented. XY proximity alone must not claim task success.
        return False
