"""E1 scene variant with the service bell 7 cm closer to the robot."""

import sapien

from .restaurant_pass_counter import restaurant_pass_counter


class restaurant_pass_counter_bell_near(restaurant_pass_counter):
    def load_actors(self):
        super().load_actors()
        pose = self.bell.get_pose()
        self.bell.actor.set_pose(
            sapien.Pose([pose.p[0], pose.p[1] - 0.07, pose.p[2]], pose.q)
        )
