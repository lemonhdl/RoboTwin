"""Checkout scene construction; no expert policy or task-success evaluator.

Private mesh/poses are used only to build the scene. Operation remains through
public Auto RGB-D/EE observations. The product is one persistent rigid actor.
"""
import json
from pathlib import Path

import sapien

from ._base_task import Base_Task
from .utils import create_actor, create_box


DEFAULT_LAYOUT = Path(__file__).resolve().parents[1] / 'env_cfg/supermarket_checkout_counter.json'


class supermarket_checkout_counter(Base_Task):
    def setup_demo(self, **kwargs):
        kwargs = dict(kwargs)
        layout_path = Path(kwargs.pop('checkout_layout_path', DEFAULT_LAYOUT))
        self.layout = json.loads(layout_path.read_text())
        if self.layout['schema_version'] != 'supermarket_checkout_layout_v1':
            raise ValueError('unsupported checkout layout')
        for side in ('left', 'right'):
            key = f'{side}_embodiment_config'
            kwargs[key] = dict(kwargs[key], homestate=self.layout['home'])
        super()._init_task_env_(**kwargs)

    def _box(self, name, center, half_size, color):
        return create_box(scene=self, pose=sapien.Pose(center), half_size=half_size,
                          color=color, is_static=True, name=name)

    def create_table_and_wall(self, table_xy_bias=(0, 0), table_height=.74,
                              table_texture_override=None, table_color_override=None):
        # A fixed counter is part of this layout; no inherited table randomization.
        self.table_xy_bias = list(table_xy_bias)
        self.wall_texture = self.table_texture = None
        c = self.layout['counter']
        height = c['top_z']
        x, y = c['center_xy']
        hx, hy = c['half_size_xy']
        self.table = self._box('table', [x, y, height-.025], [hx, hy, .025], [.82,.85,.84])
        self.counter_body = self._box('checkout_cabinet', [x,y+.025,(height-.05)/2],
                                      [hx-.035,hy-.045,(height-.05)/2], [.12,.28,.26])
        self.counter_trim = self._box('checkout_trim', [x,y-hy+.004,height-.065],
                                      [hx,.006,.018], [.79,.63,.31])
        sx, sy = c['shelf_center_xy']; shx, shy = c['shelf_half_size_xy']
        self.basket_shelf = self._box('basket_shelf', [sx,sy,height-.025],
                                      [shx,shy,.025], [.82,.85,.84])
        self.shelf_body = self._box('basket_shelf_body', [sx,sy,(height-.05)/2],
                                    [shx-.015,shy-.025,(height-.05)/2], [.12,.28,.26])
        self.wall = self._box('wall', [0,1.3,1.5], [3,.1,1.5], [.87,.9,.88])

    def _asset(self, descriptor):
        # Missing required assets fail explicitly instead of rendering substitute boxes.
        root = Path('assets/objects') / descriptor['modelname']
        for relative in (f"visual/base{descriptor['model_id']}.glb",
                         f"collision/base{descriptor['model_id']}.glb",
                         f"model_data{descriptor['model_id']}.json"):
            if not (root / relative).is_file():
                raise FileNotFoundError(root / relative)
        extra = {}
        if 'collision_boxes' in descriptor:
            extra['collision_boxes'] = descriptor['collision_boxes']
        actor = create_actor(scene=self,
                             pose=sapien.Pose(descriptor['pose'], descriptor['quaternion_wxyz']),
                             modelname=descriptor['modelname'], model_id=descriptor['model_id'],
                             is_static=descriptor['is_static'], convex=descriptor['convex'], **extra)
        if 'mass_kg' in descriptor:
            actor.set_mass(descriptor['mass_kg'])
        return actor

    def load_actors(self):
        self.zone_pads = [self._box(d['name'], d['center'], d['half_size'], d['color'])
                          for d in self.layout['zone_pads']]
        pad = self.layout['scanner_pad']
        self.scanner_pad = self._box('scanner_pad', pad['center'], pad['half_size'], [.10,.12,.13])
        self.scale = self._asset(self.layout['scale'])
        self._create_scale_pan()
        # The static nonconvex mesh preserves the basket cavity and open handle.
        self.basket = self._asset(self.layout['basket'])
        self.scanner = self._asset(self.layout['scanner'])
        self.product = self._asset(self.layout['product'])
        self.object = self.product
        self.product.set_name(self.layout['product']['instance_id'])
        self._attach_product_label(self.product)
        self.incoming_products = []
        for stock in self.layout['incoming_products']:
            descriptor = dict(self.layout['product'], **stock)
            actor = self._asset(descriptor)
            actor.set_name(descriptor['instance_id'])
            self._attach_product_label(actor)
            self.incoming_products.append(actor)
        self.zone_signs = [self._asset(d) for d in self.layout['zone_signs']]

    def _create_scale_pan(self):
        scale = self.layout['scale']; pan = scale['pan']
        p = list(scale['pose'])
        p[2] += self.table_z_bias
        pan_pose = sapien.Pose(p, scale['quaternion_wxyz']) * sapien.Pose(pan['center'])
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type('static')
        builder.add_box_collision(half_size=pan['half_size'],
                                  material=self.scene.default_physical_material)
        builder.set_initial_pose(pan_pose)
        self.scale_pan = builder.build(name='checkout_scale_pan')
        # Collision only: the user's original textured plate supplies the visual.

    def _attach_product_label(self, actor):
        label = self.layout['product']['label']
        # SAPIEN 3 requires all shapes to be built before body attachment.
        # This component shares the product entity and therefore its rigid motion.
        render = sapien.render.RenderBodyComponent()
        cx, cy, cz = label['center']; hx, hy, hz = label['half_size']
        def add(center, half_size, rgb):
            shape = sapien.render.RenderShapeBox(
                half_size, sapien.render.RenderMaterial(base_color=[*rgb, 1]))
            shape.local_pose = sapien.Pose(center)
            render.attach(shape)
        add([cx,cy,cz], [hx,hy,hz], [.97,.97,.94])
        # A visible simulated label, not a claim of barcode decoding.
        for index, bit in enumerate('110100100011010110100110101'):
            if bit == '1':
                add([cx-hx*.84+index*(hx*1.68/26),cy,cz+hz*1.1],
                    [hx*.021,hy*.79,hz*.2], [.025,.025,.025])
        actor.actor.add_component(render)

    def check_success(self):
        # Scene availability must never be reported as a completed checkout order.
        return False
