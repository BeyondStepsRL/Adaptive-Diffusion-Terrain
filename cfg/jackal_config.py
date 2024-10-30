from cfg.base_config import BaseVehicleCfg

class JackalCfg(BaseVehicleCfg):    
    class rigid_body_base(BaseVehicleCfg.context.rigid_body_base):
        base_mass_displacement = [-5, 20]
        base_inertia_displacement_x = [-0.2, 1.0]
        base_inertia_displacement_y = [-0.2, 1.0]
        base_inertia_displacement_z = [-0.2, 1.0]

    class robot(BaseVehicleCfg.robot):
        base_width = 0.37559
        wheel_radius = 0.095
        base_height = 0.095
        name = 'jackal'
        action_scale_v = 1.0
        action_scale_w = 1.0

    class asset(BaseVehicleCfg.asset):
        asset_file = 'jackal/jackal.urdf'
        asset_name = 'jackal'
        flip_visual_attachments = False