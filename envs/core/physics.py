"""
Physics Engine Wrapper

Provides basic functions for MuJoCo physics simulation.
"""

import mujoco
import numpy as np


class PhysicsEngine:
    """
    MuJoCo Physics Engine Wrapper

    Responsible for managing the lifecycle and configuration of physics simulation.
    """

    def __init__(self, gui=True, dt=0.002, arena_xml=None):
        """
        Initialize physics engine

        Args:
            gui: Whether to display GUI
            dt: Physics step time (seconds)，default 0.002 (500Hz)
            arena_xml: Arena XML file path
        """
        self.dt = dt
        self.gui = gui
        self.arena_loaded = False

        # Load model from XML
        if arena_xml:
            self.model = mujoco.MjSpec.from_file(arena_xml).compile()
            self.arena_loaded = True
            print(f"PhysicsEngine: Arena loaded from {arena_xml}")
        else:
            # 创建default地面
            spec = mujoco.MjSpec()
            spec.worldbody.add('geom', type='plane', size=[20, 20, 0.1],
                               rgba=[0.8, 0.9, 0.8, 1])
            spec.opt.timestep = dt
            spec.opt.iterations = 50
            spec.opt.solver = 'PGS'
            spec.opt.integrator = 'RK4'
            self.model = spec.compile()
            print("PhysicsEngine: Using default ground plane")

        # Create data object
        self.data = mujoco.MjData(self.model)

        # GUI mode
        if gui:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data
            )
            print("PhysicsEngine: GUI viewer launched")
        else:
            self.viewer = None

        print(f"PhysicsEngine: Initialized with timestep={dt}")

    def step(self):
        """
        Execute one physics step
        """
        mujoco.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    def reset(self):
        """
        Reset physics engine state

        Note: This does not reset loaded objects, only the physics state
        """
        # MuJoCo requires recreating MjData to fully reset
        self.data = mujoco.MjData(self.model)

    def close(self):
        """
        Close physics engine
        """
        if self.viewer:
            del self.viewer
            self.viewer = None
            print("PhysicsEngine: Viewer closed")

    def get_contact_points(self, body_a, body_b):
        """
        Get contact points between two bodies

        Args:
            body_a: Index of the first body
            body_b: Index of the second body

        Returns:
            contacts: List of contact points, each element contains (geom1, geom2, position, normal, force)
        """
        contacts = []

        # Iterate through all contact points
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Get body corresponding to geom
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            # Check if contact is between target bodies
            if (body1 == body_a and body2 == body_b) or \
               (body1 == body_b and body2 == body_a):
                # Get contact position
                contact_pos = self.data.contact[i].pos
                contact_normal = self.data.contact[i].frame

                # Get contact force (simplified version, actual requires complex calculation)
                # MuJoCo contact force calculation is complex, returning position info here
                contacts.append({
                    'geom_a': geom1,
                    'geom_b': geom2,
                    'body_a': body1,
                    'body_b': body2,
                    'position': contact_pos,
                    'normal': contact_normal,
                })

        return contacts
