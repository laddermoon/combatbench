# Simulation Environment

The simulation environment supports humanoid robots controlled with 21 Degrees of Freedom (DoF).

The simulation environment includes:

1. **Arena:**
   A fully enclosed room consisting of a floor, four walls, and a ceiling.
   - **Height:** 6.10 meters.
   - **Dimensions:** Official AIBA standard for amateur boxing - 6.10 meters (20 feet) square.
   - **Wall Texture:** Applied using a stretched image (`assets/textures/wall.png`) to fit the wall surfaces.
   - **Floor Texture:** Applied using a stretched image (`assets/textures/floor.png`) to fit the floor surface.
   - **Ceiling Texture:** Applied using a stretched image (`assets/textures/ceiling.png`) to fit the ceiling surface.

2. **Robots:**
   Two humanoid robots based on the official [MuJoCo humanoid model](https://github.com/google-deepmind/mujoco/blob/main/model/humanoid/humanoid.xml) (`assets/humanoid.xml`), with modifications made only to their visual colors.
   - One robot is colored red, and the other is colored blue.
   - The robots are placed on the center line of the room, standing face-to-face, each 1 meter away from the center point (a total of 2 meters apart).
   - The initial state of both robots is standing completely upright.

3. **Lighting:**
   Four light sources, with one located in each corner of the room at a height of 5 meters.

4. **Fixed Cameras:**
   - One camera in each corner, at a height of 4 meters, pointing towards the center of the floor (4 cameras in total).
   - One camera at the middle of each wall, at a height of 3 meters, pointing towards the center of the floor (4 cameras in total).
   - One overhead camera placed exactly in the center of the ceiling, pointing directly downwards.
