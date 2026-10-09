# Simulation Environment

> Type: Contract

The simulation environment supports humanoid robots controlled with 21 Degrees of Freedom (DoF). The active arena model is `envs/humanoid21/battle_circular_v2.xml` (`Laddermoon_Arena_Circular`).

The simulation environment includes:

1. **Arena:**
   A fully enclosed circular arena consisting of a floor, a circular wall built from 24 angled plane segments (`wall_00`–`wall_23`), and a ceiling.
   - **Height:** 6.10 meters.
   - **Radius:** ≈ 3.44 meters (diameter ≈ 6.88 m — comparable in scale to the AIBA 6.10 m / 20 ft square boxing ring).
   - **Textures:** wall segments alternate between six textures (`envs/humanoid21/textures/wall_0.png`–`wall_5.png`); the floor uses `floor_circular.png` and the ceiling `ceiling.png` from the same directory.
   - **Physics options:** `timestep = 0.002 s` (500 Hz), `condim = 3`, `impratio = 10`.

2. **Robots:**
   Two 21-DoF humanoid robots derived from the official [MuJoCo humanoid model](https://github.com/google-deepmind/mujoco/blob/main/model/humanoid/humanoid.xml), defined inline in the arena XML with color/geometry customizations.
   - `robot_a` is colored red, `robot_b` is colored blue.
   - The robots are placed facing each other across the arena center, `initial_distance` (default 2.0 m) apart.
   - The initial state of both robots is standing completely upright.

3. **Lighting:**
   The model uses the MuJoCo headlight (`ambient = diffuse = 0.4`) plus a gradient skybox. There are no discrete light fixtures.

4. **Fixed Cameras:**
   Nine fixed cameras on the arena boundary:
   - Four diagonal cameras at `(±2.22, ±2.22, 4.0)` (`cam_diag_0`–`cam_diag_3`), pointing towards the arena center.
   - Four cardinal cameras at `(±3.14, 0, 3.0)` / `(0, ±3.14, 3.0)` (`cam_east`/`south`/`west`/`north`), pointing towards the arena center.
   - One overhead camera (`ceilingcamera`) at `(0, 0, 6.0)`, pointing directly downwards.
   - Additionally, each robot carries `back`/`side` tracking cameras (`mode="trackcom"`) and an `egocentric` first-person camera.
   - The default broadcast view used by `get_broadcastview_image()` is a dynamic auto-framing camera, not one of the fixed cameras.
