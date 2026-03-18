# Scene Specification

## 1. Coordinate System
- Ground center point: `(0, 0, 0)`
- **Z-axis:** Positive direction is upwards.
- **X-axis:** Positive direction is East.
- **Y-axis:** Positive direction is South.

## 2. Walls, Floor, and Ceiling
- Constructed using MuJoCo `plane` geoms.
- **Definition:** Defined by the `x,y,z` coordinates of four corners along with the texture path. This defines the inner surface.
- The external surface and positioning are calculated automatically to ensure the inner surface normals point inward.

## 3. Texture Mapping
- When mapping an image onto a plane, a single image is stretched to cover the entire plane. 
- Images are not tiled or cropped; they are scaled to fit the plane entirely.

## 4. Camera Specification
- Except for top-down/bottom-up cameras, all cameras only pitch along a single horizontal axis.
- Standard orientation is landscape, with a resolution of `640x480`.
- Cameras are defined by two parameters:
  1. Position `(x, y, z)`.
  2. Target Look-At point `(x, y, z)`.

## 5. Scene Elements JSONL Definition
The arena is programmatically built from a `scene_elements.jsonl` file. 

Example definitions:

```jsonl
{"type": "plane", "name": "floor", "corners": [[-3.05, -3.05, 0], [3.05, -3.05, 0], [3.05, 3.05, 0], [-3.05, 3.05, 0]], "texture": "textures/floor.png"}
{"type": "plane", "name": "ceiling", "corners": [[-3.05, -3.05, 6.10], [3.05, -3.05, 6.10], [3.05, 3.05, 6.10], [-3.05, 3.05, 6.10]], "texture": "textures/ceiling.png"}
{"type": "plane", "name": "south_wall", "corners": [[-3.05, -3.05, 0], [3.05, -3.05, 0], [3.05, -3.05, 6.10], [-3.05, -3.05, 6.10]], "texture": "textures/wall.png"}
{"type": "plane", "name": "north_wall", "corners": [[-3.05, 3.05, 0], [3.05, 3.05, 0], [3.05, 3.05, 6.10], [-3.05, 3.05, 6.10]], "texture": "textures/wall.png"}
{"type": "plane", "name": "west_wall", "corners": [[-3.05, -3.05, 0], [-3.05, 3.05, 0], [-3.05, 3.05, 6.10], [-3.05, -3.05, 6.10]], "texture": "textures/wall.png"}
{"type": "plane", "name": "east_wall", "corners": [[3.05, -3.05, 0], [3.05, 3.05, 0], [3.05, 3.05, 6.10], [3.05, -3.05, 6.10]], "texture": "textures/wall.png"}

{"type": "light", "name": "sw_light", "position": [-3, -3, 5]}
{"type": "light", "name": "se_light", "position": [3, -3, 5]}
{"type": "light", "name": "nw_light", "position": [-3, 3, 5]}
{"type": "light", "name": "ne_light", "position": [3, 3, 5]}

{"type": "camera", "name": "sw_camera", "position": [-3, -3, 4], "look_at": [0, 0, 0]}
{"type": "camera", "name": "se_camera", "position": [3, -3, 4], "look_at": [0, 0, 0]}
{"type": "camera", "name": "nw_camera", "position": [-3, 3, 4], "look_at": [0, 0, 0]}
{"type": "camera", "name": "ne_camera", "position": [3, 3, 4], "look_at": [0, 0, 0]}
{"type": "camera", "name": "south_camera", "position": [0, -3, 3], "look_at": [0, 0, 0]}
{"type": "camera", "name": "north_camera", "position": [0, 3, 3], "look_at": [0, 0, 0]}
{"type": "camera", "name": "west_camera", "position": [-3, 0, 3], "look_at": [0, 0, 0]}
{"type": "camera", "name": "east_camera", "position": [3, 0, 3], "look_at": [0, 0, 0]}
```
