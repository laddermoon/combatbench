# Observation Space Design

## I. Overall Architecture

```
Observation Space
│
├── Module 1: Proprioception (42 dims)
│   ├── Joint Angles (21 dims)
│   └── Joint Velocities (21 dims)
│
├── Module 2: Root State (13 dims)
│   ├── Position and Orientation (7 dims)
│   └── Linear & Angular Velocity (6 dims)
│
├── Module 3: Tactile & Force Feedback (8 dims)
│   ├── Foot Contact (2 dims)
│   └── External Forces (6 dims)
│
└── Module 4: Opponent Observation (64 dims)
    ├── Opponent Base Pose (10 dims)
    ├── Opponent Keypoint Positions (27 dims)
    └── Opponent Keypoint Velocities (27 dims)

Total: 127 Dimensions
```

---

## II. Module Definitions

### 2.1 Module 1: Proprioception (42 dims)

**Function:** The robot's "muscle sense", determining the accuracy of limb control.

#### Joint Angles (21 dims)
| Joint Group | Indices | Joint Names | Description |
|---|---|---|---|
| Core Torso | 0-2 | abdomen_z/y/x | Waist rotation, pitch, lateral bending |
| Right Leg | 3-8 | hip_x/z/y, knee, ankle_y/x | Support, movement, kicking |
| Left Leg | 9-14 | hip_x/z/y, knee, ankle_y/x | Symmetrical to right leg |
| Right Arm | 15-17 | shoulder1/2, elbow | Attacking side, strike trajectory |
| Left Arm | 18-20 | shoulder1/2, elbow | Defensive side, auxiliary attack |

**Unit:** Radians

#### Joint Velocities (21 dims)
Corresponding to the angular velocities of the 21 joints above.

**Unit:** rad/s
**Function:** Determines the momentum of actions, such as the velocity of a heavy punch.

---

### 2.2 Module 2: Root State (13 dims)

**Function:** The robot's "spatial sense", describing its situation in the world.

#### Position and Orientation (7 dims)
| Data Item | Dims | Description |
|---|---|---|
| Height (Z-axis) | 1 | Judges if fallen, core combat metric |
| Local Orientation | 6 | World quaternion → Local rotation matrix (first two columns) |

**Note:** Absolute X/Y coordinates are intentionally omitted to avoid position dependency.

#### Velocity (6 dims)
| Data Item | Dims | Description |
|---|---|---|
| Linear Velocity | 3 | $v_x, v_y, v_z$, judges dashing or being knocked back |
| Angular Velocity | 3 | Equivalent to a gyroscope, sensing shaking when hit |

---

### 2.3 Module 3: Tactile & Force Feedback (8 dims)

**Function:** The robot's "somatosensory", learning blocking and force generation.

| Data Item | Dims | Description |
|---|---|---|
| Foot Contact | 2 | Whether left/right foot is touching the ground (0 or 1) |
| External Forces | 6 | Force vectors on torso, head, hands, and feet |

**Simplification Tip:** Taking only the force magnitude (scalar) can reduce dimensionality to accelerate training.

---

### 2.4 Module 4: Opponent Observation (64 dims)

**Principle:** Egocentric + Kinematic

**Coordinate System:** All opponent coordinates are transformed into the ego robot's local coordinate frame.

#### 2.4.1 Opponent Base Pose (10 dims)
| Data Item | Dims | Physical Meaning |
|---|---|---|
| Relative Position | 3 | Opponent torso center - Ego torso center |
| Relative Velocity | 3 | Judges if opponent is rushing or retreating |
| Relative Orientation | 4 | Quaternion difference, judges if facing each other |

#### 2.4.2 Opponent Keypoint Positions (27 dims)
| Keypoint | Dims | Role |
|---|---|---|
| Head | 3 | High-level attack target |
| Left/Right Hands | 6 | Detects hooks, blocks, or lowered arms |
| Left/Right Elbows | 6 | Blocking stance detection |
| Left/Right Knees | 6 | Kick warning |
| Left/Right Feet | 6 | Footwork judgment |

#### 2.4.3 Opponent Keypoint Velocities (27 dims)
**Role:** Distinguishes between "arm swaying" and "incoming heavy punches".

| Keypoint | Dims |
|---|---|
| Head Velocity | 3 |
| Left/Right Hand Velocity | 6 |
| Left/Right Elbow Velocity | 6 |
| Left/Right Knee Velocity | 6 |
| Left/Right Foot Velocity | 6 |

**Critical Value:**
- Speed = 0 → Static blocking stance
- High-speed approach → Imminent hit, dodge required
- Velocity direction → Strike trajectory prediction

---

## III. Notes

1. **Coordinate Transformation:** All opponent observations must be transformed to the ego's local coordinate system.
2. **Simplification Strategy:** For early baselines, external force observations can be simplified to scalar magnitudes.
3. **Avoid Position Dependency:** Absolute world X/Y coordinates are intentionally omitted from observations.
