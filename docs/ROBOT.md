# Robot Design

## 21-DoF Model Choice

In CombatBench, the robots utilize a **21 Degrees of Freedom (DoF)** configuration based on the [official MuJoCo humanoid model](https://github.com/google-deepmind/mujoco/blob/main/model/humanoid/humanoid.xml). 

Note that standard Gymnasium environments (e.g., Humanoid-v5) typically use a 17-DoF model that lacks ankle joints. We specifically chose the 21-DoF model for the following reasons:

### 1. Physical Dynamics: Ankle Joints and Directional Changes
- **Ankle Advantage:** The 21-DoF model has 4 additional degrees of freedom located in the ankles (X and Y axis rotational drives for both feet).
- **Critical Role in Combat:** In combat scenarios (fighting, dodging, quick dashes), the ankles are the core for providing explosive propulsion and maintaining dynamic balance.
- **Lateral Movement:** The 17-DoF model lacks ankle drives, meaning that during intense lateral movement or when receiving lateral impacts, it must compensate by swinging its torso and hips, resulting in stiff movements and making it prone to falling over.

### 2. Action Space: Agility in Attack and Defense
- **Limitations of 17-DoF:** It was designed as a simplified version meant primarily for basic "walking" tasks. In a combat scenario, if the robot needs to execute complex footwork or lower its center of gravity (e.g., dodging in boxing), a 17-DoF robot might fall into a "balance trap", sacrificing wide attacking motions just to maintain stability.
- **Anthropomorphism of 21-DoF:** State-of-the-art research on highly difficult control tasks (such as a humanoid wielding a badminton racket) typically favors a 21-DoF or higher configuration to ensure agility and energy efficiency.

### 3. Trade-offs in Reinforcement Learning (RL) Training
| Feature | 17-DoF (Humanoid-v5) | 21-DoF (MuJoCo Prototype) |
|---|---|---|
| **Convergence Speed** | Faster, smaller action space | Slower, needs more samples for foot coordination |
| **Balance Stability** | Worse, prone to "physics explosions" | Better, allows fine pressure adjustments |
| **Sim-to-Real** | Extremely difficult (real robots have ankles) | Better, physical topology closer to real hardware |

### Conclusion
For combat platforms, enabling the ankles (21 DoF) is essential. The ability to fine-tune foot placement is the key to winning in fierce robotic combat. A 17-DoF robot engaged in combative sports behaves like a "wooden puppet on ice skates". Therefore, CombatBench strictly utilizes the 21-DoF model to maximize the ceiling of agility and realism.
