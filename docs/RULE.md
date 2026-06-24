# CombatBench Rules V1.0 (Health Point Focused)

## I. Objective
Control a bipedal robot to deplete the opponent's health points (HP) through valid strikes. The first to reduce the opponent's HP to 0 wins.
There are no knock-down rules, no counts, no fouls, and no posture interventions. The outcome is solely determined by HP.

## II. Win/Loss Rules
1. **Initial HP:** 100 points per robot.
2. **Win Conditions:**
   - **2.1 KO Victory:** If a robot's HP is reduced to 0, the match ends immediately.
   - **2.2 Time Limit:** At the end of the match time, the robot with higher HP wins.
   - **2.3 Draw:** If HP is equal at the end of the time limit, it is declared a draw.
3. **Match Structure:** Each round lasts **30 seconds**, with a total of **6 rounds**.
4. **Reset State:** At the beginning of each round, both robots start from the initial position (standing face-to-face, 2 meters apart), regardless of their state at the end of the previous round.

## III. Valid Strike Judgment (The Only HP Deduction Logic)

### 1. Allowed Attacking Parts (Attacker)
Only strikes initiated by the following parts can cause damage:
- Hand
- Foot

**Note:** The torso and head cannot be used as valid attacking parts. Striking the opponent with the torso or head will not deduct the opponent's HP.

### 2. Valid Target Parts (Defender)
HP is deducted only when the following parts are struck:
- Head
- Torso (including chest, abdomen, upper waist, and lower waist)

Strikes to any other parts will not cause HP deduction.

### 3. Physical Conditions (Force Threshold Judgment)

Damage is computed in real-time per physics substep (500Hz, dt=0.002s). The damage formula is:

```
damage = part_weight × (force / force_scale)² × dt
```

Where:
- `force`: normal contact force at the physics substep (N)
- `force_scale`: force threshold parameter, default 100N
- `dt`: physics timestep, 0.002s
- `part_weight`: body part weight — head 3.0, torso 1.0

The `(force / force_scale)²` term creates a quadratic threshold:
- Forces below 100N are suppressed (e.g., 50N → 0.25×, negligible damage)
- Force equal to 100N is the threshold point (1.0×)
- Forces well above 100N are amplified (e.g., 200N → 4.0×, 400N → 16.0×)

This means light touches, slow pushes, and sustained contact produce negligible damage, while high-speed heavy strikes cause significant damage.

### 4. Damage Values

Damage is not a fixed value but is continuously calculated by the formula above. Reference magnitudes (force_scale=100N, dt=0.002s):

| Contact Force | Head Damage/Substep | Torso Damage/Substep |
|---------------|---------------------|----------------------|
| 50N           | 0.00075             | 0.00025              |
| 100N          | 0.006               | 0.002                |
| 500N          | 0.15                | 0.05                 |
| 1000N         | 0.6                 | 0.2                  |

A 1000N strike sustained for 50ms (25 substeps) deals approximately 15 HP to the head, 5 HP to the torso.

*The head weight is 3× the torso weight, so head damage is always 3× torso damage for the same force.*

## IV. Posture & Behavior Rules (No Restrictions)
The following behaviors will **not** result in a loss, point deduction, penalty, or reset:
- Falling, rolling, ground-and-pound, or ground defense.
- Clinching, close-quarters combat, pinning, or pulling.
- Headbutting or torso ramming (no HP deduction, but perfectly allowed).
- Any posture or movement style.

AI is free to evolve the optimal strategy without the need to mimic human martial arts strictly.

## V. Physics & Execution Rules
1. **Physics Step:** Fixed at 500Hz to ensure physics consistency.
2. **Policy Decision Frequency:** 20Hz (one step every 50ms).
3. **Timeout Behavior:** If no action is output within the required time, the previous action is automatically maintained.
4. **Consistency:** Global physics parameters and model parameters are strictly identical for both sides to ensure fairness.
