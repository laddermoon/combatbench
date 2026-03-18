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
- Hands
- Forearms
- Elbows
- Upper arms
- Feet
- Shins
- Knees
- Thighs

**Note:** The torso and head cannot be used as valid attacking parts. Striking the opponent with the torso or head will not deduct the opponent's HP.

### 2. Valid Target Parts (Defender)
HP is deducted only when the following parts are struck:
- Head
- Torso (including chest, abdomen, waist, and back)

Strikes to any other parts will not cause HP deduction.

### 3. Physical Conditions (True Strike Judgment)
Both of the following must be met simultaneously:
- **High-speed instant collision:** The relative collision velocity must be greater than the set threshold (excluding slow touches or pushing).
- **Non-continuous contact:** A single collision event resolves damage only once. Continuous contact/clinching will not trigger repeated HP deductions.

### 4. Damage Values
- **Head Hit:** -3 HP
- **Torso Hit:** -1 HP

*Damage is directly triggered by valid collisions, with no distinction between light and heavy strikes in this simplified V1.0 rule set.*

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
