"""
Scoring Module (HP Mechanism) - 21DOF Version

Pure HP version: each robot starts with 100 HP, deducted when hit.
No fall judgment, no fouls, outcome is determined solely by HP.
"""

class ScoreCalculator:
    """
    Score Calculator (V1.0 Pure HP Minimalist Version)

    Initial HP: 100 points per robot

    Damage rules (based on hit part):
    - Head: -3 points
    - Torso: -1 points
    - 四肢: 不掉血（Attacking parts，不是Target parts）
    """

    INITIAL_HEALTH = 100

    # Damage rules
    DAMAGE_RULES = {
        'head': -3,
        'torso': -1,
    }

    def __init__(self, damage_scale=100.0):
        self.damage_scale = float(damage_scale)
        self.health = {
            'robot_a': self.INITIAL_HEALTH,
            'robot_b': self.INITIAL_HEALTH,
        }

    def take_damage(self, robot, hit_part, impulse):
        """
        Take damage

        Args:
            robot: 'robot_a' 或 'robot_b'
            hit_part: Hit part ('head', 'torso')
            
        Returns:
            damage: Damage value caused (0 means no HP deduction)
        """
        damage_weight = -float(self.DAMAGE_RULES.get(hit_part, 0))
        if damage_weight <= 0.0:
            return 0.0

        damage = -damage_weight * float(impulse) / self.damage_scale

        if damage < 0:
            self.health[robot] += damage
            if self.health[robot] < 0:
                self.health[robot] = 0

        return damage

    def get_health(self, robot=None):
        if robot is None:
            return self.health.copy()
        return self.health[robot]

    def is_alive(self, robot):
        return self.health[robot] > 0

    def check_match_over(self):
        """
        Check if match is over (based on HP, KO judgment)
        """
        if not self.is_alive('robot_a'):
            return True, 'robot_b', 'Robot A HP reached zero (KO)'
        if not self.is_alive('robot_b'):
            return True, 'robot_a', 'Robot B HP reached zero (KO)'

        return False, None, None

    def get_winner_by_health(self):
        """Get winner based on HP (for round/match end judgment)"""
        if self.health['robot_a'] > self.health['robot_b']:
            return 'robot_a'
        elif self.health['robot_b'] > self.health['robot_a']:
            return 'robot_b'
        else:
            return 'draw'

    def reset(self):
        """重置得points"""
        self.health = {
            'robot_a': self.INITIAL_HEALTH,
            'robot_b': self.INITIAL_HEALTH,
        }
