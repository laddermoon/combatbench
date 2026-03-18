"""
得分计算模块（血量机制） - 21DOF 版本

纯血条版本：每个机器人初始 100 点血量，被击中扣分。
不判断摔倒、不设犯规，一切由血量决定胜负。
"""

class ScoreCalculator:
    """
    得分计算器（V1.0 纯血条极简版）

    初始血量: 每个机器人 100 分

    伤害规则（基于被击中部位）:
    - 头部: -3 分
    - 躯干: -1 分
    - 四肢: 不掉血（攻击部位，不是受击部位）
    """

    INITIAL_HEALTH = 100

    # 伤害规则
    DAMAGE_RULES = {
        'head': -3,
        'torso': -1,
    }

    def __init__(self):
        self.health = {
            'robot_a': self.INITIAL_HEALTH,
            'robot_b': self.INITIAL_HEALTH,
        }

    def take_damage(self, robot, hit_part):
        """
        受到伤害

        Args:
            robot: 'robot_a' 或 'robot_b'
            hit_part: 被击中的部位 ('head', 'torso')
            
        Returns:
            damage: 造成的伤害值（0表示不掉血）
        """
        damage = self.DAMAGE_RULES.get(hit_part, 0)

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
        检查对决是否结束（基于血量，KO判定）
        """
        if not self.is_alive('robot_a'):
            return True, 'robot_b', 'Robot A 血量归零 (KO)'
        if not self.is_alive('robot_b'):
            return True, 'robot_a', 'Robot B 血量归零 (KO)'

        return False, None, None

    def get_winner_by_health(self):
        """基于血量获取获胜者（用于回合/比赛结束时判定）"""
        if self.health['robot_a'] > self.health['robot_b']:
            return 'robot_a'
        elif self.health['robot_b'] > self.health['robot_a']:
            return 'robot_b'
        else:
            return 'draw'

    def reset(self):
        """重置得分"""
        self.health = {
            'robot_a': self.INITIAL_HEALTH,
            'robot_b': self.INITIAL_HEALTH,
        }
