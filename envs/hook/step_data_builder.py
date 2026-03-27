
# ==================== 核心接口 ====================

class StepDataBuilder(BaseHook, ABC):
    """
    Step 数据构建器（作为 Hook 实现）

    在 POST_ACTION_STEP 时被调用，构建观测、奖励和 info。
    """

    @abstractmethod
    def build_step_data(
        self,
        f_get_core_state: Callable[[], Dict[str, Any]],
        f_get_derived_state: Callable[[], Dict[str, Any]],
        f_get_sensor_data: Callable[[], Dict[str, Any]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, Any]]:
        """
        构建观测、奖励和 info

        Args:
            f_get_core_state: 获取核心状态的函数
            f_get_derived_state: 获取衍生状态的函数
            f_get_sensor_data: 获取传感器数据的函数

        Returns:
            (observation, reward, info)
        """
        pass

    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """返回观测空间"""
        pass

    def get_last_data(self) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
        """获取最近构建的数据"""
        return (
            getattr(self, '_last_observation', None),
            getattr(self, '_last_reward', None),
            getattr(self, '_last_info', None),
        )

    # Hook 接口实现
    @property
    def priority(self) -> int:
        return -50  # 在 POST_ACTION_STEP 时执行

    def invoke(self, invoke_type: InvokeType, *args, **kwargs) -> bool:
        if invoke_type == InvokeType.POST_ACTION_STEP:
            f_get_core_state = kwargs.get('f_get_core_state')
            f_get_derived_state = kwargs.get('f_get_derived_state')
            f_get_sensor_data = kwargs.get('f_get_sensor_data')

            if f_get_core_state and f_get_derived_state and f_get_sensor_data:
                observation, reward, info = self.build_step_data(
                    f_get_core_state,
                    f_get_derived_state,
                    f_get_sensor_data,
                )
                self._last_observation = observation
                self._last_reward = reward
                self._last_info = info

        return False  # 不终止
