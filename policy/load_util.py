"""
Policy 加载工具

支持从目录加载 Policy，自动检测 policy.py 中的 BaseCombatPolicy 实现。
"""

import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from .base import BaseCombatPolicy


def load_policy(
    policy_spec: Union[str, Path],
    observation_space: Any = None,
    action_space: Any = None
) -> BaseCombatPolicy:
    """
    加载 Policy

    Args:
        policy_spec: Policy 规范，支持以下格式：
            - 目录路径：如 "my_policy" 或 "/path/to/my_policy"
            - 模块路径：如 "my_policy.policy.MyPolicy"
            - 模块路径 + 参数：如 "my_policy.policy.MyPolicy?lr=0.01&epochs=100"
        observation_space: 观测空间（可选，传递给 Policy）
        action_space: 动作空间（可选，传递给 Policy）

    Returns:
        BaseCombatPolicy 实例

    Raises:
        ValueError: Policy 规范无效
        ImportError: 无法导入 Policy 模块
        RuntimeError: Policy 中没有找到 BaseCombatPolicy 实现

    Examples:
        >>> # 方式 1: 只指定目录（自动检测第一个 BaseCombatPolicy）
        >>> policy = load_policy("my_policy")

        >>> # 方式 2: 指定模块路径
        >>> policy = load_policy("my_policy.policy.MyCombatPolicy")

        >>> # 方式 3: 指定模块路径 + 参数
        >>> policy = load_policy("my_policy.policy.MyCombatPolicy?lr=0.01&epochs=100")

        >>> # 方式 4: 使用绝对路径
        >>> policy = load_policy("/path/to/my_policy")
    """
    policy_spec = str(policy_spec)

    # 解析规范字符串
    module_path, class_name, params = parse_policy_spec(policy_spec)

    # 如果没有指定模块路径，默认为 {policy_dir}/policy.py
    if '.' not in module_path:
        # 假设是目录路径，尝试导入 policy.py
        policy_dir = Path(module_path)
        if not policy_dir.is_dir():
            # 尝试作为相对路径
            policy_dir = Path(__file__).parent / module_path

        policy_file = policy_dir / "policy.py"
        if not policy_file.exists():
            raise ValueError(f"Policy directory not found: {policy_dir}")

        # 构建模块路径
        # 将目录路径转换为 Python 模块路径
        # 例如: "my_policy" -> "my_policy.policy"
        module_path = f"{module_path}.policy"

    # 动态导入模块
    try:
        policy_module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import policy module '{module_path}': {e}")

    # 查找 BaseCombatPolicy 实现
    policy_class = find_policy_class(policy_module, class_name)

    # 解析参数
    kwargs = parse_params(params) if params else {}

    # 添加观测/动作空间
    if observation_space is not None:
        kwargs['observation_space'] = observation_space
    if action_space is not None:
        kwargs['action_space'] = action_space

    # 实例化 Policy
    try:
        policy_instance = policy_class(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate policy {policy_class.__name__}: {e}")

    return policy_instance


def parse_policy_spec(spec: str) -> tuple:
    """
    解析 Policy 规范字符串

    Args:
        spec: Policy 规范字符串

    Returns:
        (module_path, class_name, params_str) 三元组
        - module_path: 模块路径（不含类名）
        - class_name: 类名（如果有）
        - params_str: 参数字符串（如果有）
    """
    # 分离查询参数
    if '?' in spec:
        module_part, params = spec.split('?', 1)
    else:
        module_part = spec
        params = None

    # 分离类名
    if ':' in module_part:
        # 支持两种格式：
        # "path/to/file.py:ClassName" (Python 文件)
        # "module.path:ClassName" (模块路径)
        module_part, class_name = module_part.rsplit(':', 1)

        # 如果是 .py 文件，需要特殊处理
        if module_part.endswith('.py'):
            # 从文件路径推导模块路径
            # 这里简化处理：如果是 .py 文件，直接使用文件所在目录
            module_path = module_part.replace('/', '.').replace('.py', '')
        else:
            module_path = module_part
    else:
        # 没有指定类名，稍后自动检测
        module_path = module_part
        class_name = None

    return module_path, class_name, params


def find_policy_class(
    module,
    class_name: Optional[str] = None
) -> Type[BaseCombatPolicy]:
    """
    在模块中查找 BaseCombatPolicy 实现类

    Args:
        module: Python 模块
        class_name: 指定的类名（可选）

    Returns:
        BaseCombatPolicy 类

    Raises:
        RuntimeError: 找不到 BaseCombatPolicy 实现
    """
    # 如果指定了类名，直接使用
    if class_name:
        if not hasattr(module, class_name):
            raise RuntimeError(
                f"Class '{class_name}' not found in module {module.__name__}"
            )

        policy_class = getattr(module, class_name)

        if not issubclass(policy_class, BaseCombatPolicy):
            raise RuntimeError(
                f"Class '{class_name}' does not inherit from BaseCombatPolicy"
            )

        return policy_class

    # 自动查找第一个 BaseCombatPolicy 子类
    policy_classes = []

    for name in dir(module):
        if name.startswith('_'):
            continue

        obj = getattr(module, name)

        # 检查是否是类且是 BaseCombatPolicy 的子类
        if (inspect.isclass(obj) and
            issubclass(obj, BaseCombatPolicy) and
            obj is not BaseCombatPolicy):
            policy_classes.append((name, obj))

    if not policy_classes:
        raise RuntimeError(
            f"No BaseCombatPolicy implementation found in module {module.__name__}. "
            f"Make sure policy.py contains a class that inherits BaseCombatPolicy."
        )

    # 返回第一个找到的类
    class_name, policy_class = policy_classes[0]
    return policy_class


def parse_params(params_str: str) -> Dict[str, Any]:
    """
    解析查询参数字符串

    支持类型：
    - 数字: scale=0.5, count=10
    - 布尔: enabled=true
    - 字符串: model_path=model.zip
    - JSON 列表: list=[1,2,3]
    - JSON 对象: config={"key":"value"}

    Args:
        params_str: 查询参数字符串（如 "scale=0.2&seed=42"）

    Returns:
        参数字典
    """
    if not params_str:
        return {}

    from urllib.parse import parse_qs
    import json

    params = {}

    for key, values in parse_qs(params_str).items():
        value = values[0]  # 取第一个值

        # 尝试解析为 JSON
        if value.startswith('{') or value.startswith('['):
            try:
                params[key] = json.loads(value)
            except json.JSONDecodeError:
                params[key] = value
        elif value.lower() == 'true':
            params[key] = True
        elif value.lower() == 'false':
            params[key] = False
        else:
            # 尝试解析为数字
            try:
                if '.' in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                params[key] = value

    return params


# 便捷函数

def load_policy_from_dir(
    policy_dir: Union[str, Path],
    observation_space: Any = None,
    action_space: Any = None
) -> BaseCombatPolicy:
    """
    从目录加载 Policy（便捷函数）

    这是 load_policy() 的简化版本，专门用于从目录加载。

    Args:
        policy_dir: Policy 目录路径
        observation_space: 观测空间（可选）
        action_space: 动作空间（可选）

    Returns:
        BaseCombatPolicy 实例

    Examples:
        >>> policy = load_policy_from_dir("my_policy")
        >>> policy = load_policy_from_dir("/path/to/my_policy")
    """
    return load_policy(policy_dir, observation_space, action_space)
