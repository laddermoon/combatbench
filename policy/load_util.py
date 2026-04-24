"""Policy 加载工具。

支持从目录加载 Policy，自动检测 policy.py 中实现了 canonical
:class:`envs.framework.policy.Policy` ABC 的类。
"""

import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from envs.framework.policy import Policy


def load_policy(
    policy_spec: Union[str, Path],
    **extra_kwargs: Any,
) -> Policy:
    """
    加载 Policy

    Args:
        policy_spec: Policy 规范，支持以下格式：
            - 目录路径：如 "my_policy" 或 "/path/to/my_policy"
            - 模块路径：如 "my_policy.policy.MyPolicy"
            - 模块路径 + 参数：如 "my_policy.policy.MyPolicy?lr=0.01&epochs=100"
        **extra_kwargs: 额外 kwargs，会 merge 到 query-string 参数之上传给
            Policy 子类的构造器（覆盖同名 query-string 值）。

    Returns:
        Policy 实例（envs.framework.policy.Policy 子类）

    Raises:
        ValueError: Policy 规范无效
        ImportError: 无法导入 Policy 模块
        RuntimeError: Policy 中没有找到 Policy 实现

    Examples:
        >>> # 方式 1: 只指定目录（自动检测第一个 Policy 子类）
        >>> policy = load_policy("my_policy")

        >>> # 方式 2: 指定模块路径 + 类名
        >>> policy = load_policy("my_policy.policy:MyPolicy")

        >>> # 方式 3: 指定模块路径 + query-string 参数
        >>> policy = load_policy("my_policy.policy:MyPolicy?lr=0.01&epochs=100")

        >>> # 方式 4: 使用绝对路径
        >>> policy = load_policy("/path/to/my_policy")
    """
    policy_spec = str(policy_spec)

    # 解析规范字符串
    module_path, class_name, params = parse_policy_spec(policy_spec)
    policy_file: Optional[Path] = None

    # 如果没有指定模块路径，默认为 {policy_dir}/policy.py
    if module_path.endswith('.py'):
        policy_file = Path(module_path)
    elif '.' not in module_path and '/' not in module_path:
        # 假设是目录路径，尝试导入 policy.py
        policy_dir = Path(module_path)
        package_policy_dir = Path(__file__).parent / module_path
        if policy_dir.is_dir():
            policy_file = policy_dir / "policy.py"
            module_path = f"{module_path}.policy"
        elif package_policy_dir.is_dir():
            policy_file = package_policy_dir / "policy.py"
            module_path = f"{__package__}.{module_path}.policy"
        else:
            policy_file = package_policy_dir / "policy.py"
        if not policy_file.exists():
            raise ValueError(f"Policy directory not found: {policy_dir}")

    # 动态导入模块
    try:
        # 首先尝试作为普通模块导入
        if policy_file is not None and module_path.endswith('.py'):
            raise ImportError()
        policy_module = importlib.import_module(module_path)
    except ImportError:
        # 如果失败，尝试从文件路径加载
        if policy_file is None:
            policy_dir = Path(module_path.replace('.', os.sep))
            if policy_dir.is_file() and policy_dir.suffix == '.py':
                policy_file = policy_dir
            else:
                policy_file = policy_dir / "policy.py"
        if not policy_file.exists():
            raise ImportError(f"Failed to import policy module '{module_path}' and policy.py not found at {policy_file}")

        # 使用 importlib.util 从文件路径加载模块
        spec = importlib.util.spec_from_file_location(policy_file.stem, policy_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module from {policy_file}")

        policy_module = importlib.util.module_from_spec(spec)
        sys.modules[policy_file.stem] = policy_module
        spec.loader.exec_module(policy_module)

    # 查找 Policy 子类
    policy_class = find_policy_class(policy_module, class_name)

    # 解析参数 (query string) + 调用方传入的额外 kwargs
    kwargs = parse_params(params) if params else {}
    kwargs.update(extra_kwargs)

    # 实例化 Policy。Policy ABC 故意不定义 __init__，由子类自行决定
    # 构造签名——建议子类接受 **kwargs 以快速忽略未知参数。
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
        module_path = module_part
    else:
        # 没有指定类名，稍后自动检测
        module_path = module_part
        class_name = None

    return module_path, class_name, params


def find_policy_class(
    module,
    class_name: Optional[str] = None,
) -> Type[Policy]:
    """在模块中查找 envs.framework.policy.Policy 的子类。

    Args:
        module: Python 模块
        class_name: 指定的类名（可选）

    Returns:
        Policy 类

    Raises:
        RuntimeError: 找不到 Policy 实现
    """
    # 如果指定了类名，直接使用
    if class_name:
        if not hasattr(module, class_name):
            raise RuntimeError(
                f"Class '{class_name}' not found in module {module.__name__}"
            )

        policy_class = getattr(module, class_name)

        if not (inspect.isclass(policy_class) and issubclass(policy_class, Policy)):
            raise RuntimeError(
                f"Class '{class_name}' does not inherit from "
                f"envs.framework.policy.Policy"
            )

        return policy_class

    # 自动查找第一个 Policy 子类
    policy_classes = []

    for name in dir(module):
        if name.startswith('_'):
            continue

        obj = getattr(module, name)

        if (inspect.isclass(obj) and
            issubclass(obj, Policy) and
            obj is not Policy):
            policy_classes.append((name, obj))

    if not policy_classes:
        raise RuntimeError(
            f"No Policy implementation found in module {module.__name__}. "
            f"Make sure policy.py contains a class that inherits from "
            f"envs.framework.policy.Policy."
        )

    # 返回第一个找到的类
    _name, policy_class = policy_classes[0]
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
    **extra_kwargs: Any,
) -> Policy:
    """从目录加载 Policy（load_policy 的别名）。

    Args:
        policy_dir: Policy 目录路径
        **extra_kwargs: 额外 kwargs，会 merge 到 query-string 参数之上传给
            构造器。

    Returns:
        Policy 实例
    """
    return load_policy(policy_dir, **extra_kwargs)
