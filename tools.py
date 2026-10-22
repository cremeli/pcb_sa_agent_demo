"""工具的声明、参数校验与执行。

这里有两套「同一份规则」：

1. `TOOLS`：给大模型看的 JSON Schema，决定它「打算怎么调」；
2. `TOOL_ARGUMENT_RULES`：执行前的最后一道关，保证「真的能执行」。

两者分开写是刻意的：Schema 只是提示，模型完全可能给出越界值或错的类型，
所以真正执行前必须自己再校验一次，并把错误信息回填给模型让它重试。
"""

import json

import config
import pcb_tool

# 这些名字导入到本模块，方便测试时用 patch 替换
from experiment_analysis import read_csv, summarize_experiment
from pcb_tool import run_pcb_optimization


def _components_choices():
    """当前可用的元器件输入文件（动态获取，新增文件后无需改代码）。"""
    return pcb_tool.list_components_files()


# ---------------------------------------------------------------- 工具声明
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_experiments",
            "description": (
                "分析已有的 PCB 布局实验 CSV 数据。"
                "统计实验总次数、成功次数、成功率、"
                "平均最终成本、最低/最高/中位数成本。"
                "当用户询问实验表现、实验统计或实验总结时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "worst_n": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 50,
                        "description": (
                            "额外列出表现最差的 N 次实验（seed 与最终成本）。"
                            "用户想知道哪些实验不好时填 3~10，默认 0。"
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_pcb_optimization",
            "description": (
                "运行一次 PCB 布局模拟退火优化，返回优化前后的成本、"
                "越界元件数、重叠对数、运行耗时、结果文件路径。"
                "当用户要求运行、优化或改善 PCB 布局时调用。"
                "可选参数用于改变搜索行为或换一个更难的算例。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "components_file": {
                        "type": "string",
                        "enum": _components_choices(),
                        "description": (
                            "输入元器件文件名，默认 components.txt。"
                            "components_dense.txt 是元件更大、更拥挤的算例；"
                            "默认算例很快就会收敛到 0 代价，"
                            "想展示真实优化效果时用它。"
                        ),
                    },
                    "seed": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2147483647,
                        "description": "随机种子，同一个种子结果可复现。",
                    },
                    "alpha": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 0.99,
                        "description": "降温系数，越大降温越慢、搜索越充分，默认 0.95。",
                    },
                    "iter_per_T": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 2000,
                        "description": "每个温度下的尝试次数，默认 100。",
                    },
                    "step": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 50.0,
                        "description": "邻域扰动的最大步长，默认 5.0。",
                    },
                    "make_plots": {
                        "type": "boolean",
                        "description": (
                            "是否额外保存布局图和收敛曲线（多花约 1 秒）。"
                            "用户想看效果时填 true。"
                        ),
                    },
                },
                "required": [],
            },
        },
    },
]

# ---------------------------------------------------------------- 参数校验
TOOL_ARGUMENT_RULES = {
    "analyze_experiments": {
        "worst_n": {"type": int, "min": 0, "max": 50},
    },
    "run_pcb_optimization": {
        "components_file": {"type": str, "choices_provider": _components_choices},
        "seed": {"type": int, "min": 0, "max": 2 ** 31 - 1},
        "alpha": {"type": float, "min": 0.5, "max": 0.99},
        "iter_per_T": {"type": int, "min": 10, "max": 2000},
        "step": {"type": float, "min": 0.5, "max": 50.0},
        "make_plots": {"type": bool},
    },
}


def _check_argument(tool_name, name, value, rule):
    """按规则校验单个参数，返回转换后的值；不合法就抛 ValueError。"""
    expected = rule["type"]

    if expected is bool:
        if not isinstance(value, bool):
            raise ValueError(
                "工具 {} 的参数 {} 必须是布尔值，收到 {!r}".format(
                    tool_name, name, value
                )
            )
        return value

    if expected is int:
        # bool 是 int 的子类，要单独排除，否则 True 会被当成 1
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "工具 {} 的参数 {} 必须是整数，收到 {!r}".format(
                    tool_name, name, value
                )
            )
    elif expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "工具 {} 的参数 {} 必须是数字，收到 {!r}".format(
                    tool_name, name, value
                )
            )
        value = float(value)
    elif expected is str:
        if not isinstance(value, str):
            raise ValueError(
                "工具 {} 的参数 {} 必须是字符串，收到 {!r}".format(
                    tool_name, name, value
                )
            )
        provider = rule.get("choices_provider")
        if provider is not None:
            choices = provider()
            if value not in choices:
                raise ValueError(
                    "工具 {} 的参数 {} 只能取 {} 之一，收到 {!r}".format(
                        tool_name, name, "、".join(choices), value
                    )
                )
        return value

    minimum = rule.get("min")
    maximum = rule.get("max")
    if minimum is not None and value < minimum:
        raise ValueError(
            "工具 {} 的参数 {} 不能小于 {}，收到 {}".format(
                tool_name, name, minimum, value
            )
        )
    if maximum is not None and value > maximum:
        raise ValueError(
            "工具 {} 的参数 {} 不能大于 {}，收到 {}".format(
                tool_name, name, maximum, value
            )
        )
    return value


def validate_arguments(tool_name, arguments):
    """校验工具名与参数，返回清理后的参数字典。"""
    if tool_name not in TOOL_ARGUMENT_RULES:
        raise ValueError("不支持的工具：{}".format(tool_name))

    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ValueError(
            "工具参数必须是 JSON 对象，收到 {!r}".format(type(arguments).__name__)
        )

    rules = TOOL_ARGUMENT_RULES[tool_name]
    unknown = sorted(set(arguments) - set(rules))
    if unknown:
        raise ValueError(
            "工具 {} 不支持参数：{}（可用参数：{}）".format(
                tool_name,
                "、".join(unknown),
                "、".join(rules) or "无",
            )
        )

    cleaned = {}
    for name, value in arguments.items():
        cleaned[name] = _check_argument(tool_name, name, value, rules[name])
    return cleaned


# ---------------------------------------------------------------- 执行工具
def _analyze_experiments(worst_n=0):
    """真正干活的地方：读 CSV -> 统计。"""
    data = read_csv(config.EXPERIMENT_CSV)
    return summarize_experiment(data, worst_n=worst_n)


def execute_tool(tool_name, arguments):
    """执行一个工具调用，返回可 JSON 序列化的字符串。"""
    cleaned = validate_arguments(tool_name, arguments)

    if tool_name == "analyze_experiments":
        result = _analyze_experiments(**cleaned)
    else:
        # 只把模型真正给出的参数传下去，未给出的交给 pcb_tool 的默认值
        result = run_pcb_optimization(**cleaned)

    return json.dumps(result, ensure_ascii=False)
