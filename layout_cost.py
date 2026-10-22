"""布局代价函数。

硬约束（越界、重叠）用惩罚项折算成一个标量代价：

    C = λ_b · N_b + λ_o · N_o

其中 N_b 是越界元器件数量，N_o 是重叠的元器件对数量。
模拟退火只认这个数字，所以代价函数是整个优化过程的目标函数。
"""

import components_io
import pcb_board
import layout_check
import config

# 默认惩罚系数：一个越界元器件、一对重叠元器件各折算 100 的代价
LAMBDA_B = 100
LAMBDA_O = 100


def calculate_layout_cost(components, board, lambda_b=LAMBDA_B, lambda_o=LAMBDA_O, verbose=True):
    """计算布局代价，返回包含明细的字典。

    N_b：越界元器件数量
    N_o：重叠元器件对数量
    """
    # 复用布局检查：越界名单 + 重叠对
    out_of_board_names, overlap_pairs = layout_check.check_layout(components, board)

    n_b = len(out_of_board_names)
    n_o = len(overlap_pairs)

    cost = lambda_b * n_b + lambda_o * n_o

    result = {
        "out_of_board_names": out_of_board_names,
        "overlap_pairs": overlap_pairs,
        "n_b": n_b,
        "n_o": n_o,
        "lambda_b": lambda_b,
        "lambda_o": lambda_o,
        "cost": cost,
    }

    if verbose:
        print_cost_report(components, board, result)

    return result


def print_cost_report(components, board, result):
    """打印布局代价统计结果。"""
    print("=" * 60)
    print("布局代价评估")
    print("电路板尺寸：{} × {}".format(board["width"], board["height"]))
    print("元器件数量：{}".format(len(components)))

    if result["out_of_board_names"]:
        print("越界元器件：{}".format(", ".join(result["out_of_board_names"])))
    else:
        print("越界元器件：无")

    if result["overlap_pairs"]:
        overlap_text = ", ".join(
            "{}-{}".format(a, b) for a, b in result["overlap_pairs"]
        )
        print("重叠元器件对：{}".format(overlap_text))
    else:
        print("重叠元器件对：无")

    print("越界数量 N_b = {}".format(result["n_b"]))
    print("重叠对数量 N_o = {}".format(result["n_o"]))
    print("惩罚系数：λ_b = {}, λ_o = {}".format(
        result["lambda_b"], result["lambda_o"]
    ))
    print("总代价 C = {} × {} + {} × {} = {}".format(
        result["lambda_b"],
        result["n_b"],
        result["lambda_o"],
        result["n_o"],
        result["cost"],
    ))
    print("=" * 60)


def print_components_table(components):
    """打印元器件坐标表。"""
    print("名称   x    y    w    h")
    for comp in components:
        print("{:<4} {:<4} {:<4} {:<4} {:<4}".format(
            comp["name"],
            comp["x"],
            comp["y"],
            comp["width"],
            comp["height"],
        ))


def get_base_components():
    """读取默认算例；文件缺失或格式错误时退回内置数据。"""
    try:
        return components_io.load_components(str(config.COMPONENTS_FILE))
    except (FileNotFoundError, ValueError) as error:
        print("读取 {} 失败：{}".format(config.COMPONENTS_FILE.name, error))
        print("改用内置默认数据。")
        return [
            {"name": "U1", "x": 10, "y": 20, "width": 8, "height": 6},
            {"name": "R1", "x": 30, "y": 30, "width": 5, "height": 5},
            {"name": "C1", "x": 32, "y": 32, "width": 5, "height": 5},
            {"name": "L1", "x": 90, "y": 70, "width": 15, "height": 15},
            {"name": "D1", "x": 60, "y": 20, "width": 10, "height": 10},
            {"name": "Q1", "x": 60, "y": 25, "width": 10, "height": 10},
            {"name": "J1", "x": -5, "y": 10, "width": 8, "height": 8},
            {"name": "K1", "x": 50, "y": 50, "width": 4, "height": 4},
            {"name": "M1", "x": 80, "y": 10, "width": 6, "height": 4},
        ]
