"""实验结果 CSV 的读取与统计。

`analyze_experiments` 这个工具就是围绕这两个函数做的。
统计口径说明：
- 成功：一次实验的最终代价为 0，即找到无越界、无重叠的布局；
- 成功率：成功次数 / 实验总次数；
- 中位数：实验次数为偶数时取中间两个数的平均值。
"""

import csv
from pathlib import Path


def read_csv(file_path):
    """读取 CSV 文件，返回由字典组成的列表（每行一个 dict）。"""
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def summarize_experiment(data, worst_n=0):
    """统计实验数据。

    参数
    ----
    data    : read_csv() 返回的行列表
    worst_n : 额外列出表现最差的 N 次实验（0 表示不列）

    返回的字典里除了总次数/成功率/平均成本/最低成本，
    还给出中位数、最差成本，以及（可选）最差实验明细，
    这样 Agent 回答「哪几个 seed 表现不好」时才有真实数据可依。
    """
    total_experiments = len(data)

    # 逐行取出最终成本，同时保留 seed，便于后面定位到具体实验
    rows = []
    for row in data:
        rows.append(
            {
                "seed": row.get("seed", ""),
                "final_cost": float(row["final_cost"]),
            }
        )
    costs = [item["final_cost"] for item in rows]

    success_count = sum(1 for cost in costs if cost == 0)

    if total_experiments == 0:
        success_rate = 0.0
        average_cost = 0.0
        best_cost = 0.0
        worst_cost = 0.0
        median_cost = 0.0
    else:
        success_rate = success_count / total_experiments * 100
        average_cost = sum(costs) / total_experiments
        best_cost = min(costs)
        worst_cost = max(costs)
        median_cost = _median(costs)

    summary = {
        "total_experiments": total_experiments,
        "success_count": success_count,
        "success_rate": success_rate,
        "average_cost": average_cost,
        "best_cost": best_cost,
        "worst_cost": worst_cost,
        "median_cost": median_cost,
    }

    # 可选：列出表现最差的 N 次实验
    if worst_n and rows:
        ranked = sorted(rows, key=lambda item: item["final_cost"], reverse=True)
        summary["worst_experiments"] = ranked[:worst_n]

    return summary


def _median(values):
    """求中位数（不依赖 statistics，逻辑更直观）。"""
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
