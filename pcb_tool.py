"""把 PCB 布局优化封装成「Agent 可以调用的工具」。

设计要点（也是这一层存在的理由）：
1. 对外只暴露 run_pcb_optimization()，返回值是纯 dict，
   可以直接 json.dumps 成工具消息交给大模型。
2. 允许调用方覆盖模拟退火的搜索参数（seed / alpha / iter_per_T / step），
   否则 Agent 只能「跑一次默认实验」，无法回答「调大迭代次数会怎样」。
3. 允许调用方指定输入文件，但只接受 data/ 目录下真实存在的 .txt，
   避免模型传入任意路径去读别处的文件。
4. 所有参数都会做范围校验，错误信息要能让模型看懂并自行修正。

直接运行本文件可以单独跑一次优化（不经过大模型）：

    python pcb_tool.py
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt

import config
import components_io
import pcb_board
import layout_cost
import sa_optimizer

# 输出文件名前缀：结果是 output/components_<prefix>_optimized.txt
DEFAULT_OUTPUT_PREFIX = "default"

# 参数合法范围，和 Agent 工具声明里的描述保持一致
SEED_RANGE = (0, 2 ** 31 - 1)
ALPHA_RANGE = (0.5, 0.99)
ITER_PER_T_RANGE = (10, 2000)
STEP_RANGE = (0.5, 50.0)


def _check_int(name, value, low, high):
    """整数参数校验：拒绝 bool（它是 int 的子类）和越界值。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("参数 {} 必须是整数，收到 {!r}".format(name, value))
    if not low <= value <= high:
        raise ValueError(
            "参数 {} 必须在 [{}, {}] 之间，收到 {}".format(name, low, high, value)
        )
    return value


def _check_float(name, value, low, high):
    """浮点参数校验：接受 int，但拒绝 bool 和越界值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("参数 {} 必须是数字，收到 {!r}".format(name, value))
    value = float(value)
    if not low <= value <= high:
        raise ValueError(
            "参数 {} 必须在 [{}, {}] 之间，收到 {}".format(name, low, high, value)
        )
    return value


def list_components_files():
    """列出 data/ 目录下可用的元器件输入文件（只暴露文件名）。"""
    return sorted(path.name for path in config.DATA_DIR.glob("*.txt"))


def _resolve_components_file(components_file):
    """把工具参数里的文件名解析成绝对路径，并做白名单校验。"""
    if components_file in (None, ""):
        return str(config.COMPONENTS_FILE)

    available = list_components_files()
    if components_file not in available:
        raise ValueError(
            "未知的元器件文件：{!r}，可用文件有：{}".format(
                components_file, "、".join(available) or "（无）"
            )
        )
    return str(config.DATA_DIR / components_file)


def _save_plots(board, before_components, after_components,
                before_result, after_result, history, output_dir, prefix):
    """保存优化前/优化后布局图和收敛曲线，返回 {名称: 路径}。"""
    images = {}

    before_path = output_dir / "pcb_layout_{}_before.png".format(prefix)
    fig, _ = sa_optimizer.draw_layout(
        board, before_components, before_result, "优化前布局",
        str(before_path),
    )
    plt.close(fig)
    images["before_layout"] = str(before_path)

    after_path = output_dir / "pcb_layout_{}_after.png".format(prefix)
    fig, _ = sa_optimizer.draw_layout(
        board, after_components, after_result, "优化后布局（模拟退火后）",
        str(after_path),
    )
    plt.close(fig)
    images["after_layout"] = str(after_path)

    history_path = output_dir / "pcb_layout_{}_history.png".format(prefix)
    fig, _ = sa_optimizer.draw_history(history, str(history_path))
    plt.close(fig)
    images["convergence"] = str(history_path)

    return images


def run_pcb_optimization(
        components_file=None,
        seed=None,
        alpha=None,
        iter_per_T=None,
        step=None,
        make_plots=False,
        output_prefix=DEFAULT_OUTPUT_PREFIX,
):
    """
    运行一次 PCB 布局模拟退火优化，返回结构化结果。

    参数
    ----
    components_file : 输入文件名（data/ 目录下的 .txt），默认 components.txt
    seed            : 随机种子，默认使用 sa_optimizer.SEED
    alpha           : 降温系数，越大降温越慢、搜索越充分
    iter_per_T      : 每个温度下的尝试次数
    step            : 邻域扰动的最大步长
    make_plots      : 是否额外保存布局图与收敛曲线（会多花一点时间）
    output_prefix   : 输出文件名前缀

    返回
    ----
    dict，包含优化前后的代价/越界数/重叠数、搜索耗时、结果文件路径等。
    """
    # ---------- 1. 解析输入与参数 ----------
    # 即使不经过 Agent（直接 import 调用），参数也要守住范围
    file_path = _resolve_components_file(components_file)
    seed = sa_optimizer.SEED if seed is None else _check_int("seed", seed, *SEED_RANGE)
    alpha = (
        sa_optimizer.ALPHA if alpha is None
        else _check_float("alpha", alpha, *ALPHA_RANGE)
    )
    iter_per_T = (
        sa_optimizer.ITER_PER_T if iter_per_T is None
        else _check_int("iter_per_T", iter_per_T, *ITER_PER_T_RANGE)
    )
    step = sa_optimizer.STEP if step is None else _check_float("step", step, *STEP_RANGE)

    # ---------- 2. 读取元器件和电路板 ----------
    components = components_io.load_components(file_path)
    board = pcb_board.create_board()

    # ---------- 3. 优化前指标 ----------
    before = layout_cost.calculate_layout_cost(
        components, board,
        sa_optimizer.LAMBDA_B, sa_optimizer.LAMBDA_O,
        verbose=False,
    )

    # ---------- 4. 运行模拟退火 ----------
    start_time = time.perf_counter()

    best, best_cost, history = sa_optimizer.simulated_annealing_layout(
        components,
        board,
        seed=seed,
        alpha=alpha,
        iter_per_T=iter_per_T,
        step=step,
        verbose=False,
    )

    elapsed = time.perf_counter() - start_time

    # ---------- 5. 优化后指标（重新算一遍，不信任搜索过程中的记录） ----------
    after = layout_cost.calculate_layout_cost(
        best, board,
        sa_optimizer.LAMBDA_B, sa_optimizer.LAMBDA_O,
        verbose=False,
    )

    # ---------- 6. 保存优化结果 ----------
    output_dir = config.ensure_output_dir()

    result_path = output_dir / "components_{}_optimized.txt".format(output_prefix)
    components_io.save_components(best, str(result_path))

    # ---------- 7. 可选：保存图片 ----------
    images = {}
    if make_plots:
        images = _save_plots(
            board, components, best, before, after, history,
            output_dir, output_prefix,
        )

    return {
        "components_file": Path(file_path).name,
        "parameters": {
            "seed": seed,
            "alpha": alpha,
            "iter_per_T": iter_per_T,
            "step": step,
        },
        "before_cost": before["cost"],
        "after_cost": after["cost"],
        "before_out_of_bounds": before["n_b"],
        "after_out_of_bounds": after["n_b"],
        "before_overlaps": before["n_o"],
        "after_overlaps": after["n_o"],
        "best_cost": best_cost,
        "temperature_stages": len(history) - 1,
        "candidate_attempts": (len(history) - 1) * iter_per_T,
        "elapsed_seconds": round(elapsed, 4),
        "result_file": str(result_path),
        "images": images,
    }


if __name__ == "__main__":
    result = run_pcb_optimization(make_plots=True)

    print("PCB 布局优化完成")
    for key, value in result.items():
        print("{}: {}".format(key, value))
