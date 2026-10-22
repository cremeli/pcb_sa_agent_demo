"""随机种子稳定性实验。

固定算法参数、只改随机种子，跑 10 次，把每次的结果写进
data/seed_experiment_results.csv —— 也就是 `analyze_experiments` 工具读的那份数据。

这样做的意义：模拟退火是随机算法，单次结果说明不了问题；
只有多个种子一起看，成功率、平均成本这些统计量才有意义。

用法：

    python run_seed_experiments.py
"""

import csv
import time

import config
import components_io
import layout_check
import layout_cost
import pcb_board
import sa_optimizer

SEEDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 实验参数（与工具默认值保持一致，改这里不影响 Agent）
INIT_T = 100.0
MIN_T = 1.0
ALPHA = 0.95
ITER_PER_T = 100
STEP = 5.0
LAMBDA_B = layout_cost.LAMBDA_B
LAMBDA_O = layout_cost.LAMBDA_O

# 用哪个算例做实验。
# components.txt        9 个元件，默认参数下几乎每个种子都能收敛到 0，
#                       成功率恒为 100%，统计结果没有区分度；
# components_dense.txt  10 个更大的元件（约占板面积 40%），默认参数下只有约
#                       4/10 的种子能找到零代价布局，统计才有信息量。
#                       把 ITER_PER_T 调大或把 ALPHA 调大会明显提升成功率，
#                       正好可以用来演示「调参」。
# 注意：不要拿优化后的输出文件当输入。
COMPONENTS_FILE = config.COMPONENTS_DENSE_FILE

# 结果直接写回数据目录，Agent 的分析工具读的就是这个文件
OUTPUT_CSV = config.EXPERIMENT_CSV


def run_one_experiment(seed):
    """运行一个随机种子的实验，返回一行统计结果。"""
    # 每次重新读取原始数据，确保初始布局一致
    components = components_io.load_components(str(COMPONENTS_FILE))
    board = pcb_board.create_board()

    start_time = time.perf_counter()

    best, best_cost, history = sa_optimizer.simulated_annealing_layout(
        components=components,
        board=board,
        init_T=INIT_T,
        min_T=MIN_T,
        alpha=ALPHA,
        iter_per_T=ITER_PER_T,
        step=STEP,
        lambda_b=LAMBDA_B,
        lambda_o=LAMBDA_O,
        seed=seed,
        verbose=False,  # 10 个种子 × 90 个温度阶段，逐行打印会淹没结果
    )

    elapsed = time.perf_counter() - start_time

    # history 每个温度阶段记录一次，初始记录不算搜索阶段
    stages = len(history) - 1
    attempts = stages * ITER_PER_T

    # history 中每项格式为 (温度, 历史最佳代价, 当前代价)
    zero_stages = [i for i, item in enumerate(history) if item[1] == 0]
    first_zero_stage = zero_stages[0] if zero_stages else "未找到"

    # 用独立路径重新算一遍代价，确认搜索过程记录的 best_cost 没出错
    check_result = layout_cost.calculate_layout_cost(
        best, board,
        lambda_b=LAMBDA_B,
        lambda_o=LAMBDA_O,
        verbose=False,
    )
    check_cost = check_result["cost"]

    if check_cost != best_cost:
        raise ValueError(
            f"SEED={seed}：best_cost={best_cost}，"
            f"重新计算代价={check_cost}，两者不一致！"
        )

    # 顺便记录最优布局的诊断信息，便于排查
    out_of_board_names, overlap_pairs = layout_check.check_layout(best, board)

    result = {
        "seed": seed,
        "components_file": COMPONENTS_FILE.name,
        "final_cost": best_cost,
        "first_zero_stage": first_zero_stage,
        "elapsed_seconds": round(elapsed, 4),
        "temperature_stages": stages,
        "candidate_attempts": attempts,
        "cost_check_passed": True,
        "out_of_board_count": len(out_of_board_names),
        "overlap_count": len(overlap_pairs),
    }

    print(
        f"SEED={seed:3d} | "
        f"最终代价={best_cost:5.0f} | "
        f"越界={len(out_of_board_names)} | "
        f"重叠对={len(overlap_pairs)} | "
        f"首次零代价阶段={first_zero_stage!s:>4} | "
        f"耗时={elapsed:.4f}s"
    )

    return result


def main():
    results = []

    print("========== 随机种子稳定性实验 ==========")
    print(f"输入文件 = {COMPONENTS_FILE.name}")
    print(f"初始温度 = {INIT_T}，终止温度 = {MIN_T}")
    print(f"降温系数 ALPHA = {ALPHA}")
    print(f"每温度迭代次数 ITER_PER_T = {ITER_PER_T}")
    print(f"移动步长 STEP = {STEP}")
    print(f"随机种子数量 = {len(SEEDS)}")
    print()

    for seed in SEEDS:
        results.append(run_one_experiment(seed))

    # 保存原始实验记录
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # 自动汇总统计
    costs = [item["final_cost"] for item in results]
    times = [item["elapsed_seconds"] for item in results]
    successes = sum(cost == 0 for cost in costs)

    print("\n========== 实验汇总 ==========")
    print(f"实验总次数：{len(results)}")
    print(f"零代价成功次数：{successes}")
    print(f"零代价成功率：{successes / len(results) * 100:.1f}%")
    print(f"最终代价平均值：{sum(costs) / len(costs):.2f}")
    print(f"最终代价最小值：{min(costs)}")
    print(f"最终代价最大值：{max(costs)}")
    print(f"平均搜索耗时：{sum(times) / len(times):.4f} 秒")
    print(f"\n实验明细已保存至：{OUTPUT_CSV}")

    return results


if __name__ == "__main__":
    main()
