"""PCB 布局的模拟退火优化。

思路：把布局当作一个解，用「随机移动某个元器件」生成邻域候选解，
以布局代价 C = λ_b·N_b + λ_o·N_o 为目标，按 Metropolis 准则决定是否接受，
温度逐步降低，最终返回搜索到的最优布局。

三个容易被忽略但很关键的细节：
1. 候选解必须 deepcopy，否则修改候选解会顺手改掉当前解；
2. 全局最优解要单独维护，不能用「当前解」代替；
3. 已越界的元件，邻域移动要有向板内偏置的倾向（boundary_repair_shift），
   否则搜索会浪费大量迭代在明显不可能的方向上。
"""

import copy

import numpy as np
import matplotlib.pyplot as plt

import config
import components_io
import layout_check
import layout_cost
import pcb_board

# ---------------- 搜索超参数 ----------------
SEED = 30            # 随机种子，固定后实验可复现
INIT_T = 100.0       # 初始温度
MIN_T = 1.0          # 终止温度
ALPHA = 0.95         # 降温系数（每个温度阶段乘一次）
ITER_PER_T = 100     # 每个温度下尝试的候选解个数
STEP = 5.0           # 高温时的最大平移步长
MIN_STEP = 1.0       # 低温时的最小平移步长，避免后期完全走不动
INWARD_PROB = 0.85   # 元件越界时，向板内方向移动的概率

# 惩罚系数统一由 layout_cost 提供，避免两处各写一份
LAMBDA_B = layout_cost.LAMBDA_B
LAMBDA_O = layout_cost.LAMBDA_O


def load_components_data(file_path=None):
    """读取元器件；失败时退回内置默认数据，保证程序还能跑起来。"""
    if file_path is None:
        file_path = str(config.COMPONENTS_FILE)

    try:
        return components_io.load_components(file_path)
    except (FileNotFoundError, ValueError) as error:
        print("读取 {} 失败：{}".format(file_path, error))
        print("改用内置默认数据。")
        return layout_cost.get_base_components()


def evaluate_cost(components, board, lambda_b=LAMBDA_B, lambda_o=LAMBDA_O):
    """计算布局代价（只返回数值）。

    SA 会算成千上万次，所以这里强制 verbose=False，不打印任何东西。
    """
    result = layout_cost.calculate_layout_cost(
        components, board,
        lambda_b=lambda_b, lambda_o=lambda_o,
        verbose=False,
    )
    return result["cost"]


def boundary_repair_shift(comp, board, s, rng, inward_prob=INWARD_PROB):
    """为「当前已越界」的元件生成带向板内偏置的位移 (dx, dy)。

    规则：
      - 默认 dx、dy 仍然是 [-s, s] 的随机位移；
      - 对越界的轴，以 inward_prob 概率强制朝板内方向移动；
      - 剩下 (1 - inward_prob) 概率保留原随机方向，保持搜索随机性；
      - 未越界的轴不做任何干预。

    越界方向按 pcb_board.is_inside_board 的口径判断：
        矩形左下角 (x, y)，向右上延伸 width、height。
        x < 0                     -> 左侧越界，应该往 +x
        y < 0                     -> 下方越界，应该往 +y
        x + width  > board.width  -> 右侧越界，应该往 -x
        y + height > board.height -> 上方越界，应该往 -y
    """
    dx = rng.uniform(-s, s)
    dy = rng.uniform(-s, s)

    # X 轴修复倾向
    if comp["x"] < 0:
        if rng.random() < inward_prob:
            dx = rng.uniform(0, s)          # 向右
    elif comp["x"] + comp["width"] > board["width"]:
        if rng.random() < inward_prob:
            dx = -rng.uniform(0, s)         # 向左

    # Y 轴修复倾向
    if comp["y"] < 0:
        if rng.random() < inward_prob:
            dy = rng.uniform(0, s)          # 向上
    elif comp["y"] + comp["height"] > board["height"]:
        if rng.random() < inward_prob:
            dy = -rng.uniform(0, s)         # 向下

    return dx, dy


def random_neighbor(components, board, rng, step=STEP, T=None, init_T=INIT_T,
                    min_step=MIN_STEP, inward_prob=INWARD_PROB):
    """随机挑一个元器件，把它的左下角坐标平移一小段，返回新的布局。

    要点：
      - 必须 deepcopy！否则新解和当前解会指向同一批字典，
        修改候选解会悄悄改掉当前解，比较结果就失真了。
      - 坐标取整，保证写回文件后仍能被 components_io 的 int() 正确解析。
      - 不做硬边界裁剪，越界交给代价函数的惩罚项处理；
        只做一个很宽松的范围限制，避免元件「飞」得太远把图拉扁。
    """
    new_components = copy.deepcopy(components)

    index = int(rng.integers(0, len(new_components)))
    comp = new_components[index]

    # 步长随温度变化：高温时大步探索，低温时小步微调
    if T is not None and T > 0:
        s = max(min_step, step * (T / init_T))
    else:
        s = step

    if not pcb_board.is_inside_board(comp, board):
        dx, dy = boundary_repair_shift(comp, board, s, rng, inward_prob)
    else:
        dx = rng.uniform(-s, s)
        dy = rng.uniform(-s, s)

    comp["x"] = int(round(comp["x"] + dx))
    comp["y"] = int(round(comp["y"] + dy))

    # 宽松范围限制：最多离板框半个板宽/板高
    margin_x = 0.5 * board["width"]
    margin_y = 0.5 * board["height"]
    comp["x"] = int(np.clip(comp["x"], -margin_x, board["width"] + margin_x))
    comp["y"] = int(np.clip(comp["y"], -margin_y, board["height"] + margin_y))

    return new_components


def simulated_annealing_layout(
        components,
        board,
        init_T=INIT_T,
        min_T=MIN_T,
        alpha=ALPHA,
        iter_per_T=ITER_PER_T,
        step=STEP,
        seed=SEED,
        lambda_b=LAMBDA_B,
        lambda_o=LAMBDA_O,
        verbose=True,
        inward_prob=INWARD_PROB,
        min_step=MIN_STEP,
):
    """用模拟退火搜索代价更低的布局。

    参数
    ----
    init_T      : 初始温度
    min_T       : 终止温度（低于它就停止）
    alpha       : 降温系数，越大降温越慢、搜索越充分
    iter_per_T  : 每个温度下尝试的候选解个数
    step        : 高温时的最大平移步长
    min_step    : 低温时的最小平移步长（步长 = max(min_step, step * T / init_T)）
    seed        : 随机种子
    inward_prob : 元件已经越界时，向板内方向移动的概率
    verbose     : 是否逐温度阶段打印搜索过程

    返回
    ----
    best_components : 找到的最优布局（深拷贝）
    best_cost       : 最优布局的代价
    history         : 每个温度结束后的记录 [(T, best_cost, current_cost), ...]
    """
    rng = np.random.default_rng(seed)

    # 当前解 / 当前代价
    current = copy.deepcopy(components)
    current_cost = evaluate_cost(current, board, lambda_b, lambda_o)

    # 历史最优解（单独维护，很关键）
    best = copy.deepcopy(current)
    best_cost = current_cost

    history = [(init_T, best_cost, current_cost)]
    accepted_worse = 0  # 统计「接受更差解」的次数，帮助理解 SA
    total_trials = 0

    T = init_T
    while T > min_T:
        for _ in range(iter_per_T):
            # 1) 生成候选布局（邻域搜索）
            #    注意：这里必须全部用关键字传参。
            #    random_neighbor 的形参顺序是
            #      (components, board, rng, step, T, init_T, min_step, inward_prob)
            #    如果按位置传 min_step / inward_prob 很容易写反，
            #    而且不会报错，只会静默改变算法行为。
            candidate = random_neighbor(
                current,
                board,
                rng,
                step=step,
                T=T,
                init_T=init_T,
                min_step=min_step,
                inward_prob=inward_prob,
            )
            candidate_cost = evaluate_cost(candidate, board, lambda_b, lambda_o)

            delta = candidate_cost - current_cost
            total_trials += 1

            # 2) Metropolis 准则：更好一定接受，更差按概率接受
            if delta < 0:
                accept = True
            else:
                probability = np.exp(-delta / T)
                accept = rng.random() < probability
                if accept and delta > 0:
                    accepted_worse += 1

            # 3) 接受则更新当前解，并尝试刷新历史最优
            if accept:
                current = candidate
                current_cost = candidate_cost
                if current_cost < best_cost:
                    best = copy.deepcopy(current)
                    best_cost = current_cost

        # 4) 记录 + 降温
        history.append((T, best_cost, current_cost))
        if verbose:
            print("T = {:9.4f}   当前代价 = {:5.0f}   历史最优 = {:5.0f}".format(
                T, current_cost, best_cost))
        T *= alpha

    if verbose and total_trials > 0:
        print("共尝试 {} 次，其中接受更差解 {} 次（{:.2%}）".format(
            total_trials, accepted_worse, accepted_worse / total_trials))

    return best, best_cost, history


def draw_layout(board, components, result, title, image_path=None):
    """画出一张布局图（越界红、重叠橙、正常绿）。"""
    config.setup_matplotlib()
    fig, ax = layout_check.draw_board(
        board,
        components,
        out_of_board_names=result["out_of_board_names"],
        overlap_pairs=result["overlap_pairs"],
    )
    ax.set_title(title)
    if image_path:
        fig.savefig(image_path, dpi=150, bbox_inches="tight")
    return fig, ax


def draw_history(history, image_path=None):
    """画出代价随温度变化的收敛曲线。"""
    config.setup_matplotlib()

    temperatures = [item[0] for item in history]
    best_costs = [item[1] for item in history]
    current_costs = [item[2] for item in history]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(temperatures, best_costs, "o-", markersize=3, label="历史最优代价")
    ax.plot(temperatures, current_costs, "s--", markersize=3, alpha=0.6, label="当前代价")
    ax.set_xscale("log")
    ax.invert_xaxis()  # 温度从高到低，更符合退火过程
    ax.set_xlabel("温度 T（对数坐标，从高到低）")
    ax.set_ylabel("布局代价 C")
    ax.set_title("模拟退火收敛过程")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if image_path:
        fig.savefig(image_path, dpi=150, bbox_inches="tight")
    return fig, ax
