"""布局检查与绘图。

- `rectangles_overlap` / `find_overlaps`：判断元器件之间是否重叠；
- `check_layout`：一次性给出越界名单和重叠对；
- `draw_board`：把板框和元器件画出来，越界标红、重叠标橙、正常标绿。
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

import pcb_board


def rectangles_overlap(a, b):
    """判断两个矩形是否有面积重叠（边贴边不算重叠）。"""
    # 先判断它们是否完全分离，再取反
    a_left_of_b = a["x"] + a["width"] <= b["x"]
    b_left_of_a = b["x"] + b["width"] <= a["x"]
    a_below_b = a["y"] + a["height"] <= b["y"]
    b_below_a = b["y"] + b["height"] <= a["y"]
    separated = a_left_of_b or b_left_of_a or a_below_b or b_below_a
    return not separated


def find_overlaps(components):
    """两两检查，返回发生重叠的元器件名称对，例如 [("U1", "R1")]。"""
    overlaps = []
    n = len(components)

    for i in range(n):
        for j in range(i + 1, n):
            a = components[i]
            b = components[j]
            if rectangles_overlap(a, b):
                overlaps.append((a["name"], b["name"]))
    return overlaps


def check_layout(components, board):
    """同时检查越界和重叠。

    返回：
        out_of_board_names: 越界元器件名称列表
        overlap_pairs:      重叠元器件名称对列表
    """
    out_of_board_names = []

    for comp in components:
        if not pcb_board.is_inside_board(comp, board):
            out_of_board_names.append(comp["name"])

    overlap_pairs = find_overlaps(components)

    return out_of_board_names, overlap_pairs


def draw_board(board, components, out_of_board_names=None, overlap_pairs=None):
    """画电路板和元器件。

    颜色规则：
        越界：红色
        重叠但未越界：橙色
        正常：绿色

    注意：越界优先级高于重叠。
    """
    out_of_board_names = set(out_of_board_names or [])

    overlap_names = set()
    for a_name, b_name in (overlap_pairs or []):
        overlap_names.add(a_name)
        overlap_names.add(b_name)

    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)

    # 画电路板
    board_rect = Rectangle(
        (0, 0),
        board["width"],
        board["height"],
        linewidth=2.5,
        edgecolor="black",
        facecolor="#f2f2f2",
        alpha=0.8,
        zorder=1
    )
    ax.add_patch(board_rect)

    # 画元器件
    for comp in components:
        name = comp["name"]

        if name in out_of_board_names:
            # 越界：红色，优先级最高
            edge_color = "#C62828"
            face_color = "#EF9A9A"
        elif name in overlap_names:
            # 重叠：橙色
            edge_color = "#EF6C00"
            face_color = "#FFCC80"
        else:
            # 正常：绿色
            edge_color = "#2E7D32"
            face_color = "#A5D6A7"

        rect = Rectangle(
            (comp["x"], comp["y"]),
            comp["width"],
            comp["height"],
            linewidth=1.5,
            edgecolor=edge_color,
            facecolor=face_color,
            alpha=0.9,
            zorder=2
        )
        ax.add_patch(rect)

        # 名称标注在矩形中心
        center_x = comp["x"] + comp["width"] / 2
        center_y = comp["y"] + comp["height"] / 2
        ax.text(
            center_x,
            center_y,
            name,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
            zorder=3
        )

    # 自动设置坐标范围，留一点边距
    xs = [0, board["width"]]
    ys = [0, board["height"]]

    for comp in components:
        xs.append(comp["x"])
        xs.append(comp["x"] + comp["width"])
        ys.append(comp["y"])
        ys.append(comp["y"] + comp["height"])

    margin = 5
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("PCB Board and Components")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # 图例
    legend_elements = [
        Patch(facecolor="#A5D6A7", edgecolor="#2E7D32", label="在板内"),
        Patch(facecolor="#EF9A9A", edgecolor="#C62828", label="越界"),
        Patch(facecolor="#FFCC80", edgecolor="#EF6C00", label="重叠"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    return fig, ax
