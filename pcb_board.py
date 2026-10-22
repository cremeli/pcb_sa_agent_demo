"""电路板模型与越界判定。

坐标约定（全项目统一）：
    元器件用左下角坐标 (x, y) 加宽高 width、height 描述，
    向右为 +x，向上为 +y，板框左下角是 (0, 0)。
"""

# 板框尺寸
BOARD_WIDTH = 100
BOARD_HEIGHT = 80


def create_board():
    """返回电路板尺寸。"""
    return {
        "width": BOARD_WIDTH,
        "height": BOARD_HEIGHT,
    }


def is_inside_board(component, board):
    """判断单个元器件是否完全落在板框内（贴边算在板内）。"""
    return (
        component["x"] >= 0
        and component["y"] >= 0
        and component["x"] + component["width"] <= board["width"]
        and component["y"] + component["height"] <= board["height"]
    )
