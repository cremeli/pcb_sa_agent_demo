"""元器件文件的读写。

数据格式：每行一个元器件，逗号分隔的 5 个字段

    name,x,y,width,height

例如 `U1,10,20,8,6`，表示左下角在 (10, 20)、宽 8、高 6 的元器件 U1。

读取时会做三层校验，并且错误信息带上行号，便于快速定位问题：
- 文件不存在 -> FileNotFoundError
- 字段数量不等于 5 -> ValueError
- 数字字段转换失败 -> ValueError
"""


def save_components(components, file_path):
    """把元器件列表保存成文本文件，每个元器件一行。"""
    with open(file_path, "w", encoding="utf-8") as file:
        for comp in components:
            line = "{},{},{},{},{}".format(
                comp["name"],
                comp["x"],
                comp["y"],
                comp["width"],
                comp["height"],
            )
            file.write(line + "\n")


def load_components(file_path):
    """从文本文件读取元器件，返回字典列表。

    出错时不会返回不完整的数据：直接抛异常。
    注意文件里的数字读出来是字符串，必须用 int() 转换。
    """
    # ---------- 1. 打开文件 ----------
    try:
        file = open(file_path, "r", encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError("文件不存在 -> {}".format(file_path))

    components = []
    with file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            # 跳过空行
            if not line:
                continue

            fields = line.split(",")

            # ---------- 2. 字段数量检查 ----------
            if len(fields) != 5:
                raise ValueError(
                    "第 {} 行字段数量不正确：需要 5 个字段 "
                    "(name,x,y,width,height)，实际 {} 个 -> {!r}".format(
                        line_no, len(fields), line
                    )
                )

            # ---------- 3. 数字转换（可能触发 ValueError） ----------
            try:
                comp = {
                    "name": fields[0],
                    "x": int(fields[1]),
                    "y": int(fields[2]),
                    "width": int(fields[3]),
                    "height": int(fields[4]),
                }
            except ValueError as error:
                raise ValueError(
                    "第 {} 行数字格式错误 -> {!r}（原因：{}）".format(
                        line_no, line, error
                    )
                )

            components.append(comp)

    return components
