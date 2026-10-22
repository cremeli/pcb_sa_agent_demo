"""PCB 布局优化 Agent 的命令行入口。

一次请求的流程：
1. 把 system prompt、用户请求、工具声明发给模型；
2. 模型若返回 tool_calls，就逐个执行，并把结果作为 role="tool" 消息回填；
3. 回到第 1 步继续让模型判断——它可能还想再调一个工具；
4. 直到模型不再请求工具，才把它的文本答案返回。

用法：

    python agent.py                                  # 交互模式，默认 DeepSeek
    python agent.py --once "帮我优化一下 PCB 布局"     # 只问一次
    python agent.py --provider ollama                # 用本地 Ollama

模型服务的配置见 config.py 和 README。
"""

import argparse
import json
import os

from openai import OpenAI

import config
import tools

SYSTEM_PROMPT = (
    "你是一个用于 PCB 布局实验的 agent。\n"
    "你可以使用以下工具：\n"
    "1. analyze_experiments：分析已有的实验结果 CSV，"
    "统计总次数、成功次数、成功率、平均/最低/最高成本，"
    "并可用 worst_n 列出表现最差的若干次实验。\n"
    "2. run_pcb_optimization：运行一次 PCB 布局模拟退火优化，"
    "可以指定输入文件、随机种子 seed、降温系数 alpha、"
    "每个温度下的迭代次数 iter_per_T、邻域步长 step，"
    "并可以选择是否生成布局图和收敛曲线。\n"
    "\n"
    "规则：\n"
    "- 用户要求分析实验数据时，调用 analyze_experiments。\n"
    "- 用户要求运行或优化 PCB 布局时，调用 run_pcb_optimization。\n"
    "- 用户想比较不同参数（例如不同 seed、不同降温速度）时，"
    "可以连续多次调用 run_pcb_optimization，每次换一个参数。\n"
    "- 如果用户只是在询问概念或原理，且不需要实际运行工具，直接回答即可，"
    "但不要假装执行过工具。\n"
    "- 工具执行后必须依据工具返回的真实结果回答，不得编造数据或结果。\n"
    "- 工具返回的路径（结果文件、图片）可以原样告诉用户。\n"
    "- 用中文回答，简洁但有信息量。"
)

# 惰性创建的客户端缓存；换了 provider / model 会重建
_client = None
_client_signature = None


def reset_client():
    """清掉缓存的客户端（切换配置或写测试时用）。"""
    global _client, _client_signature
    _client = None
    _client_signature = None


def get_client(provider=None):
    """创建（或复用）OpenAI 兼容客户端。

    只有真正要发请求时才会读取 API key，
    所以没有 key 的机器也能 import 本模块、跑单元测试。
    """
    global _client, _client_signature

    name, settings = config.resolve_provider(provider)
    signature = (name, settings["base_url"], settings["model"])

    if _client is None or _client_signature != signature:
        config.ensure_no_proxy(settings["no_proxy_host"])
        api_key = config.read_api_key(settings)
        _client = OpenAI(
            api_key=api_key,
            base_url=settings["base_url"],
            timeout=config.REQUEST_TIMEOUT,
        )
        _client_signature = signature

    return _client


def _execute_tool_call(tool_call):
    """执行一个 tool_call，返回要回填给模型的 role="tool" 消息。"""
    tool_name = tool_call.function.name

    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
        result = tools.execute_tool(tool_name, arguments)
        print("  -> 已执行 {}".format(tool_name))

    except Exception as error:
        print("  -> 工具 {} 执行失败：{}".format(tool_name, error))
        result = json.dumps(
            {"success": False, "error": str(error)},
            ensure_ascii=False,
        )

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result,
    }


def run_agent(user_request, history=None, client=None, max_rounds=None,
              provider=None):
    """处理一次用户请求，返回模型最终的自然语言回答。

    参数
    ----
    user_request : 用户的自然语言请求
    history      : 可选的对话历史（会被原地追加），
                   在同一个列表上反复调用即可实现多轮对话记忆
    client       : 可注入的客户端（测试用），默认走 get_client()
    max_rounds   : 最多允许几轮工具调用，默认 config.MAX_TOOL_ROUNDS
    provider     : 指定模型服务，默认按环境变量/配置文件
    """
    if client is None:
        client = get_client(provider)

    if max_rounds is None:
        max_rounds = config.MAX_TOOL_ROUNDS

    # 模型名在本次请求内固定，避免每轮都重新解析环境变量
    _, settings = config.resolve_provider(provider)
    model = settings["model"]

    if history is None:
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
    history.append({"role": "user", "content": user_request})

    for round_index in range(1, max_rounds + 1):
        # 每一轮都把当前完整对话发给模型，让它决定：继续调工具，还是给答案
        response = client.chat.completions.create(
            model=model,
            messages=history,
            tools=tools.TOOLS,
            tool_choice="auto",
            max_tokens=config.MAX_TOKENS,
            stream=False,
            timeout=config.REQUEST_TIMEOUT,
        )

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        # 模型不再请求工具 -> 这就是最终回答
        if not tool_calls:
            return message.content or "模型没有返回文本。"

        # 把模型的工具调用请求加入对话历史
        history.append(message.model_dump(exclude_none=True))
        print("[第 {} 轮] 模型请求调用 {} 个工具".format(
            round_index, len(tool_calls)))

        # 执行模型请求的工具，并逐个回填结果
        for tool_call in tool_calls:
            history.append(_execute_tool_call(tool_call))

    # 达到轮数上限仍未收敛：这一轮不再提供工具，逼模型基于已有结果作答
    print("[提醒] 已达到 {} 轮工具调用上限，强制收尾。".format(max_rounds))
    final_response = client.chat.completions.create(
        model=model,
        messages=history,
        max_tokens=config.MAX_TOKENS,
        stream=False,
        timeout=config.REQUEST_TIMEOUT,
    )
    return final_response.choices[0].message.content or "模型没有返回文本。"


def build_parser():
    parser = argparse.ArgumentParser(
        description="PCB 布局优化 Agent：用自然语言调用模拟退火布局优化工具",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(config.PROVIDER_DEFAULTS),
        help="模型服务，默认读环境变量 PCB_AGENT_PROVIDER，再退回 deepseek",
    )
    parser.add_argument(
        "--model",
        help="模型名，覆盖默认值（等价于设置 PCB_AGENT_MODEL）",
    )
    parser.add_argument(
        "--once",
        metavar="REQUEST",
        help="只处理这一条请求然后退出（适合脚本调用）",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.model:
        os.environ["PCB_AGENT_MODEL"] = args.model

    try:
        client = get_client(args.provider)
    except (RuntimeError, ValueError) as error:
        print(error)
        return 1

    provider_name, settings = config.resolve_provider(args.provider)

    if args.once:
        try:
            answer = run_agent(
                args.once, client=client, provider=args.provider
            )
        except Exception as error:
            print("程序运行失败：{}".format(type(error).__name__))
            print("错误详情：{}".format(error))
            return 1
        print(answer)
        return 0

    print("PCB 布局优化 Agent 已启动（provider={}, model={}）".format(
        provider_name, settings["model"]))
    print("输入 exit 或直接回车退出。")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            request = input("\n请输入任务：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not request or request.lower() in ("exit", "quit", "q"):
            break

        try:
            answer = run_agent(
                request,
                history=history,
                client=client,
                provider=args.provider,
            )
        except Exception as error:
            print("程序运行失败：{}".format(type(error).__name__))
            print("错误详情：{}".format(error))
            # 本次请求没有形成完整问答，把 user 消息撤回，避免污染历史
            if history and history[-1].get("role") == "user":
                history.pop()
            continue

        print("\nAgent 最终回答：")
        print(answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
