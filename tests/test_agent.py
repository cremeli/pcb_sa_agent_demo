"""Agent 工具层、工具调用循环与 PCB 工具的自动化测试。

全部不联网：
- 涉及大模型的部分用假客户端（_FakeClient）模拟；
- 涉及真实算法/文件的部分直接跑本地代码。

所以运行测试不需要 API key。

运行方式（在项目根目录）：

    python -m unittest discover -s tests -t . -v
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent
import config
import experiment_analysis
import pcb_tool
import tools


# ---------------------------------------------------------------- 假客户端
class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    """模拟 openai 的 ChatCompletionMessage。"""

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=False):
        data = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in (self.tool_calls or [])
            ],
        }
        if exclude_none:
            data = {key: value for key, value in data.items() if value is not None}
            if not data.get("tool_calls"):
                data.pop("tool_calls", None)
        return data


class _FakeCompletions:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = self.scripted.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeClient:
    def __init__(self, scripted):
        self.completions = _FakeCompletions(scripted)
        self.chat = SimpleNamespace(completions=self.completions)


# ------------------------------------------------------------ 工具层测试
class TestAgentTools(unittest.TestCase):

    def test_analyze_experiments_tool(self):
        """测试实验分析工具的调用和返回结果。"""
        fake_rows = [{"final_cost": "10"}]
        fake_summary = {
            "total_experiments": 1,
            "success_count": 1,
            "success_rate": 100.0,
            "average_cost": 10.0,
            "best_cost": 10.0,
        }

        with patch.object(tools, "read_csv", return_value=fake_rows):
            with patch.object(
                tools, "summarize_experiment", return_value=fake_summary
            ) as mock_summary:
                result = tools.execute_tool("analyze_experiments", {})

        self.assertEqual(json.loads(result), fake_summary)
        # 没有给 worst_n 时，默认按 0 处理
        self.assertEqual(mock_summary.call_args.kwargs["worst_n"], 0)

    def test_analyze_experiments_with_worst_n(self):
        """worst_n 应该被透传给统计函数。"""
        fake_summary = {"total_experiments": 10, "worst_experiments": []}

        with patch.object(tools, "read_csv", return_value=[]):
            with patch.object(
                tools, "summarize_experiment", return_value=fake_summary
            ) as mock_summary:
                result = tools.execute_tool(
                    "analyze_experiments", {"worst_n": 3}
                )

        self.assertEqual(json.loads(result), fake_summary)
        self.assertEqual(mock_summary.call_args.kwargs["worst_n"], 3)

    def test_pcb_optimization_tool(self):
        """测试 PCB 优化工具的调用和返回结果。"""
        fake_result = {
            "before_cost": 400,
            "after_cost": 0,
            "before_out_of_bounds": 2,
            "after_out_of_bounds": 0,
            "before_overlaps": 2,
            "after_overlaps": 0,
            "best_cost": 0,
            "temperature_stages": 90,
            "elapsed_seconds": 1.2,
            "result_file": "test_output.txt",
        }

        with patch.object(
            tools, "run_pcb_optimization", return_value=fake_result
        ) as mock_optimize:
            result = tools.execute_tool("run_pcb_optimization", {})

        self.assertEqual(json.loads(result), fake_result)
        # 模型没有给参数时，不应该凭空塞默认值进去
        mock_optimize.assert_called_once_with()

    def test_pcb_optimization_forwards_parameters(self):
        """模型给出的搜索参数要能传到底层函数。"""
        fake_result = {"after_cost": 0}

        with patch.object(
            tools, "run_pcb_optimization", return_value=fake_result
        ) as mock_optimize:
            result = tools.execute_tool(
                "run_pcb_optimization",
                {
                    "components_file": "components_dense.txt",
                    "seed": 7,
                    "alpha": 0.9,
                    "iter_per_T": 300,
                    "step": 4.5,
                    "make_plots": True,
                },
            )

        self.assertEqual(json.loads(result), fake_result)
        mock_optimize.assert_called_once_with(
            components_file="components_dense.txt",
            seed=7,
            alpha=0.9,
            iter_per_T=300,
            step=4.5,
            make_plots=True,
        )

    def test_unknown_tool(self):
        """测试不支持的工具名称是否被拒绝。"""
        with self.assertRaises(ValueError):
            tools.execute_tool("unknown_tool", {})

    def test_unexpected_arguments(self):
        """测试工具收到不支持的参数时是否报错。"""
        with self.assertRaises(ValueError):
            tools.execute_tool("run_pcb_optimization", {"unexpected": True})

    def test_parameter_out_of_range(self):
        """参数越界要在执行前被拦住，而不是让算法跑一半才出问题。"""
        with patch.object(tools, "run_pcb_optimization") as mock_optimize:
            with self.assertRaises(ValueError):
                tools.execute_tool("run_pcb_optimization", {"alpha": 2.0})
            with self.assertRaises(ValueError):
                tools.execute_tool("run_pcb_optimization", {"seed": -1})
        mock_optimize.assert_not_called()

    def test_parameter_wrong_type(self):
        """参数类型不对也要报错（True 不能当整数用）。"""
        with self.assertRaises(ValueError):
            tools.execute_tool("run_pcb_optimization", {"seed": "abc"})
        with self.assertRaises(ValueError):
            tools.execute_tool("run_pcb_optimization", {"iter_per_T": True})

    def test_unknown_components_file(self):
        """只允许使用 data/ 目录里真实存在的元器件文件。"""
        with self.assertRaises(ValueError):
            tools.execute_tool(
                "run_pcb_optimization",
                {"components_file": "C:/Windows/win.ini"},
            )

    def test_tools_schema_lists_available_files(self):
        """工具声明里的枚举值应该包含两个算例文件。"""
        properties = tools.TOOLS[1]["function"]["parameters"]["properties"]
        choices = properties["components_file"]["enum"]
        self.assertIn("components.txt", choices)
        self.assertIn("components_dense.txt", choices)


# ---------------------------------------------------------- 统计函数测试
class TestSummarizeExperiment(unittest.TestCase):

    def test_median_and_worst(self):
        """中位数与最差实验明细要算对。"""
        rows = [
            {"seed": "1", "final_cost": "0"},
            {"seed": "2", "final_cost": "100"},
            {"seed": "3", "final_cost": "200"},
            {"seed": "4", "final_cost": "300"},
        ]

        summary = experiment_analysis.summarize_experiment(rows, worst_n=2)

        self.assertEqual(summary["total_experiments"], 4)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["best_cost"], 0)
        self.assertEqual(summary["worst_cost"], 300)
        self.assertEqual(summary["median_cost"], 150)
        self.assertEqual(
            [item["seed"] for item in summary["worst_experiments"]],
            ["4", "3"],
        )

    def test_empty_data(self):
        """空数据不能抛异常。"""
        summary = experiment_analysis.summarize_experiment([], worst_n=3)
        self.assertEqual(summary["total_experiments"], 0)
        self.assertEqual(summary["median_cost"], 0)


# ------------------------------------------------------------ 配置与密钥
class TestProviderConfig(unittest.TestCase):

    def setUp(self):
        agent.reset_client()

    tearDown = setUp

    def test_default_provider_is_deepseek(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PCB_AGENT_PROVIDER", None)
            os.environ.pop("PCB_AGENT_MODEL", None)
            name, settings = config.resolve_provider()
        self.assertEqual(name, "deepseek")
        self.assertEqual(settings["model"], "deepseek-flash")

    def test_switch_to_ollama(self):
        with patch.dict(os.environ, {"PCB_AGENT_PROVIDER": "ollama"}, clear=False):
            name, settings = config.resolve_provider()
        self.assertEqual(name, "ollama")
        self.assertIn("localhost", settings["base_url"])
        # 本地 Ollama 不需要真实 key
        self.assertEqual(config.read_api_key(settings), "ollama")

    def test_model_override(self):
        with patch.dict(os.environ, {"PCB_AGENT_MODEL": "whatever"}, clear=False):
            _, settings = config.resolve_provider()
        self.assertEqual(settings["model"], "whatever")

    def test_unknown_provider(self):
        with self.assertRaises(ValueError):
            config.resolve_provider("no-such-provider")

    def test_get_client_requires_api_key(self):
        """deepseek 缺少 key 时，只在真正需要客户端时才报错。"""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PCB_AGENT_PROVIDER", None)
            os.environ.pop("DEEPSEEK_API_KEY", None)
            with self.assertRaises(RuntimeError):
                agent.get_client("deepseek")


# ------------------------------------------------------- 工具调用循环测试
class TestAgentLoop(unittest.TestCase):

    def test_single_tool_call_then_answer(self):
        """一轮工具调用后，模型给出最终回答。"""
        client = _FakeClient([
            _FakeMessage(tool_calls=[
                _FakeToolCall("call_1", "analyze_experiments", "{}")
            ]),
            _FakeMessage(content="实验分析完成。"),
        ])

        history = []
        answer = agent.run_agent("帮我看看实验表现", history=history, client=client)

        self.assertEqual(answer, "实验分析完成。")
        # 两次请求都带工具声明（中间那轮模型还可能继续调）
        self.assertTrue(client.completions.calls[0]["tools"])
        self.assertTrue(client.completions.calls[1]["tools"])

        tool_messages = [m for m in history if m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_1")
        payload = json.loads(tool_messages[0]["content"])
        self.assertIn("total_experiments", payload)

    def test_multiple_rounds_of_tool_calls(self):
        """模型连续两轮要求调工具，循环要能接住。"""
        client = _FakeClient([
            _FakeMessage(tool_calls=[
                _FakeToolCall("call_1", "analyze_experiments", "{}")
            ]),
            _FakeMessage(tool_calls=[
                _FakeToolCall("call_2", "analyze_experiments", '{"worst_n": 2}')
            ]),
            _FakeMessage(content="两轮都跑完了。"),
        ])

        history = []
        answer = agent.run_agent("多轮", history=history, client=client)

        self.assertEqual(answer, "两轮都跑完了。")
        self.assertEqual(len(client.completions.calls), 3)
        tool_messages = [m for m in history if m.get("role") == "tool"]
        self.assertEqual(
            [m["tool_call_id"] for m in tool_messages], ["call_1", "call_2"]
        )

    def test_bad_json_arguments_become_error_result(self):
        """模型给出非法 JSON 时，错误要回填给它，而不是让程序崩掉。"""
        client = _FakeClient([
            _FakeMessage(tool_calls=[
                _FakeToolCall("call_1", "analyze_experiments", "{not json}")
            ]),
            _FakeMessage(content="参数有误。"),
        ])

        history = []
        answer = agent.run_agent("错误参数", history=history, client=client)

        self.assertEqual(answer, "参数有误。")
        tool_messages = [m for m in history if m.get("role") == "tool"]
        payload = json.loads(tool_messages[0]["content"])
        self.assertFalse(payload["success"])

    def test_round_limit_forces_final_answer(self):
        """模型一直要求调工具时，达到上限要强制收尾，不能死循环。"""
        scripted = [
            _FakeMessage(tool_calls=[
                _FakeToolCall("call_%d" % i, "analyze_experiments", "{}")
            ])
            for i in range(3)
        ]
        scripted.append(_FakeMessage(content="被迫收尾。"))
        client = _FakeClient(scripted)

        answer = agent.run_agent(
            "无限调工具", history=[], client=client, max_rounds=3
        )

        self.assertEqual(answer, "被迫收尾。")
        self.assertEqual(len(client.completions.calls), 4)
        # 最后一次请求不再提供工具
        self.assertNotIn("tools", client.completions.calls[-1])


# --------------------------------------------------------- PCB 工具集成
class TestPcbToolIntegration(unittest.TestCase):
    """真实跑一遍（小规模）优化，验证工具返回值自洽。"""

    def test_run_on_dense_instance(self):
        result = pcb_tool.run_pcb_optimization(
            components_file="components_dense.txt",
            seed=10,
            iter_per_T=20,      # 缩小规模，测试跑得快
            make_plots=False,
            output_prefix="unittest",
        )

        self.assertEqual(result["components_file"], "components_dense.txt")
        self.assertEqual(result["parameters"]["iter_per_T"], 20)
        self.assertEqual(result["best_cost"], result["after_cost"])
        self.assertLessEqual(result["after_cost"], result["before_cost"])
        self.assertEqual(
            result["candidate_attempts"],
            result["temperature_stages"] * 20,
        )
        self.assertTrue(os.path.exists(result["result_file"]))

        os.remove(result["result_file"])

    def test_invalid_parameter_is_rejected(self):
        """直接调用工具层也要守住参数范围。"""
        with self.assertRaises(ValueError):
            pcb_tool.run_pcb_optimization(alpha=2.0)
        with self.assertRaises(ValueError):
            pcb_tool.run_pcb_optimization(components_file="nope.txt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
