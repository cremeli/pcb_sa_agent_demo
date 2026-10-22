"""项目配置：路径、模型服务、绘图样式。

模型服务不写死在代码里，用环境变量切换：

    PCB_AGENT_PROVIDER   deepseek（默认）或 ollama
    PCB_AGENT_MODEL      覆盖模型名
    DEEPSEEK_API_KEY     provider=deepseek 时必填
    OLLAMA_BASE_URL      provider=ollama 时的地址，默认 http://localhost:11434/v1/

例如在 PowerShell 里临时切到本地模型：

    $env:PCB_AGENT_PROVIDER="ollama"
    python agent.py
"""

import os
from pathlib import Path

# ---------------- 目录 ----------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

COMPONENTS_FILE = DATA_DIR / "components.txt"
COMPONENTS_DENSE_FILE = DATA_DIR / "components_dense.txt"
EXPERIMENT_CSV = DATA_DIR / "seed_experiment_results.csv"

# ---------------- 模型调用 ----------------
MAX_TOKENS = 1000
REQUEST_TIMEOUT = 60
MAX_TOOL_ROUNDS = 5  # 最多允许模型连续调用几轮工具，防止死循环

DEFAULT_PROVIDER = "deepseek"

# 每个 provider 的默认配置；环境变量可以覆盖 base_url / model
PROVIDER_DEFAULTS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
        "api_key": None,
        # 国内环境常见的问题是系统代理把请求拦下来，这里记下需要直连的域名
        "no_proxy_host": "api.deepseek.com",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1/",
        "model": "qwen3:1.7b",
        "api_key_env": None,
        "api_key": "ollama",  # 本地 Ollama 不校验 key，占位即可
        "no_proxy_host": "localhost",
    },
}


def resolve_provider(provider=None):
    """解析出最终使用的 provider 设置。

    返回 (provider 名称, 设置字典)，设置字典里包含 base_url / model /
    api_key_env / api_key / no_proxy_host。
    """
    name = (provider or os.getenv("PCB_AGENT_PROVIDER") or DEFAULT_PROVIDER).strip().lower()

    if name not in PROVIDER_DEFAULTS:
        raise ValueError(
            "未知的 provider：{!r}，可选值：{}".format(
                name, "、".join(sorted(PROVIDER_DEFAULTS))
            )
        )

    settings = dict(PROVIDER_DEFAULTS[name])

    # 本地服务的地址允许用环境变量覆盖（例如换了端口）
    if name == "ollama":
        settings["base_url"] = os.getenv("OLLAMA_BASE_URL", settings["base_url"])

    # 模型名允许统一用 PCB_AGENT_MODEL 覆盖
    model_override = os.getenv("PCB_AGENT_MODEL")
    if model_override:
        settings["model"] = model_override

    settings["name"] = name
    return name, settings


def read_api_key(settings):
    """读取当前 provider 需要的 API key；缺失时给出可操作的报错。"""
    env_name = settings.get("api_key_env")
    if not env_name:
        return settings.get("api_key")

    api_key = os.getenv(env_name)
    if not api_key:
        raise RuntimeError(
            "没有找到 {} 环境变量。\n"
            "PowerShell 里可以先执行：\n"
            '    $env:{}="你的真实密钥"'.format(env_name, env_name)
        )
    return api_key


def ensure_no_proxy(host):
    """把模型服务域名加入直连名单，避免被系统代理拦掉。

    只做追加，不覆盖用户已有的 NO_PROXY 设置。
    """
    if not host:
        return

    for name in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(name, "")
        items = [item.strip() for item in current.split(",") if item.strip()]
        if host not in items:
            items.append(host)
        os.environ[name] = ",".join(items)


def ensure_output_dir():
    """确保输出目录存在，返回它的路径。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def setup_matplotlib():
    """统一 matplotlib 的中文显示设置（幂等，可以重复调用）。"""
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
