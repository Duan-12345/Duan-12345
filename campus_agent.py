"""校园小助手 · MCP 智能体 —— 单文件版。

把这个文件放到任意装有 Python 3.9+ 的机器上就能跑，不需要任何第三方库：

    python campus_agent.py                # 多轮对话
    python campus_agent.py --demo         # 8 轮自动演示
    python campus_agent.py --once "算一下 (128+72)*3/4"
    python campus_agent.py --no-llm       # 纯离线，只用本地路由
    python campus_agent.py --selftest     # 内置自测（不联网、不耗 token）

它和分模块版本功能完全一致，只是把 agent/ 与 mcp_server/ 合并到了一起：

    第 1 节  常量与配置
    第 2 节  工具实现（待办 / 计算 / 天气 / 笔记）
    第 3 节  MCP 服务器（stdio + JSON-RPC 2.0）
    第 4 节  MCP 客户端（把服务器作为子进程拉起）
    第 5 节  本地零 token 路由
    第 6 节  大模型客户端 + 工具调用循环
    第 7 节  命令行界面与演示脚本
    第 8 节  自测与入口

配置：同目录放 .env，写 DEEPSEEK_API_KEY=sk-xxx；
在 MCP 客户端里注册本文件时，把 args 写成 ["--mcp-server"] 即可。
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import re
import shutil
import subprocess
import sys
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

# ============================================================================
# 第 1 节  常量与配置
# ============================================================================

FILE_PATH = Path(__file__).resolve()
ROOT = FILE_PATH.parent

# 数据目录：默认 ./data，可用环境变量覆盖（自测时指向临时目录）
DATA_DIR = Path(os.environ.get("MINI_AGENT_DATA_DIR") or (ROOT / "data"))

SERVER_INFO = {"name": "campus-mini-agent", "version": "1.0.0"}
PROTOCOL_VERSION = "2024-11-05"
SUPPORTED_VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18"}

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

FILE_NAME_MAX = 40
WEATHER_TIMEOUT = 8

SYSTEM_PROMPT = (
    "你是校园小助手，能用工具就调工具，能直接答就直接答。"
    "回复用中文，不超过 60 字，不要编造工具结果。"
    "工具：待办增删查、计算、天气、笔记读写。"
)


def data_dir() -> Path:
    return DATA_DIR


def files_dir() -> Path:
    return data_dir() / "files"


def todo_file() -> Path:
    return data_dir() / "todos.json"


# ============================================================================
# 第 2 节  工具实现
# ============================================================================

def _ensure_dirs() -> None:
    data_dir().mkdir(parents=True, exist_ok=True)
    files_dir().mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 待办 ----
def _load_todos() -> dict:
    _ensure_dirs()
    path = todo_file()
    if not path.exists():
        return {"next_id": 1, "items": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"next_id": 1, "items": []}


def _save_todos(state: dict) -> None:
    _ensure_dirs()
    path = todo_file()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # 先写临时文件再替换，避免写坏数据


def _pending(state: dict) -> int:
    return sum(1 for it in state["items"] if not it["done"])


def todo_add(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "错误：待办内容为空。"
    state = _load_todos()
    item = {
        "id": state["next_id"],
        "text": text[:200],
        "done": False,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    state["items"].append(item)
    state["next_id"] += 1
    _save_todos(state)
    return f"已添加待办 #{item['id']}：{item['text']}（当前共 {_pending(state)} 条未完成）"


def todo_list() -> str:
    state = _load_todos()
    if not state["items"]:
        return "待办清单是空的。"
    lines = []
    for it in state["items"][-20:]:  # 只回显最近 20 条，控制上下文长度
        mark = "x" if it["done"] else " "
        lines.append(f"[{mark}] #{it['id']} {it['text']}")
    return f"待办（未完成 {_pending(state)} 条）：\n" + "\n".join(lines)


def todo_done(id: int) -> str:
    state = _load_todos()
    for it in state["items"]:
        if it["id"] == int(id):
            if it["done"]:
                return f"#{id} 已经是完成状态。"
            it["done"] = True
            _save_todos(state)
            return f"已完成 #{id}：{it['text']}"
    return f"没找到编号为 {id} 的待办。"


# ---------------------------------------------------------------- 计算 ----
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("只支持数字常量")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 64:
            raise ValueError("指数过大")
        result = _BINOPS[type(node.op)](left, right)
        if abs(result) > 1e15:
            raise ValueError("结果过大")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("表达式含不支持的语法")


def calc(expression: str) -> str:
    """只允许四则运算、整除、取模、乘方和括号，绝不使用 eval。"""
    expression = (expression or "").strip().replace("×", "*").replace("÷", "/").replace("^", "**")
    if not expression:
        return "错误：表达式为空。"
    if len(expression) > 200:
        return "错误：表达式过长。"
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree)
    except ZeroDivisionError:
        return "错误：除数不能为 0。"
    except (SyntaxError, ValueError) as exc:
        return f"错误：无法计算（{exc}）。"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    elif isinstance(value, float):
        value = round(value, 6)
    return f"{expression} = {value}"


# ---------------------------------------------------------------- 天气 ----
_WMO = {
    0: "晴", 1: "晴间多云", 2: "多云", 3: "阴",
    45: "雾", 48: "雾凇", 51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷阵雨", 96: "雷阵雨伴小冰雹", 99: "雷阵雨伴大冰雹",
}


def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "campus-agent-single/1.0"})
    with urllib.request.urlopen(req, timeout=WEATHER_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _weather_open_meteo(city: str) -> str:
    geo_url = (
        "https://geocoding-api.open-meteo.com/v1/search?name="
        + urllib.parse.quote(city)
        + "&count=1&language=zh&format=json"
    )
    geo = json.loads(_http_get(geo_url))
    if not geo.get("results"):
        raise ValueError("城市未找到")
    place = geo["results"][0]
    lat, lon = place["latitude"], place["longitude"]
    fc_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
        "&timezone=auto"
    )
    data = json.loads(_http_get(fc_url))
    cur = data.get("current", {})
    desc = _WMO.get(int(cur.get("weather_code", -1)), "未知")
    name = place.get("name") or city
    return (
        f"{name} 当前 {cur.get('temperature_2m')}°C，{desc}，"
        f"体感 {cur.get('apparent_temperature')}°C，"
        f"湿度 {cur.get('relative_humidity_2m')}%，风速 {cur.get('wind_speed_10m')} km/h"
    )


def _weather_wttr(city: str) -> str:
    raw = _http_get("https://wttr.in/" + urllib.parse.quote(city) + "?format=j1")
    data = json.loads(raw)
    cur = data["current_condition"][0]
    return (
        f"{city} 当前 {cur['temp_C']}°C，{cur['weatherDesc'][0]['value']}，"
        f"体感 {cur['FeelsLikeC']}°C，湿度 {cur['humidity']}%"
    )


def weather(city: str) -> str:
    city = (city or "").strip() or "北京"
    try:
        return _weather_open_meteo(city)
    except Exception:
        try:
            return _weather_wttr(city)
        except Exception as exc:  # 网络不通时不要让整个 Agent 崩掉
            return f"暂时查不到 {city} 的天气（网络异常：{type(exc).__name__}）。"


# ---------------------------------------------------------------- 笔记 ----
def _safe_path(name: str) -> Path:
    name = (name or "").strip().replace("\\", "/").lstrip("/")
    if not name or len(name) > FILE_NAME_MAX:
        raise ValueError("文件名不合法")
    if "/" in name or ".." in name:
        raise ValueError("只允许在 notes 目录下使用简单文件名")
    if not name.endswith(".md"):
        name += ".md"  # 统一成 Markdown 笔记，避免写出奇怪后缀
    target = (files_dir() / name).resolve()
    if files_dir().resolve() not in target.parents:
        raise ValueError("路径越界")
    return target


def file_write(name: str, content: str) -> str:
    try:
        path = _safe_path(name)
    except ValueError as exc:
        return f"错误：{exc}。"
    _ensure_dirs()
    path.write_text(content or "", encoding="utf-8")
    return f"已写入 {path.name}（{len(content or '')} 字）"


def file_read(name: str) -> str:
    try:
        path = _safe_path(name)
    except ValueError as exc:
        return f"错误：{exc}。"
    if not path.exists():
        return f"文件 {path.name} 不存在。"
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:800]  # 截断，避免一次读入过长内容


# ---------------------------------------------------------- 工具注册表 ----
TOOL_SPECS = [
    {
        "name": "todo_add",
        "description": "添加一条待办",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "待办内容"}},
            "required": ["text"],
        },
        "fn": todo_add,
    },
    {
        "name": "todo_list",
        "description": "列出待办清单",
        "inputSchema": {"type": "object", "properties": {}},
        "fn": todo_list,
    },
    {
        "name": "todo_done",
        "description": "把指定编号的待办标记为完成",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "integer", "description": "待办编号"}},
            "required": ["id"],
        },
        "fn": todo_done,
    },
    {
        "name": "calc",
        "description": "计算四则运算表达式，如 (12+8)*3",
        "inputSchema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "数学表达式"}},
            "required": ["expression"],
        },
        "fn": calc,
    },
    {
        "name": "weather",
        "description": "查询某城市当前天气",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名，如 广州"}},
            "required": ["city"],
        },
        "fn": weather,
    },
    {
        "name": "file_write",
        "description": "把内容写入笔记文件",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "笔记文件名，如 会议要点"},
                "content": {"type": "string", "description": "文件内容"},
            },
            "required": ["name", "content"],
        },
        "fn": file_write,
    },
    {
        "name": "file_read",
        "description": "读取笔记文件内容",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "笔记文件名"}},
            "required": ["name"],
        },
        "fn": file_read,
    },
]

TOOL_MAP = {spec["name"]: spec for spec in TOOL_SPECS}


# ============================================================================
# 第 3 节  MCP 服务器（stdio + JSON-RPC 2.0）
# ============================================================================
# 协议要点（MCP 2024-11-05）：
#   initialize -> protocolVersion / capabilities / serverInfo
#   notifications/initialized 是通知，不回包
#   tools/list -> 工具名、描述、inputSchema
#   tools/call -> content[].text
# stdout 只允许输出协议 JSON，日志一律走 stderr。

def _rpc_result(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _rpc_error(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle_request(request: dict, log=None) -> dict | None:
    """处理单条 JSON-RPC 请求，返回响应字典；通知类请求返回 None。"""
    msg_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if method == "initialize":
        client_version = params.get("protocolVersion")
        version = client_version if client_version in SUPPORTED_VERSIONS else PROTOCOL_VERSION
        return _rpc_result(msg_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        })

    if method in ("notifications/initialized", "notifications/cancelled", "initialized"):
        return None  # 通知不需要回包

    if method == "ping":
        return _rpc_result(msg_id, {})

    if method == "tools/list":
        return _rpc_result(msg_id, {
            "tools": [
                {"name": s["name"], "description": s["description"], "inputSchema": s["inputSchema"]}
                for s in TOOL_SPECS
            ]
        })

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        spec = TOOL_MAP.get(name)
        if spec is None:
            return _rpc_result(msg_id, {
                "content": [{"type": "text", "text": f"未知工具：{name}"}],
                "isError": True,
            })
        try:
            text = spec["fn"](**arguments)
            is_error = isinstance(text, str) and text.startswith("错误：")
        except TypeError as exc:
            text, is_error = f"错误：参数不匹配（{exc}）", True
        except Exception as exc:  # 工具内部异常不应该拖垮服务器
            if log:
                log("工具异常：\n" + traceback.format_exc())
            text, is_error = f"错误：{type(exc).__name__}: {exc}", True
        return _rpc_result(msg_id, {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        })

    if method in ("resources/list", "prompts/list"):
        key = "resources" if method.startswith("resources") else "prompts"
        return _rpc_result(msg_id, {key: []})

    return _rpc_error(msg_id, -32601, f"Method not found: {method}")


def serve_stdio() -> int:
    """作为 MCP 服务器运行：从 stdin 逐行读 JSON-RPC，向 stdout 写响应。"""
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:  # pragma: no cover
        pass

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    log(f"[mcp] server {SERVER_INFO['name']} v{SERVER_INFO['version']} 已启动")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps(_rpc_error(None, -32700, "Parse error")), flush=True)
            continue

        try:
            response = handle_request(request, log=log)
        except Exception as exc:  # 兜底：未捕获异常转成 JSON-RPC 错误
            response = _rpc_error(request.get("id"), -32603, f"Internal error: {exc}")

        if response is not None:
            print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


# ============================================================================
# 第 4 节  MCP 客户端
# ============================================================================
# 把本文件以 --mcp-server 参数再拉起一个子进程，双方用 stdio 上的
# JSON-RPC 通信。工具因此跑在独立进程里，崩溃了也不会带走主对话。

class MCPError(RuntimeError):
    pass


class MCPClient:
    def __init__(self, command: list[str] | None = None, cwd: str | Path | None = None):
        self.command = command or [sys.executable, "-u", str(FILE_PATH), "--mcp-server"]
        self.cwd = str(cwd or ROOT)
        self._proc: subprocess.Popen | None = None
        self._next_id = 0
        self.server_info: dict = {}
        self.tools: list[dict] = []

    # ------------------------------------------------------ 生命周期 ----
    def start(self) -> "MCPClient":
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        self._proc = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        init = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "campus-agent-single", "version": "1.0.0"},
        })
        self.server_info = init.get("serverInfo", {})
        self._notify("notifications/initialized")
        self.tools = self.request("tools/list", {}).get("tools", [])
        return self

    def close(self) -> None:
        if self._proc:
            try:
                if self._proc.stdin and not self._proc.stdin.closed:
                    self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    pass
            finally:
                for stream in (self._proc.stdout, self._proc.stderr):
                    if stream and not stream.closed:
                        stream.close()
        self._proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info):
        self.close()

    # ------------------------------------------------------ 协议原语 ----
    def _send(self, payload: dict) -> None:
        if not self._proc or not self._proc.stdin:
            raise MCPError("MCP 服务器未启动")
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _notify(self, method: str, params: dict | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params
        self._send(payload)

    def request(self, method: str, params: dict) -> dict:
        self._next_id += 1
        msg_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})

        while True:
            line = self._proc.stdout.readline()
            if not line:
                err = self._proc.stderr.read() if self._proc.stderr else ""
                raise MCPError(f"MCP 服务器意外退出。stderr: {err[:500]}")
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue  # 理论上 stdout 只有协议报文，容错跳过
            if message.get("id") != msg_id:
                continue  # 忽略通知/其他响应
            if "error" in message:
                raise MCPError(f"{message['error'].get('message')}")
            return message.get("result", {})

    # ---------------------------------------------------------- 工具 ----
    def openai_tools(self) -> list[dict]:
        """把 MCP tools/list 的结果翻译成 OpenAI function calling 的 tools 参数。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
                },
            }
            for t in self.tools
        ]

    def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        parts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
        return "\n".join(parts).strip(), bool(result.get("isError"))


# ============================================================================
# 第 5 节  本地零 token 路由
# ============================================================================
# 确定性的口语指令（“我的待办”“算一下 3*7”）不需要大模型参与，
# 正则直接映射到工具，0 token。拿不准的一律放行给模型。

_TODO_ADD = re.compile(r"^(?:帮我)?(?:记|添加|新增|加)(?:一下|一条|个)?待办[：:，,\s]*(.+)$")
_TODO_LIST = re.compile(r"^(?:看看|查看|列出|显示|打开)?(?:我的)?待办(?:清单|列表)?[？?]?$")
_TODO_DONE = re.compile(r"^(?:把)?第?(\d+)(?:条|个|号)?(?:待办)?(?:标记为|标记|设为|改成)?(?:已)?完成$")
_CALC = re.compile(r"^(?:帮我)?(?:算一下|计算一下|计算|算算|算)\s*[：:]?\s*([0-9\s\.\+\-\*/\(\)×÷\^%]+?)\s*(?:等于几|等于多少|是多少)?[？?]?$")
_FILE_READ = re.compile(r"^(?:读一下|读取|打开|看看)\s*(?:笔记|文件)?\s*[《「]?(.{1,30}?)[》」]?\s*(?:的内容|内容)?[。？?]?$")
_WEATHER = re.compile(r"^(?:查一下|查询)?\s*(?:天气|气温)\s*[：:]?\s*([\u4e00-\u9fa5]{2,8})\s*(?:的天气)?[？?]?$")


def _is_arith(text: str) -> bool:
    has_digit = re.search(r"\d", text) is not None
    has_op = re.search(r"[\+\-\*/×÷\^%]", text) is not None
    only_arith_chars = re.fullmatch(r"[0-9\s\.\+\-\*/\(\)×÷\^%=]+", text) is not None
    return has_digit and has_op and only_arith_chars


def route(text: str) -> tuple[str, dict] | None:
    """命中规则返回 (工具名, 参数)，否则返回 None。"""
    text = (text or "").strip()
    if not text:
        return None
    core = text.rstrip("。！!？?")

    if _TODO_LIST.fullmatch(core):
        return "todo_list", {}

    m = _TODO_ADD.fullmatch(core)
    if m:
        return "todo_add", {"text": m.group(1).strip()}

    m = _TODO_DONE.fullmatch(core)
    if m:
        return "todo_done", {"id": int(m.group(1))}

    m = _CALC.fullmatch(core)
    if m and _is_arith(m.group(1)):
        return "calc", {"expression": m.group(1).strip()}

    if _is_arith(core):
        return "calc", {"expression": core}

    m = _WEATHER.fullmatch(core)
    if m:
        return "weather", {"city": m.group(1).strip()}

    m = _FILE_READ.fullmatch(core)
    if m:
        name = m.group(1).strip()
        # 排除“打开文件”“读取笔记”这类没给文件名的空指令
        if name not in ("文件", "笔记", "内容", "文件内容"):
            return "file_read", {"name": name}

    return None


# ============================================================================
# 第 6 节  大模型客户端 + 工具调用循环
# ============================================================================

class LLMError(RuntimeError):
    pass


class DeepSeekClient:
    """最小可用的 OpenAI 兼容客户端（只用标准库 urllib）。"""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL, timeout: float = 60.0, retries: int = 2):
        if not api_key:
            raise LLMError("缺少 API Key")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries

    @classmethod
    def from_env(cls) -> "DeepSeekClient":
        key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        return cls(
            key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int = 256, temperature: float = 0.0) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_error = ""
        for _ in range(self.retries + 1):
            try:
                return self._post(payload)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code in (400, 401, 403, 422):  # 鉴权/参数问题重试没意义
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        raise LLMError(last_error or "调用大模型失败")

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


class Agent:
    """两级路由 + 工具调用循环。"""

    def __init__(self, mcp: MCPClient, llm: DeepSeekClient | None = None,
                 system_prompt: str = SYSTEM_PROMPT,
                 max_tool_rounds: int = 3, history_messages: int = 6,
                 max_tokens: int = 256, use_router: bool = True):
        self.mcp = mcp
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds
        self.history_messages = history_messages
        self.max_tokens = max_tokens
        self.use_router = use_router
        self.history: list[dict] = []
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "cached_tokens": 0, "router_hits": 0, "tool_calls": 0}

    # ------------------------------------------------------ 主入口 ----
    def handle(self, text: str) -> dict:
        """处理一轮用户输入，返回 {reply, path, tools, usage}。"""
        text = (text or "").strip()
        if not text:
            return {"reply": "说点什么吧。", "path": "empty", "tools": [], "usage": {}}

        # 第一级：本地路由，命中即执行，0 token
        hit = route(text) if self.use_router else None
        if hit:
            name, args = hit
            output, is_error = self.mcp.call_tool(name, args)
            self.stats["router_hits"] += 1
            self.stats["tool_calls"] += 1
            self._remember(text, output)
            return {"reply": output, "path": "router",
                    "tools": [{"name": name, "arguments": args}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

        # 第二级：交给大模型
        if self.llm is None:
            return self._offline_reply(text)

        try:
            return self._llm_turn(text)
        except LLMError as exc:
            return {"reply": f"（大模型暂时不可用：{exc}）这条我先记着，"
                             f"你可以用“待办：xxx”的格式直接下达指令。",
                    "path": "error", "tools": [], "usage": {}}

    # ------------------------------------------------ 大模型轮次 ----
    def _llm_turn(self, text: str) -> dict:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += self.history[-self.history_messages:]
        messages.append({"role": "user", "content": text})

        tools = self.mcp.openai_tools()
        used_tools: list[dict] = []
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

        for _ in range(self.max_tool_rounds):
            response = self.llm.chat(messages, tools=tools, max_tokens=self.max_tokens)
            self._account(response.get("usage") or {}, usage_total)
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                reply = (message.get("content") or "").strip() or "（模型没有返回内容）"
                self._remember(text, reply)
                return {"reply": reply, "path": "llm", "tools": used_tools, "usage": usage_total}

            # 回填 assistant 的 tool_calls 消息（协议要求）
            messages.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": tool_calls,
            })
            for call in tool_calls:
                name = call["function"]["name"]
                raw_args = call["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except json.JSONDecodeError:
                    args = {}
                output, is_error = self.mcp.call_tool(name, args)
                self.stats["tool_calls"] += 1
                used_tools.append({"name": name, "arguments": args})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": output[:600] or ("工具出错" if is_error else "（无输出）"),
                })

        reply = "我调了几次工具还是没搞定，你能换个说法再试一次吗？"
        self._remember(text, reply)
        return {"reply": reply, "path": "llm", "tools": used_tools, "usage": usage_total}

    def _account(self, usage: dict, total: dict) -> None:
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        total["prompt_tokens"] = total.get("prompt_tokens", 0) + prompt
        total["completion_tokens"] = total.get("completion_tokens", 0) + completion
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += prompt
        self.stats["completion_tokens"] += completion
        self.stats["cached_tokens"] += cached

    # ---------------------------------------------------- 历史兜底 ----
    def _remember(self, user_text: str, reply: str) -> None:
        """只保存一问一答的摘要，工具报文不进历史，省 token。"""
        self.history.append({"role": "user", "content": user_text[:120]})
        self.history.append({"role": "assistant", "content": reply[:200]})
        self.history = self.history[-self.history_messages:]

    def _offline_reply(self, text: str) -> dict:
        names = "、".join(t["name"] for t in self.mcp.tools)
        reply = (f"现在是离线模式（未配置 DEEPSEEK_API_KEY），我只能执行固定指令。"
                 f"可用工具：{names}。例如“待办：周五交报告”“算一下 (12+8)*3”。")
        return {"reply": reply, "path": "offline", "tools": [], "usage": {}}

    def token_summary(self) -> str:
        s = self.stats
        total = s["prompt_tokens"] + s["completion_tokens"]
        return (f"模型调用 {s['calls']} 次 | 输入 {s['prompt_tokens']} + 输出 {s['completion_tokens']} "
                f"= {total} tokens（缓存命中 {s['cached_tokens']}） | 本地路由命中 {s['router_hits']} 次 | "
                f"工具调用 {s['tool_calls']} 次")


# ============================================================================
# 第 7 节  命令行界面与演示脚本
# ============================================================================

DEMO_TURNS = [
    "帮我记一条待办：周五 23:00 前交软件工程实验报告",
    "看看我的待办",
    "把第1条标记完成",
    "算一下 (128+72)*3/4",
    "明天要去广州开会，那边现在热不热？",
    "帮我把今天的组会要点写成笔记：确定选题为校园智能助理，下周三前完成原型",
    "读一下组会要点",
    "你好，你都能做什么？",
]


def load_dotenv() -> None:
    """极简 .env 解析：KEY=VALUE，忽略注释；不覆盖已有环境变量。"""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def banner(mcp: MCPClient, model: str) -> None:
    print("=" * 62)
    print(" 校园小助手 · MCP 智能体（会说 + 会做）")
    print(f" MCP 服务器：{mcp.server_info.get('name')} v{mcp.server_info.get('version')}")
    print(f" 工具数量：{len(mcp.tools)}  模型：{model}")
    print(" 输入 :help 查看命令，:quit 退出")
    print("=" * 62)


def print_turn(index: int, text: str, result: dict) -> None:
    path = {"router": "本地路由（0 token）", "llm": "大模型 + 工具调用",
            "offline": "离线模式", "error": "模型异常兜底"}.get(result.get("path"), result.get("path"))
    print(f"\n[{index:02d}] 我：{text}")
    for call in result.get("tools", []):
        args = ", ".join(f"{k}={v!r}" for k, v in call["arguments"].items())
        print(f"     ↳ 调用工具 {call['name']}({args})")
    usage = result.get("usage") or {}
    tokens = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
    print(f"     ↳ 路径：{path}，本轮 {tokens} tokens")
    print(f"     小助手：{result['reply']}")


def run_demo(agent: Agent) -> None:
    print("\n===== 自动演示开始 =====")
    for i, text in enumerate(DEMO_TURNS, 1):
        print_turn(i, text, agent.handle(text))
    print("\n===== 演示结束 =====")
    print(agent.token_summary())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校园小助手 —— 基于 MCP 的单文件智能体")
    parser.add_argument("--demo", action="store_true", help="运行自动演示脚本")
    parser.add_argument("--once", metavar="TEXT", help="只处理一句话后退出")
    parser.add_argument("--no-llm", action="store_true", help="只用本地路由，不调用大模型")
    parser.add_argument("--no-router", action="store_true",
                        help="关闭本地路由，全部交给大模型（用于对比 token 消耗）")
    parser.add_argument("--model", default=None, help="模型名，默认深寻 deepseek-chat")
    parser.add_argument("--quiet", action="store_true", help="只输出最终回复")
    parser.add_argument("--mcp-server", action="store_true", help="以 MCP 服务器模式运行（供客户端拉起）")
    parser.add_argument("--selftest", action="store_true", help="运行内置自测")
    args = parser.parse_args(argv)

    # 供 MCP 客户端拉起：只跑协议循环，不打印任何多余内容
    if args.mcp_server:
        return serve_stdio()

    if args.selftest:
        return selftest()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:  # pragma: no cover
        pass

    load_dotenv()

    llm = None
    if not args.no_llm:
        try:
            llm = DeepSeekClient.from_env()
            if args.model:
                llm.model = args.model
        except LLMError:
            llm = None

    with MCPClient() as mcp:
        agent = Agent(mcp, llm=llm, use_router=not args.no_router)
        model_name = llm.model if llm else "离线（仅本地路由）"

        if not args.quiet:
            banner(mcp, model_name)

        if args.demo:
            run_demo(agent)
            return 0

        if args.once:
            result = agent.handle(args.once)
            if args.quiet:
                print(result["reply"])
            else:
                print_turn(1, args.once, result)
                print(agent.token_summary())
            return 0

        index = 0
        while True:
            try:
                text = input("\n我：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break
            if not text:
                continue
            if text in (":quit", ":q", "exit", "quit"):
                print(agent.token_summary())
                break
            if text == ":help":
                print(":tools 查看工具 | :stats 查看 token 统计 | :reset 清空对话历史 | :quit 退出")
                continue
            if text == ":tools":
                for tool in mcp.tools:
                    print(f"  - {tool['name']}: {tool.get('description', '')}")
                continue
            if text == ":stats":
                print(agent.token_summary())
                continue
            if text == ":reset":
                agent.history.clear()
                print("对话历史已清空。")
                continue

            index += 1
            print_turn(index, text, agent.handle(text))
    return 0


# ============================================================================
# 第 8 节  自测与入口
# ============================================================================

def selftest() -> int:
    """内置自测：不联网、不调用大模型，覆盖工具、路由、MCP 协议、离线降级。"""
    global DATA_DIR
    original = DATA_DIR
    # 用项目内的临时目录：Windows 上系统临时目录可能没有写权限
    test_dir = ROOT / "data" / "_selftest"
    shutil.rmtree(test_dir, ignore_errors=True)
    DATA_DIR = test_dir

    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"  [OK]   {name}")
        else:
            print(f"  [FAIL] {name} {detail}")
            failures.append(name)

    try:
        print("\n[1] 计算器")
        check("1+2*3 = 7", calc("1+2*3") == "1+2*3 = 7", calc("1+2*3"))
        check("括号 60", calc("(12+8)*3") == "(12+8)*3 = 60", calc("(12+8)*3"))
        check("中文符号 6×7", calc("6×7") == "6*7 = 42", calc("6×7"))
        check("除零", "除数不能为 0" in calc("1/0"))
        check("注入被拒", calc("__import__('os').system('dir')").startswith("错误："))
        check("超大指数被拒", calc("9**9999").startswith("错误："))

        print("\n[2] 待办")
        check("添加", "已添加待办 #1" in todo_add("交实验报告"))
        check("列表", "#1 交实验报告" in todo_list())
        check("完成", "已完成 #1" in todo_done(1))
        check("重复完成", "已经是完成状态" in todo_done(1))

        print("\n[3] 笔记与越权")
        check("写入", "已写入 会议要点.md" in file_write("会议要点", "确定选题"))
        check("读回", file_read("会议要点") == "确定选题")
        check("拒绝 ../", file_write("../secret", "x").startswith("错误："))
        check("拒绝子目录", file_read("a/b").startswith("错误："))

        print("\n[4] 本地路由")
        check("待办新增", route("帮我记一条待办：周五交报告") == ("todo_add", {"text": "周五交报告"}))
        check("待办查询", route("看看我的待办") == ("todo_list", {}))
        check("计算", route("算一下 (128+72)*3/4") == ("calc", {"expression": "(128+72)*3/4"}))
        check("天气", route("天气 广州") == ("weather", {"city": "广州"}))
        check("模糊句交给模型", route("明天要去广州开会，那边现在热不热？") is None)

        print("\n[5] MCP 协议（真的拉起子进程）")
        with MCPClient() as client:
            check("服务器名", client.server_info.get("name") == "campus-mini-agent")
            check("工具数 = 7", len(client.tools) == 7, str(len(client.tools)))
            text, is_error = client.call_tool("calc", {"expression": "7*6"})
            check("tools/call 计算", (not is_error) and "42" in text, text)
            text, is_error = client.call_tool("不存在", {})
            check("未知工具报错", is_error)
            check("schema 转换", client.openai_tools()[0]["type"] == "function")

        print("\n[6] 离线降级")
        with MCPClient() as client:
            agent = Agent(client, llm=None)
            check("路由优先", agent.handle("算一下 6*7")["path"] == "router")
            check("无 Key 提示", agent.handle("你好呀，你能做什么")["path"] == "offline")
    finally:
        DATA_DIR = original
        shutil.rmtree(test_dir, ignore_errors=True)

    total = 28
    if failures:
        print(f"\n自测失败 {len(failures)} 项：{failures}")
        return 1
    print(f"\n自测全部通过（{total} 项检查）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
