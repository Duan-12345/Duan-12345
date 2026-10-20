# 校园小助手 · MCP 智能体（会说 + 会做）

> 软件工程课程作业 —— 用 **Model Context Protocol（MCP）** 把一个“会聊天”的助手变成“能办事”的智能体。
> 纯 Python 标准库实现，零第三方依赖；内置**本地零 token 路由**，同样的 8 轮任务比“全交给大模型”省 **65%** 的 token。

## 它能做什么

| 你说 | 它做的事 |
| --- | --- |
| 帮我记一条待办：周五 23:00 前交实验报告 | 写入待办清单，返回编号 |
| 看看我的待办 / 把第 1 条标记完成 | 查询、勾选完成 |
| 算一下 (128+72)*3/4 | 安全表达式求值，返回 150 |
| 明天要去广州开会，那边热不热？ | 调天气接口，用人话总结 |
| 帮我把组会要点写成笔记：…… | 生成 Markdown 笔记文件 |
| 读一下组会要点 | 读回笔记内容 |

前三条这类**确定指令**由本地路由直接执行，一次模型调用都不花；只有需要“听懂人话”的请求才交给大模型。

## 快速开始

```bash
git clone <你的仓库地址>
cd mcp-mini-agent
cp .env.example .env          # 填入 DEEPSEEK_API_KEY
python run.py --demo          # 一键演示（不填 Key 也能跑前 5 条）
```

无第三方依赖，Python 3.9+ 即可运行。

```bash
python run.py                 # 交互式多轮对话
python run.py --once "算一下 (12+8)*3"
python run.py --no-llm        # 纯离线模式，只用本地路由
python run.py --demo --no-router   # 关掉路由，全交给大模型（token 对比实验）
python -m unittest discover -s tests -v   # 20 项离线测试
```

## 架构

```
用户输入
   │
   ├─ 本地路由 agent/router.py（正则，0 token）
   │        └─ 命中 → 直接调工具 → 回复
   │
   └─ 未命中 → agent/brain.py 调 DeepSeek（OpenAI 兼容接口）
                     │  携带 tools/list 的工具清单
                     ├─ 模型返回 tool_calls → MCP tools/call → 结果回填
                     └─ 模型生成最终回复

工具运行在独立进程：mcp_server/server.py（stdio + JSON-RPC 2.0）
```

## 目录结构

```
mcp-mini-agent/
├── run.py                  入口
├── agent/
│   ├── cli.py              命令行界面、演示脚本
│   ├── brain.py            大模型调用 + 工具调用循环 + token 统计
│   ├── router.py           本地零 token 意图路由
│   └── mcp_client.py       MCP 客户端（stdio JSON-RPC）
├── mcp_server/
│   ├── server.py           MCP 服务器（initialize / tools/list / tools/call）
│   └── tools.py            7 个工具的实现
├── tests/test_all.py       20 项离线测试
├── docs/
│   ├── Agent说明文档.md
│   └── 博客随笔.md
└── data/                   运行时数据（待办、笔记，已 gitignore）
```

## 工具清单（MCP tools）

| 工具 | 参数 | 说明 |
| --- | --- | --- |
| `todo_add` | `text` | 添加待办 |
| `todo_list` | — | 列出待办 |
| `todo_done` | `id` | 标记完成 |
| `calc` | `expression` | AST 安全求值，不用 `eval` |
| `weather` | `city` | open-meteo / wttr.in，双源兜底 |
| `file_write` | `name`, `content` | 写笔记 |
| `file_read` | `name` | 读笔记 |

## 省 token 的六个做法

1. **本地路由**：确定指令 0 token（本项目最主要的省法，见 `docs/Agent说明文档.md` 对照表）。
2. **系统提示词极简**：60 字以内，不分点、不举例。
3. **工具描述短**：每个工具一行说明，参数只标必填。
4. **历史只留最近 3 轮**，且工具报文不写回历史。
5. `temperature=0`、`max_tokens=256`，避免长回复。
6. 工具输出截断（≤600 字）再回填模型。

## 安全设计

- 计算器用 `ast` 白名单求值，注入 `__import__('os').system(...)` 直接判错。
- 文件工具只能写 `data/files/` 下的简单文件名，`../`、绝对路径、子目录一律拒绝。
- 待办写入用临时文件 + `replace` 原子替换，避免写坏。
- API Key 只放 `.env`（已 gitignore），仓库里只有 `.env.example`。

## 把它接到别的 MCP 客户端

`mcp_server` 是标准 stdio MCP 服务器，可直接挂到 Claude Desktop、Cherry Studio 等客户端：

```json
{
  "mcpServers": {
    "campus-mini-agent": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "<项目绝对路径>"
    }
  }
}
```

## 致谢

- [Model Context Protocol](https://modelcontextprotocol.io/)
- DeepSeek（OpenAI 兼容 Chat Completions）
- open-meteo / wttr.in 免费天气数据
