# LangChain 多智能体协作系统 v2.0

基于 **LangGraph** + **LangChain** 的生产级多智能体编排平台。6 个专业化 AI Agent 通过状态机流水线协同工作，支持 RAG 知识库检索、实时联网搜索、安全代码执行、可配置护栏、多会话记忆，以及四种交互界面。

---

## 架构总览

```
用户输入
    │
    ▼
┌─────────────────┐     ┌──────────────────┐
│   Guardrails     │────▶│  config/         │
│   (输入校验)      │     │  guardrails.yml  │
└─────────────────┘     └──────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│           LangGraph StateGraph 流水线              │
│                                                   │
│  router ──simple──▶ simple_reply (直接 LLM 回答)   │
│      │                                            │
│      ├─code─────▶ coder ──▶ summary               │
│      ├─rag──────▶ retrieval ──▶ summary           │
│      ├─search───▶ retrieval ──▶ summary           │
│      └─complex──▶ planner ──▶ retrieval ──▶ coder │
│                                      └─▶ summary  │
│                                                   │
│  Agent: Router │ Plan │ RAG │ Search │ Coder │    │
│                                                   │
│  Middleware: 监控 │ 性能 │ 异常处理 │ 日志 │       │
│  记忆注入 │ Token 计数                              │
└──────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐
│   Guardrails     │
│   (输出校验)      │
└─────────────────┘
    │
    ▼
 最终答案
```

---

## 核心特性

### 🤖 多 Agent 流水线（LangGraph StateGraph）
- **RouterAgent** — 一次轻量 LLM 调用，将查询分类为 `simple | code | rag | search | complex`
- **PlanAgent** — 将复杂需求拆解为结构化执行计划
- **RAGAgent** — 从本地 ChromaDB 向量库检索文档并生成带引用的回答
- **SearchAgent** — 实时联网搜索：Tavily（主）+ DuckDuckGo（备选）
- **CoderAgent** — 生成并执行 Python 代码（安全沙箱内）
- **SummaryAgent** — 汇聚所有中间结果，合成最终回答

### 🛡️ Guardrails（安全护栏）
- **输入规则：** 最大长度截断 / 拦截、敏感主题检测、Prompt 注入扫描
- **输出规则：** 系统提示泄露检测、输出长度上限
- 通过 `config/guardrails.yml` 配置，规则可插拔

### 💾 多会话记忆系统
- **滑动窗口：** 保留最近 N 轮对话（FIFO，数量可配）
- **摘要压缩：** 超出阈值时自动用 LLM 压缩历史为摘要，节省 Token
- **JSON 持久化：** 按 `session_id` 存储到 `memory/sessions/{id}.json`

### 🔒 安全代码沙箱
多层防护：
1. **AST 语法预检** — 执行前验证语法合法性
2. **导入拦截** — 禁止 `os`、`subprocess`、`socket`、`shutil`、`ctypes` 等危险包
3. **关键词过滤** — `eval(`、`exec(`、`open(`、`__import__` 等
4. **子进程隔离** — 空环境变量、30s 超时、临时文件自动清理

### ⚡ 并行执行
RAG 和 Search Agent 通过 `ThreadPoolExecutor` 并发执行，流水线延迟降低约 50%。

### 🧩 Middleware 中间件系统
6 个 LangChain 中间件（基于装饰器 AOP）：

| 中间件 | 作用 |
|--------|------|
| `monitor_tool` | 记录每次工具调用及其参数 |
| `log_before_model` | 记录调用 LLM 前的消息数量和最后一条内容 |
| `performance_middleware` | 统计每个工具的执行耗时（平均/最大） |
| `error_handler_middleware` | 工具调用异常统一兜底 |
| `memory_inject_middleware` | 将历史对话注入 LLM 上下文 |
| `token_counter_middleware` | 预估每次请求的 Token 消耗 |

### 🔧 四种交互界面

| 界面 | 启动方式 | 说明 |
|------|----------|------|
| **CLI 命令行** | `python main.py` | 交互式 REPL，退出时输出性能统计 |
| **Streamlit 网页** | `streamlit run ui/app.py` | 聊天式 Web UI，可展开中间结果 |
| **PyQt5 桌面** | `python ui/qt_app.py` | 原生桌面应用，异步非阻塞，折叠面板 |
| **FastAPI 接口** | `uvicorn api.main_api:app` | REST API，含健康检查/任务执行/统计 |

---

## 技术栈

| 层级 | 技术 |
|------|------|
| **流程编排** | LangGraph 1.2（StateGraph、条件边） |
| **Agent 框架** | LangChain 1.3（Agent、Tool、Middleware、Prompt） |
| **语言模型** | DeepSeek（OpenAI 兼容接口，通过 ChatOpenAI 调用） |
| **向量嵌入** | DashScope（阿里云通义）— `text-embedding-v4` |
| **向量数据库** | ChromaDB（本地持久化） |
| **联网搜索** | Tavily API（主）+ DuckDuckGo（备选） |
| **文档解析** | PyPDF、Unstructured（txt/pdf） |
| **后端 API** | FastAPI + Uvicorn + Pydantic v2 |
| **Web 界面** | Streamlit |
| **桌面界面** | PyQt5（QThread 异步） |
| **配置管理** | YAML（护栏、提示词、搜索、向量库） |
| **安全机制** | AST 沙箱、子进程隔离 |
| **MCP 集成** | MCP 客户端（stdio/SSE 协议连接 MCP Server，动态发现工具与资源） |
| **持久化** | JSON（对话记忆）、ChromaDB（向量数据） |

---

## 快速开始

### 环境要求
- Python 3.10+
- API Key：[DeepSeek](https://platform.deepseek.com)、[DashScope](https://dashscope.aliyun.com)、[Tavily](https://tavily.com)

### 安装

```bash
# 克隆项目
git clone <repo-url>
cd multi-agent

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate      # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# （可选）加载知识库文档
# 将 .txt 或 .pdf 文件放入 Knowledge_base_materials/
python -c "from vector_db.vector_store import VectorStore; VectorStore().load_document()"
```

### 运行

```bash
# CLI 模式
python main.py

# Web 界面
streamlit run ui/app.py

# 桌面应用
python ui/qt_app.py

# REST API
uvicorn api.main_api:app --reload
```

---

## 项目结构

```
multi-agent/
├── main.py                       # CLI 入口
├── requirements.txt              # Python 依赖
├── .env                          # 环境变量（不提交）
├── .env.example                  # 环境变量模板
│
├── agents/                       # Agent 层（6 个 Agent）
│   ├── base_agent.py             # 抽象基类
│   ├── router_agent.py           # 路由分类
│   ├── plan_agent.py             # 任务规划
│   ├── rag_agent.py              # RAG 知识库检索
│   ├── search_agent.py           # 联网搜索
│   ├── coder_agent.py            # 代码生成与执行
│   └── summary_agent.py          # 结果汇总
│
├── agent_tools/                  # 工具层
│   ├── rag_tool.py               # RAG 检索工具
│   ├── web_search_tool.py        # 联网搜索（Tavily + DuckDuckGo）
│   ├── code_interpreter.py       # 安全 Python 沙箱
│   ├── mcp_tool.py               # MCP 异步客户端（stdio/SSE）
│   ├── mcp_tools.py              # MCP 同步工具封装（@tool，可直接注入 Agent）
│   └── middleware.py             # 6 个中间件
│
├── graph/
│   └── agent_graph.py            # LangGraph StateGraph 编排
│
├── guardrails/                   # 安全护栏
│   ├── guardrails.py             # 护栏引擎
│   └── rules/                    # 可插拔规则
│       ├── base_rule.py          # 规则抽象基类
│       ├── input_rules.py        # 输入校验规则
│       └── output_rules.py       # 输出校验规则
│
├── memory/                       # 对话记忆
│   └── conversation_memory.py    # 滑动窗口 + 摘要 + 持久化
│
├── models/
│   └── my_model.py               # LLM 和 Embedding 模型工厂
│
├── RAG/
│   └── RAG_Service.py            # 检索增强生成流水线
│
├── vector_db/
│   └── vector_store.py           # ChromaDB 向量库封装
│
├── config/                       # YAML 配置文件
│   ├── guardrails.yml            # 安全规则配置
│   ├── prompts.yml               # 提示词映射
│   ├── search_tool.yml           # 搜索工具设置
│   └── vector_db.yml             # 向量库参数
│
├── prompts/                      # Agent 提示词模板
│   ├── router_prompt.txt
│   ├── plan_prompt.txt
│   ├── rag_prompt.txt
│   ├── search_prompt.txt
│   ├── coder_prompt.txt
│   └── summary_prompt.txt
│
├── utils/                        # 通用工具
│   ├── config_tool.py            # YAML 配置加载
│   ├── path_tool.py              # 路径解析
│   ├── prompts_tool.py           # 提示词文件读取
│   ├── log_tool.py               # 日志配置
│   └── file_tool.py              # 文档处理
│
├── ui/                           # 用户界面
│   ├── app.py                    # Streamlit Web 界面
│   └── qt_app.py                 # PyQt5 桌面界面
│
├── api/
│   └── main_api.py               # FastAPI REST 接口
│
├── knowledge_base_materials/     # RAG 知识库源文件
└── logs/                         # 运行时日志（自动生成）
```

---

## 配置说明

### 环境变量（`.env`）

| 变量 | 必填 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | 是 | DeepSeek API Key |
| `OPENAI_API_BASE` | 是 | API 地址（默认 `https://api.deepseek.com`） |
| `MODEL_NAME` | 是 | 模型名称（如 `deepseek-chat`） |
| `DASHSCOPE_API_KEY` | 是 | 阿里云 DashScope API Key（向量嵌入用） |
| `EMBEDDING_MODEL_NAME` | 是 | 嵌入模型（如 `text-embedding-v2`） |
| `TAVILY_API_KEY` | 是 | Tavily 联网搜索 API Key |
| `TEMPERATURE` | 否 | LLM 温度（默认 `0.1`） |

### YAML 配置文件

| 文件 | 用途 |
|------|------|
| `config/guardrails.yml` | 输入/输出安全规则、阈值、处理动作 |
| `config/prompts.yml` | 每个 Agent 对应的提示词模板文件 |
| `config/search_tool.yml` | 每次搜索返回的结果数量 |
| `config/vector_db.yml` | ChromaDB 集合配置、文本分块参数 |

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查（返回版本号） |
| `POST` | `/run_task` | 执行完整 Agent 流水线 |
| `GET` | `/stats` | 性能统计和 Token 用量 |

### 调用示例

```bash
curl -X POST http://localhost:8000/run_task \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是 LangGraph？", "session_id": "demo1"}'
```

---

## 设计模式

| 模式 | 实现位置 |
|------|----------|
| **状态图 (FSM)** | LangGraph StateGraph + 条件边路由 |
| **抽象基类 / 模板方法** | `BaseAgent` — 6 个 Agent 继承并实现 `run()` |
| **工厂方法** | `models/my_model.py` — 创建 LLM 和 Embedding 实例 |
| **单例** | `chat_model`、`embedding_model`、`agent_graph`、`guardrails` |
| **职责链** | Guardrails 规则链、Middleware 链、代码沙箱多层检查 |
| **中间件 / 拦截器** | AOP 风格，`@wrap_tool_call` / `@before_model` 装饰器 |
| **组合模式** | `ConversationMemory` = `SlidingWindowMemory` + `SummaryMemory` |
| **策略模式** | 双搜索引擎：Tavily 为主，DuckDuckGo 备选降级 |
| **重试模式** | 联网搜索自动重试 2 次 |
| **外观模式** | `VectorStore` 封装 ChromaDB、`Guardrails.wrap_graph()` |
| **并发执行** | `ThreadPoolExecutor` 并行执行 RAG + Search |
| **流水线** | Router → Plan → Retrieval → Code → Summary 顺序执行 |

---

## License

MIT
误隔离**: 每个 Graph 节点独立 try/except，单节点崩溃不影响流水线
- **记忆持久化**: 按 session_id 独立存储，支持多会话隔离
