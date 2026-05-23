# LangChain Multi-Agent Workbench — 多智能体协同工作台

基于 **LangGraph** 构建的生产级多智能体任务处理系统，支持任务规划、RAG 检索、联网搜索、代码执行、结果汇总全流程。

---

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置密钥
cp .env.example .env
# 编辑 .env 填入 API Key

# 3. 启动方式
python main.py                # CLI 交互模式
uvicorn api.main_api:app      # FastAPI 后端
streamlit run ui/app.py       # Web 可视化界面
```

## 系统架构

```
用户输入 → PlanAgent(任务规划)
               ↓
          RAGAgent(知识库检索)
               ↓
        SearchAgent(联网搜索)
               ↓
         CoderAgent(代码执行)
               ↓
       SummaryAgent(结果汇总)
               ↓
           最终输出
```

- **编排引擎**: LangGraph StateGraph — 5 个节点线性流转，共享状态对象
- **智能体**: 5 个 Agent 均继承 `BaseAgent` 抽象基类，统一 `init_agent()` / `run()` 接口
- **中间件**: 6 个 LangGraph 中间件（性能监控、Token 统计、记忆注入、错误兜底、日志追踪、工具监控）
- **记忆系统**: `ConversationMemory` = 滑动窗口 + 摘要压缩 + 按 session_id JSON 持久化
- **工具链**: Tavily/DuckDuckGo 搜索、ChromaDB 向量检索、subprocess 代码沙箱

## 技术栈

| 层 | 技术 |
|---|---|
| 编排 | LangChain 1.3 + LangGraph 1.2 |
| LLM | DashScope / OpenAI (兼容) |
| 向量库 | ChromaDB |
| 后端 | FastAPI + Pydantic v2 |
| 前端 | Streamlit |
| 配置 | YAML + .env |

## 目录结构

```
├── agents/          # 智能体层（5 个 Agent + 基类）
├── agent_tools/     # 工具 + 中间件
├── graph/           # LangGraph 状态图编排
├── memory/          # 会话记忆（窗口+摘要+持久化）
├── RAG/             # RAG 检索生成链路
├── vector_db/       # ChromaDB 向量库
├── chroma_db/       # 持久化存储
├── api/             # FastAPI 接口
├── ui/              # Streamlit 界面
├── config/          # YAML 配置中心
├── prompts/         # 系统提示词
├── models/          # LLM & Embedding 工厂
├── utils/           # 工具函数（日志、路径、文件）
├── knowledge_base_materials/  # 知识库原始文档
├── memory/sessions/ # 会话记忆持久化
├── main.py          # CLI 入口
└── requirements.txt # 依赖清单
```

## 工程设计亮点

- **抽象基类模式**: 所有 Agent 遵循同一生命周期，新增 Agent 只需实现 `init_agent()` 和 `run()`
- **配置代码分离**: 业务参数、提示词、模型名全部外置 YAML，零代码切换环境
- **中间件 AOP**: 通过 `@wrap_tool_call` / `@before_model` 解耦横切关注点
- **安全沙箱**: 代码执行在 subprocess 隔离环境，AST 扫描拦截危险操作
- **错误隔离**: 每个 Graph 节点独立 try/except，单节点崩溃不影响流水线
- **记忆持久化**: 按 session_id 独立存储，支持多会话隔离
