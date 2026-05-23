"""
FastAPI 后端接口服务 — 提供多智能体系统的 HTTP API。

端点：
  GET  /health    健康检查
  POST /run_task  执行任务（支持 session 隔离）
  GET  /stats     查看性能与 Token 统计数据
"""

import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from graph.agent_graph import agent_graph
from agent_tools.middleware import get_perf_stats, get_token_usage

app = FastAPI(
    title="多智能体协作系统 API",
    description="基于 LangGraph 的多智能体编排引擎，支持任务规划、RAG 检索、联网搜索、代码执行、结果汇总",
    version="2.0.0",
)


# ═══════════════════════════════════════════════
# 请求/响应模型
# ═══════════════════════════════════════════════

class TaskReq(BaseModel):
    """任务执行请求。"""
    query: str = Field(..., min_length=1, description="用户任务描述")
    session_id: str = Field(default="", description="会话ID（留空自动生成）")


class TaskResp(BaseModel):
    """任务执行响应，包含全流程中间结果和最终答案。"""
    session_id: str
    sub_tasks: str
    rag_result: str
    search_result: str
    code_result: str
    final_answer: str


class HealthResp(BaseModel):
    """健康检查响应。"""
    status: str
    version: str


# ═══════════════════════════════════════════════
# API 端点
# ═══════════════════════════════════════════════

@app.get("/health", response_model=HealthResp)
def health_check():
    """健康检查端点，用于探活。"""
    return {"status": "ok", "version": "2.0.0"}


@app.post("/run_task", response_model=TaskResp)
def run_task(req: TaskReq):
    """
    执行多智能体任务。
    流程：计划 → RAG → 搜索 → 代码 → 汇总。
    Args:
        req: { query, session_id? }
    Returns: 全流程中间结果 + 最终回答
    """
    # session_id 为空时自动生成
    session_id = req.session_id or str(uuid.uuid4())[:8]
    try:
        res = agent_graph.invoke({
            "query": req.query,
            "session_id": session_id,
        })
        return TaskResp(
            session_id=session_id,
            sub_tasks=res.get("sub_tasks", ""),
            rag_result=res.get("rag_result", ""),
            search_result=res.get("search_result", ""),
            code_result=res.get("code_result", ""),
            final_answer=res.get("final_answer", ""),
        )
    except Exception as e:
        # 兜底异常处理，返回 500
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    """
    获取系统运行时统计数据。
    Returns: 工具性能指标 + Token 消耗预估
    """
    return {
        "performance": get_perf_stats(),
        "token_usage": get_token_usage(),
    }
