"""
RAG 检索工具 — 基于 ChromaDB 向量库的本地知识库检索工具。

将用户提问送到 RAG 流水线（检索 + 生成），返回
引用知识库内容的回答。作为 LangChain @tool 对外暴露，
可被 RAGAgent 工具调用。
"""

from langchain_core.tools import tool
from RAG.RAG_Service import RAGService

# 全局单例 RAG 服务（内部持向量库连接池）
ragService = RAGService()


@tool(description='从知识库/向量库中检索参考资料')
def rag_summarize(que: str) -> str:
    """
    从本地知识库中检索与查询最相关的文档片段，
    并基于检索结果生成带引用的回答。
    Args:
        que: 用户查询字符串
    Returns: RAG 检索结果文本（含来源引用）
    """
    return ragService.RAG_result(que)
