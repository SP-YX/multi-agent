from abc import ABC, abstractmethod
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from utils.config_tool import model_config, agent_config


class BaseModel(ABC):
    @abstractmethod
    def create_model(self) -> Optional[BaseChatModel | Embeddings]:
        pass


class ChatModel(BaseModel):
    def create_model(self) -> Optional[BaseChatModel | Embeddings]:
        llm_cfg = agent_config.get("llm", {})
        return ChatOpenAI(
            model=model_config["chat_model_name"],
            temperature=llm_cfg.get("temperature", 0.1),
            request_timeout=15,
            max_retries=0,
            base_url=llm_cfg.get("api_base"),
        )


class EmbeddingsModel(BaseModel):
    def create_model(self) -> Optional[BaseChatModel | Embeddings]:
        return DashScopeEmbeddings(model=model_config["embedding_model_name"])


chat_model = ChatModel().create_model()
embedding_model = EmbeddingsModel().create_model()