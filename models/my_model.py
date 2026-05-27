from abc import ABC, abstractmethod
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
import os
from dotenv import load_dotenv
load_dotenv(override=True)

class BaseModel(ABC):
    @abstractmethod
    def create_model(self) -> Optional[BaseChatModel | Embeddings]:
        pass


class ChatModel(BaseModel):
    def create_model(self) -> Optional[BaseChatModel | Embeddings]:
        return ChatOpenAI(
            model = os.getenv("MODEL_NAME"),
            temperature = os.getenv("TEMPERATURE"),
            request_timeout = 15,
            max_retries = 0,
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_API_BASE")
        )


class EmbeddingsModel(BaseModel):
    def create_model(self) -> Optional[BaseChatModel | Embeddings]:
        return DashScopeEmbeddings(model = os.getenv("EMBEDDING_MODEL_NAME"),
            dashscope_api_key= os.getenv("DASHSCOPE_API_KEY"))


chat_model = ChatModel().create_model()
embedding_model = EmbeddingsModel().create_model()