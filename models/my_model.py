from abc import ABC,abstractmethod
from typing import Optional
from langchain_community.chat_models.tongyi import ChatTongyi,BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from utils.config_tool import model_config

class BaseModel(ABC):
    @abstractmethod
    def create_model(self) -> Optional[BaseChatModel | Embeddings]:
        pass

class ChatModel(BaseModel):
    def create_model(self) -> Optional[BaseChatModel | Embeddings]:
        return ChatTongyi(model=model_config["chat_model_name"])
    
class EmbeddingsModel(BaseModel):
    def create_model(self) -> Optional[BaseChatModel | Embeddings]:
        return DashScopeEmbeddings(model=model_config["embedding_model_name"])


chat_model = ChatModel().create_model()
embedding_model = EmbeddingsModel().create_model()