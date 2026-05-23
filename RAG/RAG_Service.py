from vector_db.vector_store import VectorStore
from utils.prompts_tool import get_rag_prompts
from langchain_core.prompts import PromptTemplate
from models.my_model import chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

class RAGService():
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text =get_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._chain()

    def _chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain
    
    def retriever_docs(self, input:str) -> list[Document]:
        return self.retriever.invoke(input)
    
    def RAG_result(self, input: str):
        docs = self.retriever_docs(input)
        context = ''
        count = 0
        for doc in docs:
            count += 1
            context += f"[参考资料{count}]:参考资料:{doc.page_content} | 元数据:{doc.metadata}\n"
        
        return self.chain.invoke(
            {
                "input":input,
                "context":context
            }
        )
    
if __name__ == "__main__":
    rag = RAGService()
    print(rag.RAG_result("你的身份"))