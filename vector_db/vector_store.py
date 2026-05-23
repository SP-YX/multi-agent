from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config_tool import vector_db_config
from models.my_model import embedding_model
from utils.file_tool import *
from utils.log_tool import logger

class VectorStore:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name= vector_db_config["collection_name"],
            embedding_function=embedding_model,
            persist_directory=vector_db_config["persist_directory"]
        )
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = vector_db_config["chunk_size"],
            chunk_overlap = vector_db_config["chunk_overlap"],
            separators = vector_db_config["separators"],
            length_function = len
        )
    
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k":vector_db_config["match_num"]})
    
    def load_document(self):
        allowed_files = allowed_file_type(
            get_abs_path(vector_db_config["data_path"]),
            tuple(vector_db_config["allow_file_type"])
        )

        for path in allowed_files:
            md5_str = get_md5_hex(path)
            if check_md5_hex(md5_str):
                logger.info(f"{md5_str}该资料文件已存在于知识库(向量库)内，已忽略")
                continue

            try:
                docs = get_file_documents(path)
                if not docs:
                    logger.warning(f"[获取Documents失败!]{path}无有效内容")
                    continue

                split_doc = self.splitter.split_documents(docs)

                if not split_doc:
                    logger.warning(f"[Document分片失败!]{path}无法分片")
                    continue

                self.vector_store.add_documents(split_doc)
                save_md5_hex(md5_str)

                logger.info(f"{path}知识库资料已加载成功!")

            except Exception as e:
                logger.error(f"[知识库资料加载失败!]:{str(e)}", exc_info = True)