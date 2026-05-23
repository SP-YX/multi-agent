import yaml
from .path_tool import get_abs_path

def get_model_config(path:str = get_abs_path('config/model.yml'), encoding:str = 'utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader= yaml.FullLoader)
    
def get_rag_config(path:str = get_abs_path('config/rag.yml'), encoding:str = 'utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader= yaml.FullLoader)
    
def get_agent_config(path:str = get_abs_path('config/agent.yml'), encoding:str = 'utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader= yaml.FullLoader)
    
def get_prompts_config(path:str = get_abs_path('config/prompts.yml'), encoding:str = 'utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader= yaml.FullLoader)
    
def get_vector_db_config(path:str = get_abs_path('config/vector_db.yml'), encoding:str = 'utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader= yaml.FullLoader)
    
def get_search_tool_config(path:str = get_abs_path('config/search_tool.yml'), encoding:str = 'utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader= yaml.FullLoader)
    
model_config = get_model_config() # 模型配置
rag_config = get_rag_config() # RAG配置
agent_config = get_agent_config() # 智能体配置
prompts_config = get_prompts_config() # 提示词配置
vector_db_config = get_vector_db_config() # 向量库配置
search_tool_config = get_search_tool_config() # 搜索工具配置

if __name__ == "__main__":
    print("sss")