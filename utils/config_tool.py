import yaml
from .path_tool import get_abs_path

def get_prompts_config(path:str = get_abs_path('config/prompts.yml'), encoding:str = 'utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader= yaml.FullLoader)
    
def get_vector_db_config(path:str = get_abs_path('config/vector_db.yml'), encoding:str = 'utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader= yaml.FullLoader)
    
def get_search_tool_config(path:str = get_abs_path('config/search_tool.yml'), encoding:str = 'utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader= yaml.FullLoader)

prompts_config = get_prompts_config() # 提示词配置
vector_db_config = get_vector_db_config() # 向量库配置
search_tool_config = get_search_tool_config() # 搜索工具配置
