from .config_tool import prompts_config
from .path_tool import get_abs_path
from .log_tool import logger

def get_router_prompts() -> str:
    """
    获取router提示词

    Returns:
        str: 返回提示词文本
    """
    try:
        path = get_abs_path(prompts_config['router_prompt_path'])
    except KeyError as e:
        logger.error(f"prompts.yml缺少key:router_prompt_path")
        raise e
    
    try:
        return open(path,'r',encoding="utf-8").read()
    except Exception as e:
        logger.error(f"解析router_prompts失败,{str(e)}")
        raise e

def get_plan_prompts() -> str:
    """
    获取plan提示词

    Returns:
        _type_: 返回提示词文本
    """
    try:
        path = get_abs_path(prompts_config['plan_prompt_path'])
    except KeyError as e:
        logger.error(f"prompts.yml缺少key:plan_prompt_path")
        raise e
    
    try:
        return open(path,'r',encoding="utf-8").read()
    except Exception as e:
        logger.error(f"解析plan_prompts失败,{str(e)}")
        raise e
    
def get_rag_prompts() -> str:
    """
    获取rag提示词

    Returns:
        _type_: 返回提示词文本
    """
    try:
        path = get_abs_path(prompts_config['rag_prompt_path'])
    except KeyError as e:
        logger.error(f"prompts.yml缺少key:rag_prompt_path")
        raise e
    
    try:
        return open(path,'r',encoding="utf-8").read()
    except Exception as e:
        logger.error(f"解析rag_prompts失败,{str(e)}")
        raise e

def get_search_prompts() -> str:
    """
    获取search提示词

    Returns:
        str: 返回提示词文本
    """
    try:
        path = get_abs_path(prompts_config['search_prompt_path'])
    except KeyError as e:
        logger.error(f"prompts.yml缺少key:search_prompt_path")
        raise e
    
    try:
        return open(path,'r',encoding="utf-8").read()
    except Exception as e:
        logger.error(f"解析coder_prompts失败,{str(e)}")
        raise e

def get_coder_prompts() -> str:
    """
    获取coder提示词

    Returns:
        str: 返回提示词文本
    """
    try:
        path = get_abs_path(prompts_config['coder_prompt_path'])
    except KeyError as e:
        logger.error(f"prompts.yml缺少key:coder_prompt_path")
        raise e
    
    try:
        return open(path,'r',encoding="utf-8").read()
    except Exception as e:
        logger.error(f"解析coder_prompts失败,{str(e)}")
        raise e

def get_summary_prompts() -> str:
    """
    获取summary提示词

    Returns:
        _type_: 返回提示词文本
    """
    try:
        path = get_abs_path(prompts_config['summary_prompt_path'])
    except KeyError as e:
        logger.error(f"prompts.yml缺少key:summary_prompt_path")
        raise e
    
    try:
        return open(path,'r',encoding="utf-8").read()
    except Exception as e:
        logger.error(f"解析rag_prompts失败,{str(e)}")
        raise e
