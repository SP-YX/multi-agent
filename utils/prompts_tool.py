from .config_tool import prompts_config
from .path_tool import get_abs_path
from .log_tool import logger
from .harness_tool import get_harness_rules

_HARNESS_RULES = ""


def _inject_harness(prompt: str) -> str:
    """向提示词末尾追加 Harness 约束规则（仅首次加载）。"""
    global _HARNESS_RULES
    if not _HARNESS_RULES:
        _HARNESS_RULES = get_harness_rules()
    if _HARNESS_RULES:
        return prompt + "\n" + _HARNESS_RULES
    return prompt

def _read_prompt(path: str) -> str:
    """读取提示词文件并注入 Harness 约束规则。"""
    with open(path, 'r', encoding="utf-8") as f:
        return _inject_harness(f.read())


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
        return _read_prompt(path)
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
        return _read_prompt(path)
    except Exception as e:
        logger.error(f"解析plan_prompts失败,{str(e)}")
        raise e

def get_rag_prompt() -> str:
    try:
        path = get_abs_path(prompts_config['rag_prompt_path'])
    except KeyError as e:
        logger.error(f"prompts.yml缺少key:rag_prompt_path")
        raise e
    try:
        return _read_prompt(path)
    except Exception as e:
        logger.error(f"解析rag_prompt失败,{str(e)}")
        raise e

def get_rag_agent_prompt() -> str:
    try:
        path = get_abs_path(prompts_config['rag_agent_prompt_path'])
    except KeyError as e:
        logger.error(f"prompts.yml缺少key:rag_agent_prompt_path")
        raise e
    try:
        return _read_prompt(path)
    except Exception as e:
        logger.error(f"解析rag_agent_prompt失败,{str(e)}")
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
        return _read_prompt(path)
    except Exception as e:
        logger.error(f"解析search_prompts失败,{str(e)}")
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
        return _read_prompt(path)
    except Exception as e:
        logger.error(f"解析coder_prompts失败,{str(e)}")
        raise e

def get_skill_prompts() -> str:
    try:
        path = get_abs_path(prompts_config['skill_prompt_path'])
    except KeyError as e:
        logger.error(f"prompts.yml缺少key:skill_prompt_path")
        raise e
    try:
        return _read_prompt(path)
    except Exception as e:
        logger.error(f"解析skill_prompt失败,{str(e)}")
        raise e


def get_eval_prompts() -> str:
    """
    获取eval提示词

    Returns:
        str: 返回提示词文本
    """
    try:
        path = get_abs_path(prompts_config['eval_prompt_path'])
    except KeyError as e:
        logger.error(f"prompts.yml缺少key:eval_prompt_path")
        raise e
    try:
        return _read_prompt(path)
    except Exception as e:
        logger.error(f"解析eval_prompts失败,{str(e)}")
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
        return _read_prompt(path)
    except Exception as e:
        logger.error(f"解析summary_prompts失败,{str(e)}")
        raise e
