import os
import hashlib
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from .log_tool import logger
from .config_tool import vector_db_config
from .path_tool import get_abs_path

def get_pdf_loader(path: str, password = None) -> list[Document]:
    """
    获取PDF文件Document

    Args:
        path (str): PDF文件路径
        password (_type_, optional): PDF密码. Defaults to None.

    Returns:
        list[Document]: Document集合
    """
    return PyPDFLoader(path, password).load()

def get_txt_loader(path: str) -> list[Document]:
    """
    获取TXT文件Document

    Args:
        path (str): TXT文件路径

    Returns:
        list[Document]: Document集合
    """
    return TextLoader(path, encoding="utf-8").load()

def get_md5_hex(filepath: str):
    """
    获取MD5十六进制字符

    Args:
        filepath (str): 将要转换MD5的文件路径
    """

    if not os.path.exists(filepath):
        logger.error(f"md5转换失败!{filepath}路径不存在")
        return
    
    if not os.path.isfile(filepath):
        logger.error(f"md5转换失败!{filepath}不是'文件'")
        return
    
    obj = hashlib.md5()

    chunk_size = 4096 # 4KB(Page Size)
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                obj.update(chunk)

            return obj.hexdigest()
    except Exception as e:
        logger.error(f"md5转换失败!,{str(e)}")
        return None

def check_md5_hex(md5_str: str) -> bool:
    """
    检查MD5是否存在

    Args:
        md5_str (str): MD5字符串

    Returns:
        bool: 返回是否存在该MD5
    """

    # MD5数据文件不存在时，创建
    if not os.path.exists(get_abs_path(vector_db_config["md5_hex_store"])):
        open(get_abs_path(vector_db_config["md5_hex_store"]),'w',encoding="utf-8").close()
        return False
    
    with open(get_abs_path(vector_db_config["md5_hex_store"]),'r',encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line == md5_str:
                return True
        return False

def save_md5_hex(md5_str: str):
    """
    保存MD5

    Args:
        md5_str (str): MD5字符串
    """
    with open(get_abs_path(vector_db_config["md5_hex_store"]),'a',encoding="utf-8") as f:
        f.write(md5_str + "\n")

def get_file_documents(path: str) -> list[Document]:
    """
    获取指定文件的Documents

    Args:
        path (str): 文件路径

    Returns:
        list[Document]: 返回Langchain中的文件Documents
    """
    if path.endswith("txt"):
        return get_txt_loader(path)
    if path.endswith("pdf"):
        return get_pdf_loader(path)
    return []

def allowed_file_type(path: str, types: tuple[str]) -> tuple[str]:
    """
    获取指定文件夹内 被允许的文件类型的文件路径集合

    Args:
        path (str): 文件夹 路径
        types (tuple[str]): 扩展名元组

    Returns:
        tuple[str]: 返回被允许的文件类型的文件路径集合
    """

    files = []

    if not os.path.isdir(path):
        logger.error(f"获取允许文件类型 {path}不是文件夹")
        return types
    
    for f in os.listdir(path):
        if f.endswith(types):
            files.append(os.path.join(path, f))

    return tuple(files)
    



    