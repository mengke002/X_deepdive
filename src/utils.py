"""
工具模块 - 提供通用工具函数
"""
import os
import logging
from typing import Optional

from .config import config, DATA_SOURCE_DATABASE

logger = logging.getLogger(__name__)


def get_data_source() -> str:
    """
    获取当前数据源类型（仅支持数据库模式）
    
    Returns:
        'database'
    """
    return config.get_data_source()


def is_using_database() -> bool:
    """检查是否使用数据库作为数据源（始终返回True）"""
    return True


def ensure_output_dir(output_dir: str = 'output') -> str:
    """
    确保输出目录存在
    
    Args:
        output_dir: 输出目录路径
    
    Returns:
        输出目录的绝对路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")
    return os.path.abspath(output_dir)


def format_number(num: int) -> str:
    """
    格式化数字显示（K/M格式）
    
    Args:
        num: 要格式化的数字
    
    Returns:
        格式化后的字符串
    """
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)
