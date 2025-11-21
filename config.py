"""
配置管理模块
支持环境变量 > config.ini > 默认值的优先级机制
"""
import os
import configparser
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class Config:
    """配置管理类，支持环境变量优先级的配置加载"""

    def __init__(self, config_path: str = 'config.ini'):
        # 本地开发时可加载 .env
        try:
            load_dotenv()
        except Exception:
            pass

        self.config_parser = configparser.ConfigParser()

        # 兼容多种位置查找 config.ini
        possible_paths = [
            config_path,
            os.path.join(os.getcwd(), config_path),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path),
        ]

        self.config_file = None
        for p in possible_paths:
            if os.path.exists(p):
                self.config_file = p
                break

        if self.config_file:
            try:
                self.config_parser.read(self.config_file, encoding='utf-8')
                logger.info(f"已加载配置文件: {self.config_file}")
            except (configparser.Error, UnicodeDecodeError):
                logger.warning("读取配置文件失败，跳过。")
        else:
            pass

    def _get_config_value(self, section: str, key: str, env_var: str, default_value: Any, value_type=str) -> Any:
        """按优先级获取配置值：环境变量 > config.ini > 默认值"""
        env_val = os.getenv(env_var)
        if env_val is not None:
            try:
                return value_type(env_val)
            except (ValueError, TypeError):
                return default_value

        try:
            if self.config_parser.has_section(section) and self.config_parser.has_option(section, key):
                cfg_val = self.config_parser.get(section, key)
                try:
                    return value_type(cfg_val)
                except (ValueError, TypeError):
                    return default_value
        except (configparser.Error, UnicodeDecodeError):
            pass

        return default_value

    def _parse_list(self, raw_value: str) -> List[str]:
        """将逗号分隔的字符串解析为有序且去重的列表"""
        if not raw_value:
            return []

        items: List[str] = []
        for item in raw_value.split(','):
            candidate = item.strip()
            if candidate and candidate not in items:
                items.append(candidate)
        return items

    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置，优先级：环境变量 > config.ini > 默认值"""
        # 移除硬编码的敏感信息，默认值设为空或安全默认值
        default_key = ""
        default_base = "https://api.openai.com/v1"
        default_fast = ""
        default_deep = ""

        openai_api_key = self._get_config_value('llm', 'openai_api_key', 'OPENAI_API_KEY', default_key)
        openai_base_url = self._get_config_value('llm', 'openai_base_url', 'OPENAI_BASE_URL', default_base)

        # 解析模型列表
        fast_models_raw = self._get_config_value('llm', 'fast_models', 'LLM_FAST_MODELS', default_fast, str)
        deep_models_raw = self._get_config_value('llm', 'deep_models', 'LLM_DEEP_MODELS', default_deep, str)

        fast_models = self._parse_list(fast_models_raw)
        deep_models = self._parse_list(deep_models_raw)

        return {
            'openai_api_key': openai_api_key,
            'openai_base_url': openai_base_url,
            'fast_models': fast_models,
            'deep_models': deep_models,
            'max_tokens': self._get_config_value('llm', 'max_tokens', 'LLM_MAX_TOKENS', 20000, int),
        }

# 全局配置实例
config = Config()
