"""
配置管理模块
支持环境变量 > config.ini > 默认值的优先级机制

环境变量说明：
- DATABASE_URL：数据源数据库连接字符串（必需）
- ANALYSIS_DATABASE_URL：分析结果存储数据库连接字符串（可选）
"""
import os
import configparser
import logging
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# 数据源类型常量（仅支持数据库模式）
DATA_SOURCE_DATABASE = 'database'


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

    def get_data_source(self) -> str:
        """
        获取数据源类型（仅支持数据库模式）
        
        必须配置 DATABASE_URL 或 DB_HOST 环境变量
        """
        # 检测数据库配置是否存在
        db_url = os.getenv('DATABASE_URL')
        db_host = os.getenv('DB_HOST')
        
        if not db_url and not db_host:
            raise ValueError("必须配置 DATABASE_URL 或 DB_HOST 环境变量，本系统仅支持数据库模式")
        
        logger.info("使用数据库作为数据源")
        return DATA_SOURCE_DATABASE
    
    def get_database_config(self) -> Optional[Dict[str, Any]]:
        """获取数据源数据库配置"""
        db_uri = os.getenv('DATABASE_URL')
        
        if db_uri:
            config = self._parse_database_url(db_uri)
            if config:
                return config
        
        # 从单独的环境变量读取
        db_host = os.getenv('DB_HOST')
        if not db_host:
            return None
            
        config = {
            'host': db_host,
            'port': int(os.getenv('DB_PORT', '3306')),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'twitter'),
            'charset': 'utf8mb4',
            'autocommit': True,
        }
        
        if os.getenv('DB_SSL', 'false').lower() == 'true':
            config['ssl'] = {'ssl_mode': 'REQUIRED'}
        
        return config
    
    def _parse_database_url(self, db_uri: str) -> Optional[Dict[str, Any]]:
        """解析数据库连接字符串"""
        pattern = r'mysql://([^:]+):([^@]+)@([^:]+):(\d+)/([^?]+)(\?.*)?'
        match = re.match(pattern, db_uri)
        if match:
            user, password, host, port, database, params = match.groups()
            config = {
                'host': host,
                'port': int(port),
                'user': user,
                'password': password,
                'database': database,
                'charset': 'utf8mb4',
                'autocommit': True,
            }
            
            if params and 'ssl-mode=REQUIRED' in params:
                config['ssl'] = {'ssl_mode': 'REQUIRED'}
            
            return config
        return None
    
    def get_analysis_database_config(self) -> Optional[Dict[str, Any]]:
        """获取分析结果数据库配置（ANALYSIS_DATABASE_URL）"""
        db_uri = os.getenv('ANALYSIS_DATABASE_URL')
        
        if db_uri:
            config = self._parse_database_url(db_uri)
            if config:
                logger.info(f"已配置分析数据库: {config['host']}:{config['port']}/{config['database']}")
                return config
        
        logger.info("未配置分析数据库（ANALYSIS_DATABASE_URL），分析结果将仅保存到本地文件")
        return None

# 全局配置实例
config = Config()
