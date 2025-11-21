"""
LLM客户端模块
支持OpenAI compatible接口的streaming实现，支持 Fast/Deep 双轨模型策略
"""
import logging
import time
from typing import Dict, Any, List, Optional
from openai import OpenAI

from .config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMClient:
    """统一的LLM客户端，支持 Fast/Deep 两种模式的托底调用"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        llm_config = config.get_llm_config()
        self.api_key = llm_config.get('openai_api_key')
        self.base_url = llm_config.get('openai_base_url')

        self.fast_models = llm_config.get('fast_models', [])
        self.deep_models = llm_config.get('deep_models', [])

        self.max_tokens = llm_config.get('max_tokens', 20000)

        if not self.api_key:
            raise ValueError("未找到OPENAI_API_KEY配置")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.logger.info(f"LLM客户端初始化成功")
        self.logger.info(f"Fast Models: {self.fast_models}")
        self.logger.info(f"Deep Models: {self.deep_models}")

    def call_fast_model(self, prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        """调用快速模型列表（用于高频、短上下文任务）"""
        return self._call_model_list(prompt, self.fast_models, temperature, "Fast")

    def call_deep_model(self, prompt: str, temperature: float = 0.5) -> Dict[str, Any]:
        """调用深度模型列表（用于深度思考、长上下文任务）"""
        return self._call_model_list(prompt, self.deep_models, temperature, "Deep")

    def _call_model_list(self, prompt: str, model_list: List[str], temperature: float, mode_name: str) -> Dict[str, Any]:
        """
        按顺序遍历模型列表进行调用，一旦成功即返回
        """
        if not model_list:
             raise ValueError(f"未配置任何可用的 {mode_name} Models")

        last_response: Dict[str, Any] = {
            'success': False,
            'error': f'所有 {mode_name} 模型均调用失败'
        }

        for index, model_name in enumerate(model_list):
            self.logger.info(f"[{mode_name}] 尝试使用模型: {model_name}")
            # 每个模型最多重试2次，避免卡太久
            result = self._make_request(prompt, model_name, temperature, max_retries=2)

            if result.get('success'):
                return result

            last_response = result
            if index < len(model_list) - 1:
                fallback_target = model_list[index + 1]
                self.logger.warning(
                    f"模型 {model_name} 失败，将托底回退至 {fallback_target}"
                )

        return last_response

    def _make_request(self, prompt: str, model_name: str, temperature: float, max_retries: int = 2) -> Dict[str, Any]:
        """
        执行具体的LLM请求
        """
        for attempt in range(max_retries):
            try:
                self.logger.info(f"调用LLM: {model_name} (尝试 {attempt + 1}/{max_retries})")

                # 创建streaming请求
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {'role': 'system', 'content': '你是一个专业的社交网络分析师。请只输出JSON格式的结果，不要包含markdown代码块标记。'},
                        {'role': 'user', 'content': prompt}
                    ],
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )

                # 收集所有streaming内容
                full_content = ""

                for chunk in response:
                    if not hasattr(chunk, 'choices') or not chunk.choices:
                        continue

                    if len(chunk.choices) == 0:
                        continue

                    delta = chunk.choices[0].delta
                    content_chunk = getattr(delta, 'content', None)

                    if content_chunk:
                        full_content += content_chunk

                if not full_content.strip():
                    raise ValueError("LLM返回空响应")

                return {
                    'success': True,
                    'content': full_content.strip(),
                    'model': model_name,
                    'attempt': attempt + 1
                }

            except Exception as e:
                error_msg = f"LLM调用失败 ({model_name}): {str(e)}"
                self.logger.error(error_msg)

                if attempt < max_retries - 1:
                    time.sleep(1) # 快速重试
                else:
                    return {
                        'success': False,
                        'error': error_msg,
                        'model': model_name
                    }
        return {'success': False, 'error': 'Unknown error'}

def get_llm_client() -> Optional[LLMClient]:
    """获取LLM客户端实例"""
    try:
        return LLMClient()
    except Exception as e:
        logger.error(f"创建LLM客户端失败: {e}")
        return None
