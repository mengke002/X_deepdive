#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: LLM 深度定性分析脚本
读取 Phase 1 生成的 CSV 清单，调用 LLM API 进行深度分析，生成 JSON/Markdown 报告。
支持 Fast/Deep 双轨模型策略。
"""

import os
import json
import pandas as pd
import time
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from .llm_client import get_llm_client
from .prompts import (
    PROMPT_RELATIONSHIP_INFERENCE,
    PROMPT_USER_STRATEGY,
    PROMPT_VIRAL_CONTENT,
    PROMPT_CONTENT_OPPORTUNITY
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deep_analysis_llm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 任务模型配置 (Task Configuration)
# ==========================================
# 定义每个子任务使用哪类模型 ('fast' or 'deep')
TASK_MODEL_CONFIG = {
    'viral_content': 'deep',       # 爆款拆解：需要深度洞察 -> Deep
    'user_strategy': 'deep',       # 策略画像：需要综合推理 -> Deep
    'relationship': 'fast',        # 关系推理：文本较短，数量大 -> Fast
    'content_opportunity': 'fast'  # 机会挖掘：提取类任务，数量大 -> Fast
}

class DeepAnalysisLLM:
    def __init__(self, input_dir='output', output_dir='output/llm_insights'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.client = get_llm_client()
        os.makedirs(self.output_dir, exist_ok=True)

        # 结果存储
        self.results = {
            'viral_blueprints': [],
            'strategy_dossiers': [],
            'relationship_insights': [],
            'content_ideas': []
        }

    def _call_llm_safe(self, prompt, identifier, task_key):
        """安全的 LLM 调用包装器，根据 task_key 选择模型策略"""
        model_type = TASK_MODEL_CONFIG.get(task_key, 'deep') # 默认 deep

        try:
            if model_type == 'fast':
                response = self.client.call_fast_model(prompt)
            else:
                response = self.client.call_deep_model(prompt)

            if response['success']:
                content = response['content']
                # 尝试清理 markdown 标记
                content = content.replace('```json', '').replace('```', '').strip()
                try:
                    data = json.loads(content)
                    data['id'] = identifier
                    data['model_used'] = response.get('model') # 记录实际使用的模型
                    return data
                except json.JSONDecodeError:
                    logger.error(f"[{task_key}] JSON 解析失败 (ID: {identifier}): {content[:100]}...")
                    return None
            else:
                logger.error(f"[{task_key}] API 调用失败 (ID: {identifier}): {response.get('error')}")
                return None
        except Exception as e:
            logger.error(f"[{task_key}] 未知错误 (ID: {identifier}): {str(e)}")
            return None

    def analyze_viral_content(self, limit=20):
        """任务一：爆款内容逆向工程"""
        task_key = 'viral_content'
        logger.info(f">>> 开始任务：爆款内容逆向工程 (Mode: {TASK_MODEL_CONFIG[task_key]})")

        input_file = f"{self.input_dir}/list_posts_outliers.csv"
        if not os.path.exists(input_file):
            logger.warning(f"未找到输入文件: {input_file}")
            return

        df = pd.read_csv(input_file)
        candidates = df.head(limit).to_dict('records')

        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_item = {}
            for item in candidates:
                input_data = json.dumps({
                    "text": item['text'],
                    "stats": {
                        "views": item.get('view_count'),
                        "likes": item.get('like_count'),
                        "bookmarks": item.get('bookmark_count'),
                        "replies": item.get('reply_count')
                    }
                }, ensure_ascii=False)

                prompt = PROMPT_VIRAL_CONTENT.format(input_data=input_data)
                future = executor.submit(self._call_llm_safe, prompt, item.get('id', str(hash(item['text']))), task_key)
                future_to_item[future] = item

            for future in tqdm(as_completed(future_to_item), total=len(candidates), desc="Analyzing Viral Content"):
                res = future.result()
                if res:
                    original = future_to_item[future]
                    res['original_text'] = original['text']
                    res['author'] = original['author']
                    results.append(res)

        self.results['viral_blueprints'] = results
        self._save_json(results, 'Content_Blueprints.json')

    def analyze_user_strategy(self, limit=20):
        """任务二：用户策略画像"""
        task_key = 'user_strategy'
        logger.info(f">>> 开始任务：用户策略画像 (Mode: {TASK_MODEL_CONFIG[task_key]})")

        input_file = f"{self.input_dir}/list_users_key_players.csv"
        if not os.path.exists(input_file):
            logger.warning(f"未找到输入文件: {input_file}")
            return

        df = pd.read_csv(input_file)
        candidates = df.head(limit).to_dict('records')

        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_item = {}
            for item in candidates:
                input_data = json.dumps({
                    "username": item['username'],
                    "bio": item['bio'],
                    "metrics": {
                        "pagerank_rank": "Top Tier",
                        "professionalism_index": item.get('professionalism_index'),
                        "talkativity_ratio": item.get('talkativity_ratio')
                    }
                }, ensure_ascii=False)

                prompt = PROMPT_USER_STRATEGY.format(input_data=input_data)
                future = executor.submit(self._call_llm_safe, prompt, item['username'], task_key)
                future_to_item[future] = item

            for future in tqdm(as_completed(future_to_item), total=len(candidates), desc="Analyzing User Strategy"):
                res = future.result()
                if res:
                    results.append(res)

        self.results['strategy_dossiers'] = results
        self._save_json(results, 'User_Strategy_Dossiers.json')

    def analyze_relationships(self, limit=20):
        """任务三：关系内涵推理"""
        task_key = 'relationship'
        logger.info(f">>> 开始任务：关系内涵推理 (Mode: {TASK_MODEL_CONFIG[task_key]})")

        input_file = f"{self.input_dir}/list_interactions_strong_ties.csv"
        if not os.path.exists(input_file):
            logger.warning(f"未找到输入文件: {input_file}")
            return

        df = pd.read_csv(input_file)
        candidates = df.head(limit).to_dict('records')

        results = []
        with ThreadPoolExecutor(max_workers=5) as executor: # Fast model 可以开多一点并发
            future_to_item = {}
            for item in candidates:
                try:
                    samples = json.loads(item['interaction_samples'])
                    samples = samples[:3]
                except:
                    samples = ["(无法加载互动样本)"]

                input_data = json.dumps({
                    "user_a": item['user_a'],
                    "user_b": item['user_b'],
                    "interaction_samples": samples
                }, ensure_ascii=False)

                prompt = PROMPT_RELATIONSHIP_INFERENCE.format(input_data=input_data)
                pair_id = f"{item['user_a']}-{item['user_b']}"
                future = executor.submit(self._call_llm_safe, prompt, pair_id, task_key)
                future_to_item[future] = item

            for future in tqdm(as_completed(future_to_item), total=len(candidates), desc="Analyzing Relationships"):
                res = future.result()
                if res:
                    res['user_a'] = future_to_item[future]['user_a']
                    res['user_b'] = future_to_item[future]['user_b']
                    results.append(res)

        self.results['relationship_insights'] = results
        self._save_json(results, 'Community_Insights.json')

    def analyze_content_opportunities(self, limit=20):
        """任务四：内容机会挖掘"""
        task_key = 'content_opportunity'
        logger.info(f">>> 开始任务：内容机会挖掘 (Mode: {TASK_MODEL_CONFIG[task_key]})")

        input_file = f"{self.input_dir}/list_content_opportunities.csv"
        if not os.path.exists(input_file):
            logger.warning(f"未找到输入文件: {input_file}")
            return

        df = pd.read_csv(input_file)
        candidates = df.head(limit).to_dict('records')

        results = []
        with ThreadPoolExecutor(max_workers=5) as executor: # Fast model 可以开多一点并发
            future_to_item = {}
            for item in candidates:
                input_data = json.dumps({
                    "text": item['text'],
                    "type": item.get('opportunity_type'),
                    "stats": {
                        "replies": item.get('reply_count'),
                        "views": item.get('view_count')
                    }
                }, ensure_ascii=False)

                prompt = PROMPT_CONTENT_OPPORTUNITY.format(input_data=input_data)
                future = executor.submit(self._call_llm_safe, prompt, item.get('id', 'unknown'), task_key)
                future_to_item[future] = item

            for future in tqdm(as_completed(future_to_item), total=len(candidates), desc="Analyzing Opportunities"):
                res = future.result()
                if res:
                    res['original_text'] = future_to_item[future]['text']
                    results.append(res)

        self.results['content_ideas'] = results
        self._save_json(results, 'Content_Idea_Bank.json')

    def _save_json(self, data, filename):
        """保存 JSON 结果"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存分析结果: {filepath}")

    def run(self):
        """运行所有任务"""
        logger.info("初始化 LLM 分析流程...")
        if not self.client:
            logger.error("LLM Client 初始化失败，终止任务。")
            return

        self.analyze_viral_content()
        self.analyze_user_strategy()
        self.analyze_relationships()
        self.analyze_content_opportunities()
        logger.info("LLM 分析流程结束。")

if __name__ == "__main__":
    analyzer = DeepAnalysisLLM()
    analyzer.run()
