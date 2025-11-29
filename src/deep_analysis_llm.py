#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: LLM 深度定性分析脚本
读取 Phase 1 生成的 CSV 清单，调用 LLM API 进行深度分析，生成 JSON/Markdown 报告。
支持 Fast/Deep 双轨模型策略。

新增功能（基于 ANALYSIS_MINING_PLAN.md 和 RESULT_DB_RECOMMENDATIONS.md）：
- LLM 输出自动入库
- 商业模式解码
- Thread 结构分析
- 资产四象限内容蓝图
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
from .analysis_db import get_analysis_db_adapter
from .prompts import (
    PROMPT_RELATIONSHIP_INFERENCE,
    PROMPT_USER_STRATEGY,
    PROMPT_VIRAL_CONTENT,
    PROMPT_CONTENT_OPPORTUNITY,
    PROMPT_THREAD_ANALYSIS,
    PROMPT_MONETIZATION_DECODE
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
    'content_opportunity': 'fast', # 机会挖掘：提取类任务，数量大 -> Fast
    'thread_analysis': 'deep',     # Thread 结构分析：需要深度理解 -> Deep
    'monetization_decode': 'deep'  # 商业模式解码：需要综合推理 -> Deep
}

class DeepAnalysisLLM:
    def __init__(self, input_dir='output', output_dir='output/llm_insights'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.client = get_llm_client()
        self.analysis_db = get_analysis_db_adapter()
        self.session_id = None
        self.source_session_id = None  # 用于从 DB 读取候选数据的会话 ID
        os.makedirs(self.output_dir, exist_ok=True)

        # 结果存储
        self.results = {
            'viral_blueprints': [],
            'strategy_dossiers': [],
            'relationship_insights': [],
            'content_ideas': [],
            'thread_blueprints': [],
            'monetization_insights': []
        }

    def _load_candidates_with_fallback(self, csv_filename: str, db_fallback_method: str, limit: int = 50) -> list:
        """
        智能候选数据加载器：优先从 CSV 文件读取，不存在时从数据库回退
        
        Args:
            csv_filename: CSV 文件名（相对于 input_dir）
            db_fallback_method: 数据库回退方法名（analysis_db 的方法）
            limit: 最大返回数量
        
        Returns:
            候选数据列表
        """
        csv_path = os.path.join(self.input_dir, csv_filename)
        
        # 优先尝试从 CSV 文件读取
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                logger.info(f"从 CSV 文件加载候选数据: {csv_path} ({len(df)} 条)")
                return df.head(limit).to_dict('records')
            except Exception as e:
                logger.warning(f"读取 CSV 文件失败 ({csv_path}): {e}")
        
        # 回退到数据库查询
        if self.analysis_db and hasattr(self.analysis_db, db_fallback_method):
            logger.info(f"CSV 文件不存在，尝试从数据库加载候选数据 ({db_fallback_method})...")
            try:
                db_method = getattr(self.analysis_db, db_fallback_method)
                # 传入 source_session_id 以获取特定会话的数据
                candidates = db_method(limit=limit, session_id=self.source_session_id)
                if candidates:
                    logger.info(f"从数据库加载候选数据成功: {len(candidates)} 条")
                    return candidates
                else:
                    logger.warning(f"数据库中无可用候选数据")
            except Exception as e:
                logger.error(f"从数据库加载候选数据失败: {e}")
        
        logger.warning(f"无法加载候选数据: {csv_filename}")
        return []

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
                    
                    # 保存 LLM 输出到数据库 (用于审计)
                    if self.analysis_db and self.session_id:
                        self.analysis_db.save_llm_output(
                            session_id=self.session_id,
                            task_type=task_key,
                            target_id=str(identifier),
                            model_used=response.get('model', ''),
                            raw_output={'content': response['content']},
                            parsed_output=data
                        )
                    
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

        # 使用智能加载器：优先 CSV，回退数据库
        candidates = self._load_candidates_with_fallback(
            csv_filename='list_posts_outliers.csv',
            db_fallback_method='get_content_outliers_for_llm',
            limit=limit
        )
        
        if not candidates:
            logger.warning("无法获取爆款内容候选数据，跳过此任务")
            return

        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_item = {}
            for item in candidates:
                input_data = json.dumps({
                    "text": item.get('text', ''),
                    "stats": {
                        "views": item.get('view_count'),
                        "likes": item.get('like_count'),
                        "bookmarks": item.get('bookmark_count'),
                        "replies": item.get('reply_count')
                    }
                }, ensure_ascii=False)

                prompt = PROMPT_VIRAL_CONTENT.format(input_data=input_data)
                future = executor.submit(self._call_llm_safe, prompt, item.get('id', str(hash(item.get('text', '')))), task_key)
                future_to_item[future] = item

            for future in tqdm(as_completed(future_to_item), total=len(candidates), desc="Analyzing Viral Content"):
                res = future.result()
                if res:
                    original = future_to_item[future]
                    res['original_text'] = original.get('text', '')
                    res['author'] = original.get('author', '')
                    results.append(res)

        self.results['viral_blueprints'] = results
        self._save_json(results, 'viral_deconstruction.json')

    def analyze_user_strategy(self, limit=20):
        """任务二：用户策略画像"""
        task_key = 'user_strategy'
        logger.info(f">>> 开始任务：用户策略画像 (Mode: {TASK_MODEL_CONFIG[task_key]})")

        # 使用智能加载器：优先 CSV，回退数据库
        candidates = self._load_candidates_with_fallback(
            csv_filename='list_users_key_players.csv',
            db_fallback_method='get_key_users_for_llm',
            limit=limit
        )
        
        if not candidates:
            logger.warning("无法获取关键用户候选数据，跳过此任务")
            return

        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_item = {}
            for item in candidates:
                input_data = json.dumps({
                    "username": item.get('username', ''),
                    "bio": item.get('bio', ''),
                    "metrics": {
                        "pagerank_rank": "Top Tier",
                        "professionalism_index": item.get('professionalism_index'),
                        "talkativity_ratio": item.get('talkativity_ratio')
                    }
                }, ensure_ascii=False)

                prompt = PROMPT_USER_STRATEGY.format(input_data=input_data)
                future = executor.submit(self._call_llm_safe, prompt, item.get('username', ''), task_key)
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

        # 使用智能加载器：优先 CSV，回退数据库
        candidates = self._load_candidates_with_fallback(
            csv_filename='list_interactions_strong_ties.csv',
            db_fallback_method='get_strong_ties_for_llm',
            limit=limit
        )
        
        if not candidates:
            logger.warning("无法获取强互惠关系候选数据，跳过此任务")
            return

        results = []
        with ThreadPoolExecutor(max_workers=5) as executor: # Fast model 可以开多一点并发
            future_to_item = {}
            for item in candidates:
                try:
                    samples = item.get('interaction_samples', '[]')
                    if isinstance(samples, str):
                        samples = json.loads(samples)
                    samples = samples[:3] if samples else []
                except Exception as e:
                    logger.warning(f"解析互动样本失败: {e}")
                    samples = ["(无法加载互动样本)"]
                
                # 调试日志：检查样本数据
                if not samples or (len(samples) == 1 and samples[0] == ''):
                    logger.debug(f"警告: 用户对 {item.get('user_a')}-{item.get('user_b')} 无有效互动样本")

                input_data = json.dumps({
                    "user_a": item.get('user_a', ''),
                    "user_b": item.get('user_b', ''),
                    "interaction_samples": samples
                }, ensure_ascii=False)

                prompt = PROMPT_RELATIONSHIP_INFERENCE.format(input_data=input_data)
                pair_id = f"{item.get('user_a', '')}-{item.get('user_b', '')}"
                future = executor.submit(self._call_llm_safe, prompt, pair_id, task_key)
                future_to_item[future] = item

            for future in tqdm(as_completed(future_to_item), total=len(candidates), desc="Analyzing Relationships"):
                res = future.result()
                if res:
                    res['user_a'] = future_to_item[future].get('user_a', '')
                    res['user_b'] = future_to_item[future].get('user_b', '')
                    results.append(res)

        self.results['relationship_insights'] = results
        self._save_json(results, 'relationship_insight.json')

    def analyze_content_opportunities(self, limit=20):
        """任务四：内容机会挖掘"""
        task_key = 'content_opportunity'
        logger.info(f">>> 开始任务：内容机会挖掘 (Mode: {TASK_MODEL_CONFIG[task_key]})")

        # 使用智能加载器：优先 CSV，回退数据库
        candidates = self._load_candidates_with_fallback(
            csv_filename='list_content_opportunities.csv',
            db_fallback_method='get_content_opportunities_for_llm',
            limit=limit
        )
        
        if not candidates:
            logger.warning("无法获取内容机会候选数据，跳过此任务")
            return

        results = []
        with ThreadPoolExecutor(max_workers=5) as executor: # Fast model 可以开多一点并发
            future_to_item = {}
            for item in candidates:
                input_data = json.dumps({
                    "text": item.get('text', ''),
                    "type": item.get('opportunity_type', ''),
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
                    res['original_text'] = future_to_item[future].get('text', '')
                    results.append(res)

        self.results['content_ideas'] = results
        self._save_json(results, 'Content_Idea_Bank.json')

    def _save_json(self, data, filename):
        """保存 JSON 结果"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存分析结果: {filepath}")

    def analyze_threads(self, limit=15):
        """任务五：Thread 结构分析（新增）"""
        task_key = 'thread_analysis'
        logger.info(f">>> 开始任务：Thread 结构分析 (Mode: {TASK_MODEL_CONFIG[task_key]})")

        # 使用智能加载器：优先 CSV，回退数据库
        candidates = self._load_candidates_with_fallback(
            csv_filename='list_threads_viral.csv',
            db_fallback_method='get_viral_threads_for_llm',
            limit=limit
        )
        
        if not candidates:
            logger.warning("无法获取 Thread 候选数据，跳过此任务")
            return

        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_item = {}
            for item in candidates:
                input_data = json.dumps({
                    "conversation_id": item.get('conversation_id', ''),
                    "thread_length": item.get('thread_length'),
                    "retention_rate": item.get('retention_rate'),
                    "author": item.get('author', '')
                }, ensure_ascii=False)

                prompt = PROMPT_THREAD_ANALYSIS.format(input_data=input_data)
                future = executor.submit(self._call_llm_safe, prompt, item.get('conversation_id', 'unknown'), task_key)
                future_to_item[future] = item

            for future in tqdm(as_completed(future_to_item), total=len(candidates), desc="Analyzing Threads"):
                res = future.result()
                if res:
                    original = future_to_item[future]
                    res['conversation_id'] = original.get('conversation_id', '')
                    res['thread_length'] = original.get('thread_length')
                    results.append(res)

        self.results['thread_blueprints'] = results
        self._save_json(results, 'Thread_Blueprints.json')

    def analyze_monetization(self, limit=15):
        """任务六：商业模式解码（新增）"""
        task_key = 'monetization_decode'
        logger.info(f">>> 开始任务：商业模式解码 (Mode: {TASK_MODEL_CONFIG[task_key]})")

        # 读取带有商业信号的内容
        input_file = f"{self.input_dir}/stats_funnel_signals.csv"
        
        # 优先尝试从 CSV 文件读取
        user_signals = None
        if os.path.exists(input_file):
            try:
                df = pd.read_csv(input_file)
                # 按用户聚合，分析每个用户的变现模式
                user_signals = df.groupby('author').agg({
                    'funnel_signal': lambda x: list(x),
                    'view_count': 'sum',
                    'like_count': 'sum'
                }).reset_index()
                logger.info(f"从 CSV 文件加载商业信号数据: {input_file} ({len(user_signals)} 用户)")
            except Exception as e:
                logger.warning(f"读取 CSV 文件失败 ({input_file}): {e}")
        
        # 回退到数据库查询
        if user_signals is None or user_signals.empty:
            if self.analysis_db:
                logger.info("CSV 文件不存在，尝试从数据库加载商业信号数据...")
                try:
                    raw_signals = self.analysis_db.get_funnel_signals_for_llm(limit=200, session_id=self.source_session_id)
                    if raw_signals:
                        df = pd.DataFrame(raw_signals)
                        user_signals = df.groupby('author').agg({
                            'funnel_signal': lambda x: list(x),
                            'view_count': 'sum',
                            'like_count': 'sum'
                        }).reset_index()
                        logger.info(f"从数据库加载商业信号数据成功: {len(user_signals)} 用户")
                except Exception as e:
                    logger.error(f"从数据库加载商业信号数据失败: {e}")
        
        if user_signals is None or user_signals.empty:
            logger.warning("无法获取商业信号候选数据，跳过此任务")
            return

        candidates = user_signals.head(limit).to_dict('records')

        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_item = {}
            for item in candidates:
                input_data = json.dumps({
                    "username": item.get('author', ''),
                    "funnel_signals": item.get('funnel_signal', []),
                    "total_views": item.get('view_count'),
                    "total_likes": item.get('like_count')
                }, ensure_ascii=False)

                prompt = PROMPT_MONETIZATION_DECODE.format(input_data=input_data)
                future = executor.submit(self._call_llm_safe, prompt, item.get('author', 'unknown'), task_key)
                future_to_item[future] = item

            for future in tqdm(as_completed(future_to_item), total=len(candidates), desc="Analyzing Monetization"):
                res = future.result()
                if res:
                    results.append(res)

        self.results['monetization_insights'] = results
        self._save_json(results, 'monetization_analysis.json')

    def _save_results_to_db(self):
        """保存 LLM 分析结果到数据库"""
        if not self.analysis_db or not self.session_id:
            logger.info("分析数据库未配置，跳过数据库保存")
            return

        logger.info("保存 LLM 分析结果到数据库...")

        try:
            # 保存用户策略画像
            if self.results['strategy_dossiers']:
                self.analysis_db.save_user_strategy_dossiers(
                    self.session_id, 
                    self.results['strategy_dossiers']
                )

            # 保存爆款内容蓝图
            if self.results['viral_blueprints']:
                self.analysis_db.save_content_blueprints(
                    self.session_id,
                    self.results['viral_blueprints']
                )

            # 保存内容创意库
            if self.results['content_ideas']:
                self.analysis_db.save_content_idea_bank(
                    self.session_id,
                    self.results['content_ideas']
                )

            logger.info("LLM 分析结果已保存到数据库")

        except Exception as e:
            logger.error(f"保存到数据库时出错: {e}")

    def run(self, source_session_id: str = None):
        """运行所有任务
        
        Args:
            source_session_id: 用于从数据库读取候选数据的源会话 ID（可选）
                             如果不指定，且 CSV 文件不存在时会自动使用最近完成的会话
        """
        logger.info("初始化 LLM 分析流程...")
        if not self.client:
            logger.error("LLM Client 初始化失败，终止任务。")
            return

        # 设置源会话 ID（用于数据库回退查询）
        if source_session_id:
            self.source_session_id = source_session_id
        elif self.analysis_db:
            # 自动获取最近完成的会话 ID
            self.source_session_id = self.analysis_db.get_latest_completed_session_id()
            if self.source_session_id:
                logger.info(f"自动检测到最近完成的分析会话: {self.source_session_id}")

        # 创建分析会话
        if self.analysis_db:
            self.session_id = self.analysis_db.create_session({
                'type': 'llm_analysis',
                'output_dir': self.output_dir,
                'source_session_id': self.source_session_id
            })
            logger.info(f"创建分析会话: {self.session_id}")

        # 核心分析任务
        self.analyze_viral_content()
        self.analyze_user_strategy()
        self.analyze_relationships()
        self.analyze_content_opportunities()

        # 新增分析任务
        self.analyze_threads()
        self.analyze_monetization()

        # 保存结果到数据库
        self._save_results_to_db()

        # 完成会话
        if self.analysis_db and self.session_id:
            stats = {
                'viral_blueprints': len(self.results['viral_blueprints']),
                'strategy_dossiers': len(self.results['strategy_dossiers']),
                'relationship_insights': len(self.results['relationship_insights']),
                'content_ideas': len(self.results['content_ideas']),
                'thread_blueprints': len(self.results['thread_blueprints']),
                'monetization_insights': len(self.results['monetization_insights'])
            }
            self.analysis_db.complete_session(self.session_id, stats)

        logger.info("LLM 分析流程结束。")

def main():
    """命令行入口，支持细粒度任务拆分"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Deep Analysis LLM - LLM深度定性分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 完整运行所有任务
  python -m src.deep_analysis_llm
  
  # 只运行 Fast 模型任务（节省成本）
  python -m src.deep_analysis_llm --tasks relationship,content_opportunity
  
  # 只运行 Deep 模型任务
  python -m src.deep_analysis_llm --tasks viral,user_strategy
  
  # 运行特定任务
  python -m src.deep_analysis_llm --tasks viral,thread --limit 10
  
  # 指定从特定会话读取候选数据（当 CSV 不存在时）
  python -m src.deep_analysis_llm --source-session 20241129_120000
  
可用任务:
  viral              - 爆款内容逆向工程 (Deep, ~4000 tokens/条)
  user_strategy      - 用户策略画像 (Deep, ~6000 tokens/条)
  relationship       - 关系内涵推理 (Fast, ~1500 tokens/条)
  content_opportunity- 内容机会挖掘 (Fast, ~1500 tokens/条)
  thread             - Thread 结构分析 (Deep, ~3000 tokens/条)
  monetization       - 商业模式解码 (Deep, ~3000 tokens/条)
  fast               - 所有 Fast 任务 (relationship + content_opportunity)
  deep               - 所有 Deep 任务 (viral + user_strategy + thread + monetization)
  all                - 全部任务 (默认)

数据源说明:
  候选数据优先从 CSV 文件 (output/*.csv) 读取，如果 CSV 不存在，
  会自动从 Analysis DB 读取最近完成的分析会话数据作为备选。
  可以通过 --source-session 参数指定特定的源会话 ID。
        """
    )
    
    parser.add_argument(
        '--tasks',
        type=str,
        default='all',
        help='要执行的任务列表，逗号分隔'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='每个任务处理的最大数量 (默认: 20)'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='output',
        help='输入目录 (默认: output)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/llm_insights',
        help='输出目录 (默认: output/llm_insights)'
    )
    
    parser.add_argument(
        '--source-session',
        type=str,
        default=None,
        help='源会话 ID，用于从数据库读取候选数据（当 CSV 不存在时）。如不指定，自动使用最近完成的会话'
    )
    
    args = parser.parse_args()
    
    # 解析任务列表
    task_list = [t.strip() for t in args.tasks.split(',')]
    
    # 展开任务组
    if 'all' in task_list:
        task_list = ['viral', 'user_strategy', 'relationship', 'content_opportunity', 'thread', 'monetization']
    elif 'fast' in task_list:
        task_list = [t for t in task_list if t != 'fast'] + ['relationship', 'content_opportunity']
    elif 'deep' in task_list:
        task_list = [t for t in task_list if t != 'deep'] + ['viral', 'user_strategy', 'thread', 'monetization']
    
    # 去重
    task_list = list(dict.fromkeys(task_list))
    
    # 验证任务名称
    valid_tasks = {'viral', 'user_strategy', 'relationship', 'content_opportunity', 'thread', 'monetization'}
    invalid_tasks = set(task_list) - valid_tasks
    if invalid_tasks:
        print(f"❌ 错误: 无效的任务名称: {', '.join(invalid_tasks)}")
        print(f"✅ 有效任务: {', '.join(valid_tasks)}")
        print(f"✅ 任务组: fast, deep, all")
        return
    
    # 统计 Token 消耗预估
    token_estimates = {
        'viral': 4000,
        'user_strategy': 6000,
        'relationship': 1500,
        'content_opportunity': 1500,
        'thread': 3000,
        'monetization': 3000
    }
    
    estimated_tokens = sum(token_estimates.get(t, 0) * args.limit for t in task_list)
    estimated_cost = estimated_tokens / 1_000_000 * 2  # 假设 $2/M tokens
    
    print("\n" + "=" * 60)
    print("LLM 深度分析任务配置")
    print("=" * 60)
    print(f"执行任务: {', '.join(task_list)}")
    print(f"每任务限制: {args.limit} 条")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    if args.source_session:
        print(f"源会话 ID: {args.source_session}")
    else:
        print(f"源会话 ID: 自动检测（CSV 优先）")
    print(f"预估 Token: ~{estimated_tokens:,} ({estimated_cost:.2f} USD)")
    print("=" * 60)
    
    # 创建分析器
    analyzer = DeepAnalysisLLM(input_dir=args.input_dir, output_dir=args.output_dir)
    
    if not analyzer.client:
        logger.error("LLM Client 初始化失败，终止任务。")
        return
    
    # 设置源会话 ID
    if args.source_session:
        analyzer.source_session_id = args.source_session
    elif analyzer.analysis_db:
        # 自动获取最近完成的会话 ID
        analyzer.source_session_id = analyzer.analysis_db.get_latest_completed_session_id()
        if analyzer.source_session_id:
            logger.info(f"自动检测到最近完成的分析会话: {analyzer.source_session_id}")
    
    # 创建分析会话
    if analyzer.analysis_db:
        analyzer.session_id = analyzer.analysis_db.create_session({
            'type': 'llm_analysis',
            'output_dir': analyzer.output_dir,
            'tasks': task_list,
            'limit': args.limit,
            'source_session_id': analyzer.source_session_id
        })
        logger.info(f"创建分析会话: {analyzer.session_id}")
    
    # 执行选定的任务
    if 'viral' in task_list:
        logger.info(">>> 执行任务: 爆款内容逆向工程")
        analyzer.analyze_viral_content(limit=args.limit)
    
    if 'user_strategy' in task_list:
        logger.info(">>> 执行任务: 用户策略画像")
        analyzer.analyze_user_strategy(limit=args.limit)
    
    if 'relationship' in task_list:
        logger.info(">>> 执行任务: 关系内涵推理")
        analyzer.analyze_relationships(limit=args.limit)
    
    if 'content_opportunity' in task_list:
        logger.info(">>> 执行任务: 内容机会挖掘")
        analyzer.analyze_content_opportunities(limit=args.limit)
    
    if 'thread' in task_list:
        logger.info(">>> 执行任务: Thread 结构分析")
        analyzer.analyze_threads(limit=args.limit)
    
    if 'monetization' in task_list:
        logger.info(">>> 执行任务: 商业模式解码")
        analyzer.analyze_monetization(limit=args.limit)
    
    # 保存结果到数据库
    analyzer._save_results_to_db()
    
    # 完成会话
    if analyzer.analysis_db and analyzer.session_id:
        stats = {
            'viral_blueprints': len(analyzer.results['viral_blueprints']),
            'strategy_dossiers': len(analyzer.results['strategy_dossiers']),
            'relationship_insights': len(analyzer.results['relationship_insights']),
            'content_ideas': len(analyzer.results['content_ideas']),
            'thread_blueprints': len(analyzer.results['thread_blueprints']),
            'monetization_insights': len(analyzer.results['monetization_insights'])
        }
        analyzer.analysis_db.complete_session(analyzer.session_id, stats)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ LLM 分析流程结束")
    logger.info("=" * 60)
    logger.info(f"输出目录: {analyzer.output_dir}")
    logger.info(f"执行的任务: {', '.join(task_list)}")

if __name__ == "__main__":
    main()
