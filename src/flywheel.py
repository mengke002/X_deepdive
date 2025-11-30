#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复利系统模块 (Flywheel System)
基于 ANALYSIS_MINING_PLAN.md 阶段三设计

核心功能：
1. 信息时差与截流 (Smart Reply & Arbitrage)
   - 爆款早期检测
   - 智能回复草稿生成
2. 自动化辅助创作 (Content Co-pilot)
   - 知识库复用
   - 写作辅助
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from collections import defaultdict

import pandas as pd
import numpy as np

from .llm_client import get_llm_client
from .analysis_db import get_analysis_db_adapter
from .prompts import PROMPT_SMART_REPLY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VelocityDetector:
    """
    爆款早期检测器
    通过分析推文的互动增速来识别潜在爆款
    """
    
    def __init__(self, velocity_threshold_percentile: float = 95):
        """
        Args:
            velocity_threshold_percentile: 增速阈值的百分位数
        """
        self.velocity_threshold_percentile = velocity_threshold_percentile
        self.historical_velocities = []  # 存储历史增速数据用于计算阈值
        
    def calculate_velocity(self, 
                          current_stats: Dict[str, int], 
                          previous_stats: Dict[str, int],
                          time_delta_minutes: float) -> Dict[str, float]:
        """
        计算互动增速
        
        Args:
            current_stats: 当前统计数据 {likes, replies, retweets, views}
            previous_stats: 上次统计数据
            time_delta_minutes: 时间差（分钟）
            
        Returns:
            各指标的每分钟增速
        """
        if time_delta_minutes <= 0:
            return {}
            
        velocities = {}
        for key in ['likes', 'replies', 'retweets', 'views']:
            current = current_stats.get(key, 0)
            previous = previous_stats.get(key, 0)
            delta = current - previous
            velocity = delta / time_delta_minutes
            velocities[f'{key}_velocity'] = velocity
            
        # 计算综合增速评分
        # 权重：replies > retweets > likes > views
        weights = {'likes': 1.0, 'replies': 3.0, 'retweets': 2.0, 'views': 0.1}
        weighted_velocity = sum(
            velocities.get(f'{k}_velocity', 0) * w 
            for k, w in weights.items()
        )
        velocities['weighted_velocity'] = weighted_velocity
        
        return velocities
    
    def is_velocity_spike(self, velocity: float) -> bool:
        """
        判断是否是增速异常（潜在爆款）
        """
        if not self.historical_velocities:
            return velocity > 0  # 没有历史数据时，任何正增速都标记
            
        threshold = np.percentile(self.historical_velocities, self.velocity_threshold_percentile)
        return velocity > threshold
    
    def update_historical(self, velocity: float):
        """更新历史增速数据"""
        self.historical_velocities.append(velocity)
        # 保持最近1000条记录
        if len(self.historical_velocities) > 1000:
            self.historical_velocities = self.historical_velocities[-1000:]


class SmartReplyGenerator:
    """
    智能截流回复生成器
    当检测到潜在爆款时，生成高质量回复草稿
    """
    
    def __init__(self):
        self.llm_client = get_llm_client()
        self.analysis_db = get_analysis_db_adapter()
        
    def generate_reply_draft(self, 
                            tweet_text: str, 
                            tweet_author: str,
                            tweet_stats: Dict[str, int],
                            author_context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        生成智能回复草稿
        
        Args:
            tweet_text: 原推文内容
            tweet_author: 原推文作者
            tweet_stats: 推文统计数据
            author_context: 作者背景信息（可选）
            
        Returns:
            回复草稿及相关元数据
        """
        if not self.llm_client:
            logger.error("LLM Client 未初始化")
            return None
            
        # 构建输入数据
        input_data = json.dumps({
            "original_tweet": {
                "text": tweet_text[:500],  # 限制长度
                "author": tweet_author,
                "stats": tweet_stats
            },
            "author_context": author_context or "未知",
            "goal": "在这条正在快速传播的推文下留下高价值评论，建立专业形象"
        }, ensure_ascii=False)
        
        prompt = PROMPT_SMART_REPLY.format(input_data=input_data)
        
        try:
            response = self.llm_client.call_fast_model(prompt)
            
            if response['success']:
                content = response['content']
                content = content.replace('```json', '').replace('```', '').strip()
                
                try:
                    result = json.loads(content)
                    result['model_used'] = response.get('model')
                    result['generated_at'] = datetime.now().isoformat()
                    return result
                except json.JSONDecodeError:
                    logger.error(f"JSON 解析失败: {content[:100]}...")
                    return None
            else:
                logger.error(f"LLM 调用失败: {response.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"生成回复草稿时出错: {e}")
            return None
    
    def save_candidate(self, 
                      tweet_id: str,
                      author_username: str,
                      detected_signal: str,
                      tweet_text: str,
                      reply_result: Dict[str, Any]):
        """保存截流候选到数据库"""
        if not self.analysis_db:
            logger.warning("分析数据库未配置，跳过保存")
            return
            
        candidate = {
            'target_tweet_id': tweet_id,
            'author_username': author_username,
            'detected_signal': detected_signal,
            'tweet_text': tweet_text,
            'draft_reply_text': reply_result.get('reply_draft', '')
        }
        
        self.analysis_db.save_smart_reply_candidates([candidate])
        logger.info(f"已保存截流候选: {tweet_id}")


class ContentCopilot:
    """
    内容辅助创作系统
    基于知识库复用和历史高价值内容辅助新内容创作
    """
    
    def __init__(self, knowledge_dir: str = 'output/llm_insights'):
        self.knowledge_dir = knowledge_dir
        self.content_blueprints = []
        self.thread_templates = []
        self._load_knowledge_base()
        
    def _load_knowledge_base(self):
        """加载知识库"""
        # 加载爆款内容蓝图
        blueprints_file = os.path.join(self.knowledge_dir, 'Content_Blueprints.json')
        if os.path.exists(blueprints_file):
            with open(blueprints_file, 'r', encoding='utf-8') as f:
                self.content_blueprints = json.load(f)
            logger.info(f"加载了 {len(self.content_blueprints)} 个内容蓝图")
        
        # 加载 Thread 模板
        threads_file = os.path.join(self.knowledge_dir, 'Thread_Blueprints.json')
        if os.path.exists(threads_file):
            with open(threads_file, 'r', encoding='utf-8') as f:
                self.thread_templates = json.load(f)
            logger.info(f"加载了 {len(self.thread_templates)} 个 Thread 模板")
    
    def find_similar_content(self, 
                            topic: str, 
                            content_type: str = 'any',
                            top_k: int = 5) -> List[Dict]:
        """
        查找相似的高价值内容作为参考
        
        Args:
            topic: 主题关键词
            content_type: 内容类型 ('thread', 'single', 'any')
            top_k: 返回数量
            
        Returns:
            相关内容列表
        """
        results = []
        
        # 简单的关键词匹配（后续可以升级为向量检索）
        topic_lower = topic.lower()
        
        for bp in self.content_blueprints:
            original_text = bp.get('original_text', '').lower()
            why_viral = bp.get('why_viral', '').lower()
            
            # 计算相关性分数
            score = 0
            if topic_lower in original_text:
                score += 2
            if topic_lower in why_viral:
                score += 1
                
            if score > 0:
                results.append({
                    'type': 'blueprint',
                    'score': score,
                    'content': bp
                })
        
        # 排序并返回 top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def get_writing_suggestions(self, topic: str) -> Dict[str, Any]:
        """
        获取写作建议
        
        Args:
            topic: 要写的主题
            
        Returns:
            写作建议和参考资料
        """
        similar_content = self.find_similar_content(topic)
        
        suggestions = {
            'topic': topic,
            'similar_successful_content': similar_content,
            'hook_suggestions': [],
            'structure_suggestions': [],
            'cta_suggestions': []
        }
        
        # 从相似内容中提取建议
        for item in similar_content:
            bp = item.get('content', {})
            
            hook = bp.get('hook_style', {})
            if hook:
                suggestions['hook_suggestions'].append({
                    'type': hook.get('type'),
                    'example': hook.get('text')
                })
            
            body = bp.get('body_structure')
            if body:
                suggestions['structure_suggestions'].append(body)
                
            cta = bp.get('call_to_action', {})
            if cta:
                suggestions['cta_suggestions'].append({
                    'type': cta.get('type'),
                    'example': cta.get('text')
                })
        
        # 去重
        suggestions['structure_suggestions'] = list(set(suggestions['structure_suggestions']))
        
        return suggestions
    
    def get_thread_template(self, style: str = 'any') -> Optional[Dict]:
        """
        获取 Thread 写作模板
        
        Args:
            style: 模板风格 ('educational', 'story', 'listicle', 'any')
            
        Returns:
            Thread 模板
        """
        if not self.thread_templates:
            return None
            
        if style == 'any':
            # 随机返回一个
            import random
            return random.choice(self.thread_templates)
        
        # 按风格筛选
        for template in self.thread_templates:
            pattern = template.get('body_structure', {}).get('pattern', '')
            if style.lower() in pattern.lower():
                return template
        
        return self.thread_templates[0] if self.thread_templates else None


class FlywheelSystem:
    """
    复利系统主类
    整合爆款检测、智能回复、内容辅助等功能
    """
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        self.velocity_detector = VelocityDetector()
        self.reply_generator = SmartReplyGenerator()
        self.content_copilot = ContentCopilot(
            knowledge_dir=os.path.join(output_dir, 'llm_insights')
        )
        self.analysis_db = get_analysis_db_adapter()
        
        # 追踪状态
        self.tracked_tweets = {}  # tweet_id -> last_stats
        
    def process_tweet_update(self, 
                            tweet_id: str,
                            tweet_text: str,
                            author_username: str,
                            current_stats: Dict[str, int],
                            tweet_age_minutes: float) -> Optional[Dict]:
        """
        处理推文更新，检测是否需要截流
        
        Args:
            tweet_id: 推文 ID
            tweet_text: 推文内容
            author_username: 作者用户名
            current_stats: 当前统计数据
            tweet_age_minutes: 推文发布后的分钟数
            
        Returns:
            如果触发截流，返回回复草稿；否则返回 None
        """
        # 获取上次统计数据
        previous = self.tracked_tweets.get(tweet_id, {
            'stats': {'likes': 0, 'replies': 0, 'retweets': 0, 'views': 0},
            'timestamp': datetime.now()
        })
        
        previous_stats = previous['stats']
        time_delta = (datetime.now() - previous['timestamp']).total_seconds() / 60
        
        if time_delta < 1:
            return None  # 更新间隔太短
        
        # 计算增速
        velocities = self.velocity_detector.calculate_velocity(
            current_stats, previous_stats, time_delta
        )
        
        weighted_velocity = velocities.get('weighted_velocity', 0)
        
        # 更新追踪状态
        self.tracked_tweets[tweet_id] = {
            'stats': current_stats,
            'timestamp': datetime.now()
        }
        
        # 更新历史增速
        self.velocity_detector.update_historical(weighted_velocity)
        
        # 检测是否是爆款
        if self.velocity_detector.is_velocity_spike(weighted_velocity):
            logger.info(f"检测到潜在爆款: {tweet_id}, 增速: {weighted_velocity:.2f}")
            
            # 生成回复草稿
            reply_result = self.reply_generator.generate_reply_draft(
                tweet_text=tweet_text,
                tweet_author=author_username,
                tweet_stats=current_stats
            )
            
            if reply_result:
                # 保存候选
                self.reply_generator.save_candidate(
                    tweet_id=tweet_id,
                    author_username=author_username,
                    detected_signal=f"velocity_spike_{self.velocity_detector.velocity_threshold_percentile}th_percentile",
                    tweet_text=tweet_text,
                    reply_result=reply_result
                )
                
                return {
                    'tweet_id': tweet_id,
                    'author': author_username,
                    'velocity': weighted_velocity,
                    'reply_draft': reply_result
                }
        
        return None
    
    def get_pending_replies(self, limit: int = 20) -> List[Dict]:
        """获取待处理的截流回复"""
        if not self.analysis_db:
            return []
        return self.analysis_db.get_pending_smart_reply_candidates(limit)
    
    def mark_reply_status(self, candidate_id: int, status: str):
        """更新回复状态"""
        if not self.analysis_db:
            return
        self.analysis_db.update_smart_reply_status(candidate_id, status)
    
    def get_content_suggestions(self, topic: str) -> Dict:
        """获取内容创作建议"""
        return self.content_copilot.get_writing_suggestions(topic)
    
    def get_thread_template(self, style: str = 'any') -> Optional[Dict]:
        """获取 Thread 模板"""
        return self.content_copilot.get_thread_template(style)


def main():
    """命令行入口，支持实时模式和批处理模式"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Flywheel System - 复利系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 批处理模式 - 生成回复候选（GitHub Actions）
  python -m src.flywheel --mode batch --limit 20
  
  # 内容建议生成
  python -m src.flywheel --mode suggest --topic "AI产品设计"
  
  # 获取Thread模板
  python -m src.flywheel --mode template --style educational
  
  # 查看待处理的回复候选
  python -m src.flywheel --mode pending
  
  # 实时模式 - 爆款监控（本地运行，需要实时数据流）
  python -m src.flywheel --mode realtime
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['batch', 'realtime', 'suggest', 'pending', 'template'],
                       help='运行模式')
    parser.add_argument('--limit', type=int, default=20, help='批处理模式下的处理数量')
    parser.add_argument('--topic', type=str, help='主题（用于 suggest）')
    parser.add_argument('--style', type=str, default='any', help='模板风格（用于 template）')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(f"Flywheel System - 模式: {args.mode}")
    print("=" * 60)
    
    flywheel = FlywheelSystem()
    
    if args.mode == 'batch':
        # 批处理模式：从内容机会列表生成回复候选
        logger.info("批处理模式：生成智能回复候选")
        from .db_adapter import get_db_adapter
        db = get_db_adapter()
        
        if not db:
            logger.error("数据库未配置")
            return
        
        # 获取最近的高互动推文
        recent_posts = db.get_all_posts()
        recent_posts['Created At'] = pd.to_datetime(recent_posts['Created At'], errors='coerce')
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_posts = recent_posts[recent_posts['Created At'] > cutoff_time]
        
        # 按互动量排序
        recent_posts['engagement'] = (
            recent_posts.get('Favorite Count', 0) + 
            recent_posts.get('Reply Count', 0) * 3 +
            recent_posts.get('Retweet Count', 0) * 2
        )
        top_posts = recent_posts.nlargest(args.limit, 'engagement')
        
        logger.info(f"找到 {len(top_posts)} 条高互动推文")
        
        generated = 0
        all_candidates = []  # 收集所有候选用于本地保存
        
        for _, row in top_posts.iterrows():
            reply_result = flywheel.reply_generator.generate_reply_draft(
                tweet_text=row['Text'],
                tweet_author=row['Author Username'],
                tweet_stats={
                    'views': row.get('View Count', 0),
                    'likes': row.get('Favorite Count', 0),
                    'replies': row.get('Reply Count', 0),
                    'retweets': row.get('Retweet Count', 0)
                }
            )
            
            if reply_result:
                flywheel.reply_generator.save_candidate(
                    tweet_id=row['ID'],
                    author_username=row['Author Username'],
                    detected_signal='batch_high_engagement',
                    tweet_text=row['Text'],
                    reply_result=reply_result
                )
                generated += 1
                
                # 收集候选数据用于本地保存
                all_candidates.append({
                    'tweet_id': str(row['ID']),
                    'author_username': row['Author Username'],
                    'tweet_text': row['Text'][:500],
                    'detected_signal': 'batch_high_engagement',
                    'reply_draft': reply_result.get('reply_draft', ''),
                    'reply_style': reply_result.get('style', ''),
                    'generated_at': reply_result.get('generated_at', datetime.now().isoformat()),
                    'model_used': reply_result.get('model_used', ''),
                    'stats': {
                        'views': int(row.get('View Count', 0) or 0),
                        'likes': int(row.get('Favorite Count', 0) or 0),
                        'replies': int(row.get('Reply Count', 0) or 0),
                        'retweets': int(row.get('Retweet Count', 0) or 0)
                    }
                })
        
        logger.info(f"✅ 生成了 {generated} 个回复候选")
        
        # 保存到本地文件
        if all_candidates:
            output_dir = 'output/flywheel'
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(output_dir, f'smart_reply_candidates_{timestamp}.json')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'generated_at': datetime.now().isoformat(),
                    'total_candidates': len(all_candidates),
                    'candidates': all_candidates
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📁 本地文件已保存: {output_file}")
        
    elif args.mode == 'realtime':
        # 实时模式：持续监控（本地使用）
        logger.info("实时模式：开始监控爆款...")
        logger.warning("注意：实时模式需要持续的数据流输入")
        logger.info("此模式适合本地运行，不适合 GitHub Actions")
        logger.info("请使用外部工具推送推文更新到 process_tweet_update() 方法")
        
    elif args.mode == 'suggest':
        if not args.topic:
            print("❌ 请提供 --topic 参数")
            return
        suggestions = flywheel.get_content_suggestions(args.topic)
        print(json.dumps(suggestions, ensure_ascii=False, indent=2))
        
    elif args.mode == 'pending':
        pending = flywheel.get_pending_replies(limit=args.limit)
        if pending:
            print(f"\n待处理的截流回复 ({len(pending)} 条):")
            for item in pending:
                print(f"  - [{item['id']}] @{item['author_username']}: {item['tweet_text'][:50]}...")
                print(f"    草稿: {item['draft_reply_text'][:100]}...")
        else:
            print("✅ 没有待处理的截流回复")
            
    elif args.mode == 'template':
        template = flywheel.get_thread_template(args.style)
        if template:
            print(json.dumps(template, ensure_ascii=False, indent=2))
        else:
            print("❌ 未找到匹配的模板")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
