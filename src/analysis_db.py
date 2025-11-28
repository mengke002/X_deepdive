#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析结果数据库模块
用于将分析结果存储到 ANALYSIS_DATABASE_URL 指定的数据库中
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


# 数据库表结构定义（SQL）
SCHEMA_SQL = """
-- 分析会话表：记录每次分析运行的元数据
CREATE TABLE IF NOT EXISTS analysis_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) UNIQUE NOT NULL,
    started_at DATETIME NOT NULL,
    completed_at DATETIME,
    status ENUM('running', 'completed', 'failed') DEFAULT 'running',
    config_json TEXT,
    stats_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- LLM 输出记录表 (用于审计与调试)
CREATE TABLE IF NOT EXISTS llm_outputs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50),
    task_type VARCHAR(50) NOT NULL,
    target_id VARCHAR(128),
    model_used VARCHAR(128),
    prompt_tokens INT DEFAULT 0,
    completion_tokens INT DEFAULT 0,
    total_cost DOUBLE DEFAULT 0,
    raw_output JSON,
    parsed_output JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_target (target_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 用户分析结果表：存储每个用户的计算指标 (不存储冗余Profile信息)
CREATE TABLE IF NOT EXISTS user_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    username VARCHAR(100) NOT NULL,
    followers_count INT DEFAULT 0,
    pagerank DOUBLE DEFAULT 0,
    betweenness DOUBLE DEFAULT 0,
    in_degree INT DEFAULT 0,
    community_id INT,
    talkativity_ratio DOUBLE DEFAULT 0,
    professionalism_index DOUBLE DEFAULT 0,
    avg_reply_latency_seconds DOUBLE,
    rising_star_velocity DOUBLE DEFAULT 0,
    avg_utility_score DOUBLE DEFAULT 0,
    category ENUM('authority', 'connector', 'rising_star', 'utility_provider') DEFAULT NULL,
    analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_username (username),
    INDEX idx_community (community_id),
    INDEX idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 社群分析表：存储社群级别的统计
CREATE TABLE IF NOT EXISTS community_stats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    community_id INT NOT NULL,
    member_count INT DEFAULT 0,
    avg_pagerank DOUBLE DEFAULT 0,
    avg_betweenness DOUBLE DEFAULT 0,
    total_followers INT DEFAULT 0,
    top_members_json TEXT,
    topic_keywords_json TEXT,
    analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_community (community_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 互动关系表：存储强互惠关系
CREATE TABLE IF NOT EXISTS strong_ties (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    user_a VARCHAR(100) NOT NULL,
    user_b VARCHAR(100) NOT NULL,
    interaction_weight INT DEFAULT 0,
    interaction_samples_json TEXT,
    relationship_type ENUM('reciprocal', 'one_way') DEFAULT 'reciprocal',
    analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_users (user_a, user_b)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 推文特征表 (轻量级，无文本)
CREATE TABLE IF NOT EXISTS post_features (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    tweet_id VARCHAR(50) NOT NULL,
    conversation_id VARCHAR(50),
    utility_score DOUBLE DEFAULT 0,
    discussion_rate DOUBLE DEFAULT 0,
    is_question BOOLEAN DEFAULT FALSE,
    topic_ids JSON,
    sentiment_score DOUBLE,
    embedding_id VARCHAR(128),
    analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_tweet (tweet_id),
    INDEX idx_conv (conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 对话结构表 (仅存拓扑关系)
CREATE TABLE IF NOT EXISTS conversation_structures (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    conversation_id VARCHAR(50) NOT NULL,
    tweet_id VARCHAR(50) NOT NULL,
    in_reply_to_tweet_id VARCHAR(50),
    depth INT DEFAULT 0,
    branch_id INT DEFAULT 0,
    analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_conv_tweet (conversation_id, tweet_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 内容异常/高价值筛选表 (保留部分快照)
CREATE TABLE IF NOT EXISTS content_outliers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    tweet_id VARCHAR(50) NOT NULL,
    author VARCHAR(100),
    text TEXT,
    created_at DATETIME,
    outlier_type ENUM('high_utility', 'high_traffic', 'high_discussion', 'unanswered_question', 'hot_debate') DEFAULT NULL,
    score DOUBLE DEFAULT 0,
    analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_author (author),
    INDEX idx_outlier_type (outlier_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 活跃度统计表：存储时间维度的活跃统计
CREATE TABLE IF NOT EXISTS activity_stats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    stat_type ENUM('hourly_heatmap', 'daily_trend', 'weekly_pattern') NOT NULL,
    time_key VARCHAR(20) NOT NULL,
    activity_count INT DEFAULT 0,
    activity_percentage DOUBLE DEFAULT 0,
    analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_type (stat_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 内容效能统计表
CREATE TABLE IF NOT EXISTS content_efficiency (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    media_type VARCHAR(50) NOT NULL,
    post_count INT DEFAULT 0,
    avg_views DOUBLE DEFAULT 0,
    avg_likes DOUBLE DEFAULT 0,
    avg_replies DOUBLE DEFAULT 0,
    avg_bookmarks DOUBLE DEFAULT 0,
    avg_utility_score DOUBLE DEFAULT 0,
    analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 潜在新用户表
CREATE TABLE IF NOT EXISTS potential_new_users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    username VARCHAR(100) NOT NULL,
    weighted_reply_score DOUBLE DEFAULT 0,
    reply_count INT DEFAULT 0,
    avg_replier_pagerank DOUBLE DEFAULT 0,
    analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_score (weighted_reply_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


class AnalysisDatabaseAdapter:
    """分析结果数据库适配器"""
    
    def __init__(self):
        self.connection = None
        self._connect()
    
    def _get_database_config(self) -> Optional[Dict[str, Any]]:
        """获取分析数据库配置"""
        db_uri = os.getenv('ANALYSIS_DATABASE_URL')
        
        if not db_uri:
            logger.info("未配置 ANALYSIS_DATABASE_URL，分析结果将不会存储到数据库")
            return None
        
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
        
        logger.error(f"无法解析 ANALYSIS_DATABASE_URL: {db_uri}")
        return None
    
    def _connect(self):
        """建立数据库连接"""
        config = self._get_database_config()
        if not config:
            return
        
        try:
            import pymysql
            
            # 移除 None 值的 ssl 配置
            if config.get('ssl') is None:
                config.pop('ssl', None)
            
            self.connection = pymysql.connect(**config)
            logger.info(f"分析数据库连接成功: {config['host']}:{config['port']}/{config['database']}")
            
            # 初始化数据库表
            self._init_schema()
        except ImportError:
            logger.error("未安装 pymysql，请运行: pip install pymysql")
        except Exception as e:
            logger.error(f"分析数据库连接失败: {e}")
    
    def _init_schema(self):
        """初始化数据库表结构"""
        if not self.connection:
            return
        
        cursor = self.connection.cursor()
        
        # 分割并执行每个 CREATE TABLE 语句
        for statement in SCHEMA_SQL.split(';'):
            statement = statement.strip()
            if statement and statement.startswith('CREATE TABLE'):
                try:
                    cursor.execute(statement)
                except Exception as e:
                    logger.warning(f"创建表时出错（可能已存在）: {e}")
        
        self.connection.commit()
        cursor.close()
        logger.info("分析数据库表结构初始化完成")
    
    def _ensure_connection(self):
        """确保数据库连接有效"""
        if self.connection is None:
            self._connect()
        elif not self.connection.open:
            self._connect()
    
    def is_available(self) -> bool:
        """检查分析数据库是否可用"""
        return self.connection is not None and self.connection.open
    
    def close(self):
        """关闭数据库连接"""
        if self.connection and self.connection.open:
            self.connection.close()
            logger.info("分析数据库连接已关闭")
    
    def create_session(self, config_dict: Optional[Dict] = None) -> str:
        """
        创建新的分析会话
        
        Returns:
            session_id: 会话ID
        """
        if not self.is_available():
            return datetime.now().strftime('%Y%m%d_%H%M%S')
        
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO analysis_sessions (session_id, started_at, status, config_json)
            VALUES (%s, %s, 'running', %s)
            """,
            (session_id, datetime.now(), json.dumps(config_dict) if config_dict else None)
        )
        self.connection.commit()
        cursor.close()
        
        logger.info(f"创建分析会话: {session_id}")
        return session_id
    
    def complete_session(self, session_id: str, stats_dict: Optional[Dict] = None):
        """标记会话完成"""
        if not self.is_available():
            return
        
        cursor = self.connection.cursor()
        cursor.execute(
            """
            UPDATE analysis_sessions 
            SET completed_at = %s, status = 'completed', stats_json = %s
            WHERE session_id = %s
            """,
            (datetime.now(), json.dumps(stats_dict) if stats_dict else None, session_id)
        )
        self.connection.commit()
        cursor.close()
        
        logger.info(f"会话 {session_id} 已完成")
    
    def save_llm_output(self, session_id: str, task_type: str, target_id: str,
                       model_used: str, prompt_tokens: int = 0, completion_tokens: int = 0,
                       raw_output: Dict = None, parsed_output: Dict = None):
        """保存 LLM 的输出记录"""
        if not self.is_available():
            return
            
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO llm_outputs 
            (session_id, task_type, target_id, model_used, prompt_tokens, completion_tokens, 
             raw_output, parsed_output)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                session_id,
                task_type,
                target_id,
                model_used,
                prompt_tokens,
                completion_tokens,
                json.dumps(raw_output, ensure_ascii=False) if raw_output else None,
                json.dumps(parsed_output, ensure_ascii=False) if parsed_output else None
            )
        )
        self.connection.commit()
        cursor.close()

    def save_user_metrics(self, session_id: str, users_df: pd.DataFrame):
        """保存用户指标数据"""
        if not self.is_available() or users_df.empty:
            return
        
        cursor = self.connection.cursor()
        
        for _, row in users_df.iterrows():
            cursor.execute(
                """
                INSERT INTO user_metrics 
                (session_id, username, followers_count, pagerank, betweenness,
                 in_degree, community_id, talkativity_ratio, professionalism_index,
                 avg_reply_latency_seconds, rising_star_velocity, avg_utility_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    row.get('username', ''),
                    int(row.get('followers_count', 0) or 0),
                    float(row.get('pagerank', 0) or 0),
                    float(row.get('betweenness', 0) or 0),
                    int(row.get('in_degree', 0) or 0),
                    int(row.get('community_id', 0) or 0) if pd.notna(row.get('community_id')) else None,
                    float(row.get('talkativity_ratio', 0) or 0),
                    float(row.get('professionalism_index', 0) or 0),
                    float(row.get('avg_reply_latency_seconds')) if pd.notna(row.get('avg_reply_latency_seconds')) else None,
                    float(row.get('rising_star_velocity', 0) or 0),
                    float(row.get('avg_utility_score', 0) or 0)
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(users_df)} 条用户指标数据")
    
    def save_community_stats(self, session_id: str, community_stats: List[Dict]):
        """保存社群统计数据"""
        if not self.is_available() or not community_stats:
            return
        
        cursor = self.connection.cursor()
        
        for stat in community_stats:
            cursor.execute(
                """
                INSERT INTO community_stats 
                (session_id, community_id, member_count, avg_pagerank, avg_betweenness,
                 total_followers, top_members_json, topic_keywords_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    stat.get('community_id', 0),
                    stat.get('member_count', 0),
                    stat.get('avg_pagerank', 0),
                    stat.get('avg_betweenness', 0),
                    stat.get('total_followers', 0),
                    json.dumps(stat.get('top_members', []), ensure_ascii=False),
                    json.dumps(stat.get('topic_keywords', []), ensure_ascii=False)
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(community_stats)} 个社群统计数据")
    
    def save_strong_ties(self, session_id: str, ties_df: pd.DataFrame):
        """保存强互惠关系数据"""
        if not self.is_available() or ties_df.empty:
            return
        
        cursor = self.connection.cursor()
        
        for _, row in ties_df.iterrows():
            cursor.execute(
                """
                INSERT INTO strong_ties 
                (session_id, user_a, user_b, interaction_weight, interaction_samples_json, relationship_type)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    row.get('user_a', ''),
                    row.get('user_b', ''),
                    int(row.get('weight', 0) or 0),
                    row.get('interaction_samples', '[]'),
                    'reciprocal'
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(ties_df)} 条强互惠关系数据")
    
    def save_post_features(self, session_id: str, features_df: pd.DataFrame):
        """保存推文计算特征"""
        if not self.is_available() or features_df.empty:
            return
        
        cursor = self.connection.cursor()
        
        for _, row in features_df.iterrows():
            cursor.execute(
                """
                INSERT INTO post_features 
                (session_id, tweet_id, conversation_id, utility_score, discussion_rate, 
                 is_question, topic_ids, sentiment_score, embedding_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    str(row.get('tweet_id', '')),
                    row.get('conversation_id'),
                    float(row.get('utility_score', 0) or 0),
                    float(row.get('discussion_rate', 0) or 0),
                    bool(row.get('is_question', False)),
                    json.dumps(row.get('topic_ids', []), ensure_ascii=False),
                    float(row.get('sentiment_score', 0) or 0) if pd.notna(row.get('sentiment_score')) else None,
                    row.get('embedding_id')
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(features_df)} 条推文特征数据")

    def save_conversation_structures(self, session_id: str, structure_df: pd.DataFrame):
        """保存对话拓扑结构"""
        if not self.is_available() or structure_df.empty:
            return
        
        cursor = self.connection.cursor()
        
        for _, row in structure_df.iterrows():
            cursor.execute(
                """
                INSERT INTO conversation_structures 
                (session_id, conversation_id, tweet_id, in_reply_to_tweet_id, depth)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    str(row.get('conversation_id', '')),
                    str(row.get('tweet_id', '')),
                    row.get('in_reply_to_tweet_id'),
                    int(row.get('depth', 0) or 0)
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(structure_df)} 条对话结构数据")

    def save_content_outliers(self, session_id: str, outliers_df: pd.DataFrame):
        """保存高价值内容数据 (已优化为轻量级快照)"""
        if not self.is_available() or outliers_df.empty:
            return
        
        cursor = self.connection.cursor()
        
        for _, row in outliers_df.iterrows():
            # 确定 outlier_type
            outlier_type = row.get('outlier_type')
            if not outlier_type:
                if row.get('utility_score', 0) > 0.5:
                    outlier_type = 'high_utility'
                elif row.get('opportunity_type') == 'unanswered_question':
                    outlier_type = 'unanswered_question'
                else:
                    outlier_type = 'high_traffic'
            
            cursor.execute(
                """
                INSERT INTO content_outliers 
                (session_id, tweet_id, author, text, created_at, outlier_type, score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    str(row.get('id', '')),
                    row.get('author', ''),
                    str(row.get('text', ''))[:1000],  # 轻量级快照
                    row.get('created_at'),
                    outlier_type,
                    float(row.get('utility_score', 0) or row.get('score', 0) or 0)
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(outliers_df)} 条高价值内容数据")
    
    def save_activity_stats(self, session_id: str, heatmap_data: List[Dict]):
        """保存活跃度统计数据"""
        if not self.is_available() or not heatmap_data:
            return
        
        cursor = self.connection.cursor()
        
        for stat in heatmap_data:
            cursor.execute(
                """
                INSERT INTO activity_stats 
                (session_id, stat_type, time_key, activity_count, activity_percentage)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    'hourly_heatmap',
                    str(stat.get('hour', 0)),
                    int(stat.get('activity_count', 0) or 0),
                    float(stat.get('activity_percentage', 0) or 0)
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(heatmap_data)} 条活跃度统计数据")
    
    def save_potential_new_users(self, session_id: str, users_df: pd.DataFrame):
        """保存潜在新用户数据"""
        if not self.is_available() or users_df.empty:
            return
        
        cursor = self.connection.cursor()
        
        for _, row in users_df.iterrows():
            cursor.execute(
                """
                INSERT INTO potential_new_users 
                (session_id, username, weighted_reply_score, reply_count, avg_replier_pagerank)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    row.get('Username', ''),
                    float(row.get('WeightedReplyScore', 0) or 0),
                    int(row.get('ReplyCount', 0) or 0),
                    float(row.get('AvgReplierPageRank', 0) or 0)
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(users_df)} 条潜在新用户数据")


def get_analysis_db_adapter() -> Optional[AnalysisDatabaseAdapter]:
    """获取分析数据库适配器实例"""
    try:
        adapter = AnalysisDatabaseAdapter()
        if adapter.is_available():
            return adapter
        return None
    except Exception as e:
        logger.error(f"创建分析数据库适配器失败: {e}")
        return None