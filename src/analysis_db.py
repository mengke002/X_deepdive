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
# 基于 RESULT_DB_RECOMMENDATIONS.md 设计，核心原则：数据归一化，减少冗余存储
SCHEMA_SQL = """
-- =====================================================
-- 1. 基础元数据
-- =====================================================

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

-- =====================================================
-- 2. 用户维度 (User Analysis)
-- =====================================================

-- 用户计算指标表：存储每个用户的计算指标 (不存储冗余Profile信息)
CREATE TABLE IF NOT EXISTS user_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    username VARCHAR(100) NOT NULL,
    -- 网络指标
    pagerank DOUBLE DEFAULT 0,
    betweenness DOUBLE DEFAULT 0,
    community_id INT,
    in_degree INT DEFAULT 0,
    -- 行为指标
    talkativity_ratio DOUBLE DEFAULT 0,
    professionalism_index DOUBLE DEFAULT 0,
    avg_reply_latency_seconds DOUBLE,
    -- 增长指标 (基于 user_stats_history 计算)
    growth_velocity_7d DOUBLE DEFAULT 0,
    growth_acceleration_7d DOUBLE DEFAULT 0,
    rising_star_velocity DOUBLE DEFAULT 0,
    avg_utility_score DOUBLE DEFAULT 0,
    -- 综合评级
    category ENUM('authority', 'connector', 'rising_star', 'utility_provider') DEFAULT NULL,
    analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_username (username),
    INDEX idx_community (community_id),
    INDEX idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 用户历史快照表 (用于时序分析，计算增长速度与加速度)
CREATE TABLE IF NOT EXISTS user_stats_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    followers_count INT UNSIGNED,
    following_count INT UNSIGNED,
    tweets_count INT UNSIGNED,
    listed_count INT UNSIGNED,
    snapshot_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_username_date (username, snapshot_at)
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

-- LLM: 用户策略画像
CREATE TABLE IF NOT EXISTS user_strategy_dossiers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    username VARCHAR(100) NOT NULL,
    core_identity VARCHAR(255),
    growth_tactics JSON,
    monetization_model VARCHAR(255),
    content_style_summary TEXT,
    actionable_takeaways JSON,
    model_used VARCHAR(128),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- =====================================================
-- 3. 内容维度 (Content Analysis)
-- =====================================================

-- 推文特征表 (升级版：含资产属性与Thread特征)
CREATE TABLE IF NOT EXISTS post_features (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    tweet_id VARCHAR(50) NOT NULL,
    conversation_id VARCHAR(50),
    -- 基础计算特征
    utility_score DOUBLE DEFAULT 0,
    discussion_rate DOUBLE DEFAULT 0,
    virality_rate DOUBLE DEFAULT 0,
    is_question BOOLEAN DEFAULT FALSE,
    topic_ids JSON,
    sentiment_score DOUBLE,
    -- 资产属性 (内容四象限)
    asset_quadrant ENUM('library', 'controversy', 'news', 'cult', 'other') DEFAULT 'other',
    -- Thread 特征
    thread_retention_rate DOUBLE,
    thread_length INT,
    -- 商业信号
    funnel_signal VARCHAR(50),
    embedding_id VARCHAR(128),
    analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_tweet (tweet_id),
    INDEX idx_conv (conversation_id),
    INDEX idx_quadrant (asset_quadrant)
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

-- 内容异常/高价值筛选表 (保留部分快照以便快速展示)
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

-- LLM: 爆款内容蓝图
CREATE TABLE IF NOT EXISTS content_blueprints (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    source_tweet_id VARCHAR(50),
    quadrant ENUM('library', 'controversy', 'news', 'cult'),
    hook_style JSON,
    body_structure VARCHAR(255),
    readability_features JSON,
    emotional_tone VARCHAR(100),
    call_to_action JSON,
    why_viral TEXT,
    replication_template TEXT,
    model_used VARCHAR(128),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_quadrant (quadrant)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- LLM: 内容创意库
CREATE TABLE IF NOT EXISTS content_idea_bank (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    source_tweet_id VARCHAR(50),
    idea_type ENUM('question_to_answer', 'debate_to_join', 'topic_to_cover'),
    topic VARCHAR(255),
    user_intent TEXT,
    suggested_angle TEXT,
    suggested_title VARCHAR(500),
    status ENUM('new', 'drafting', 'published', 'discarded') DEFAULT 'new',
    model_used VARCHAR(128),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- =====================================================
-- 4. 统计维度 (Statistics)
-- =====================================================

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

-- =====================================================
-- 5. 复利系统 (Flywheel)
-- =====================================================

-- 智能截流候选表：存储由"信息时差"策略触发的截流回复建议
CREATE TABLE IF NOT EXISTS smart_reply_candidates (
    id INT AUTO_INCREMENT PRIMARY KEY,
    target_tweet_id VARCHAR(50) NOT NULL,
    author_username VARCHAR(100) NOT NULL,
    detected_signal VARCHAR(255),
    tweet_text TEXT,
    draft_reply_text TEXT,
    status ENUM('pending', 'posted', 'ignored') DEFAULT 'pending',
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_author (author_username)
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
    
    def save_activity_stats(self, session_id: str, stats_data: List[Dict], stat_type: str = 'hourly_heatmap'):
        """
        保存活跃度统计数据
        
        Args:
            session_id: 会话 ID
            stats_data: 统计数据列表
            stat_type: 统计类型 ('hourly_heatmap', 'daily_trend', 'weekly_pattern')
        """
        if not self.is_available() or not stats_data:
            return
        
        cursor = self.connection.cursor()
        
        for stat in stats_data:
            # 根据不同类型获取 time_key
            if stat_type == 'hourly_heatmap':
                time_key = str(stat.get('hour', 0))
            elif stat_type == 'daily_trend':
                time_key = str(stat.get('date', ''))
            elif stat_type == 'weekly_pattern':
                time_key = str(stat.get('day_of_week', 0))
            else:
                time_key = str(stat.get('time_key', '0'))
            
            cursor.execute(
                """
                INSERT INTO activity_stats 
                (session_id, stat_type, time_key, activity_count, activity_percentage)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    stat_type,
                    time_key,
                    int(stat.get('activity_count', stat.get('post_count', 0)) or 0),
                    float(stat.get('activity_percentage', 0) or 0)
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(stats_data)} 条 {stat_type} 统计数据")
    
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

    # =====================================================
    # 新增保存方法 (基于 RESULT_DB_RECOMMENDATIONS.md)
    # =====================================================
    
    def save_user_stats_history(self, users_df: pd.DataFrame):
        """
        保存用户历史快照数据（用于时序分析）
        建议每天运行一次快照任务
        """
        if not self.is_available() or users_df.empty:
            return
        
        cursor = self.connection.cursor()
        
        for _, row in users_df.iterrows():
            cursor.execute(
                """
                INSERT INTO user_stats_history 
                (username, followers_count, following_count, tweets_count, listed_count)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    row.get('username', ''),
                    int(row.get('followers_count', 0) or 0),
                    int(row.get('following_count', 0) or 0),
                    int(row.get('tweets_count', 0) or 0),
                    int(row.get('listed_count', 0) or 0)
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(users_df)} 条用户历史快照数据")

    def calculate_growth_metrics(self, username: str, days: int = 7) -> Dict[str, float]:
        """
        基于用户历史快照计算增长指标
        返回 7 天平均日增粉数(一阶导)和增长加速度(二阶导)
        """
        if not self.is_available():
            return {'growth_velocity_7d': 0, 'growth_acceleration_7d': 0}
        
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT followers_count, snapshot_at
            FROM user_stats_history
            WHERE username = %s
            ORDER BY snapshot_at DESC
            LIMIT %s
            """,
            (username, days + 1)
        )
        rows = cursor.fetchall()
        cursor.close()
        
        if len(rows) < 2:
            return {'growth_velocity_7d': 0, 'growth_acceleration_7d': 0}
        
        # 计算一阶导（增长速度）
        velocities = []
        for i in range(len(rows) - 1):
            delta = rows[i][0] - rows[i + 1][0]  # followers diff
            velocities.append(delta)
        
        avg_velocity = sum(velocities) / len(velocities) if velocities else 0
        
        # 计算二阶导（加速度）
        accelerations = []
        for i in range(len(velocities) - 1):
            accel = velocities[i] - velocities[i + 1]
            accelerations.append(accel)
        
        avg_acceleration = sum(accelerations) / len(accelerations) if accelerations else 0
        
        return {
            'growth_velocity_7d': avg_velocity,
            'growth_acceleration_7d': avg_acceleration
        }

    def save_user_strategy_dossiers(self, session_id: str, dossiers: List[Dict]):
        """保存 LLM 生成的用户策略画像"""
        if not self.is_available() or not dossiers:
            return
        
        cursor = self.connection.cursor()
        
        for dossier in dossiers:
            cursor.execute(
                """
                INSERT INTO user_strategy_dossiers 
                (session_id, username, core_identity, growth_tactics, monetization_model, 
                 content_style_summary, actionable_takeaways, model_used)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    dossier.get('id', dossier.get('username', '')),
                    dossier.get('persona', dossier.get('core_identity', '')),
                    json.dumps(dossier.get('growth_tactics', dossier.get('content_focus', [])), ensure_ascii=False),
                    dossier.get('inferred_goal', dossier.get('monetization_model', '')),
                    dossier.get('summary', dossier.get('content_style_summary', '')),
                    json.dumps(dossier.get('actionable_takeaways', []), ensure_ascii=False),
                    dossier.get('model_used', '')
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(dossiers)} 条用户策略画像数据")

    def save_content_blueprints(self, session_id: str, blueprints: List[Dict]):
        """保存 LLM 生成的爆款内容蓝图"""
        if not self.is_available() or not blueprints:
            return
        
        cursor = self.connection.cursor()
        
        for bp in blueprints:
            cursor.execute(
                """
                INSERT INTO content_blueprints 
                (session_id, source_tweet_id, quadrant, hook_style, body_structure, 
                 readability_features, emotional_tone, call_to_action, why_viral, 
                 replication_template, model_used)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    str(bp.get('id', '')),
                    bp.get('quadrant', bp.get('asset_quadrant')),
                    json.dumps(bp.get('hook_style', {}), ensure_ascii=False),
                    bp.get('body_structure', ''),
                    json.dumps(bp.get('readability_features', []), ensure_ascii=False),
                    bp.get('emotional_tone', ''),
                    json.dumps(bp.get('call_to_action', {}), ensure_ascii=False),
                    bp.get('why_viral', ''),
                    bp.get('replication_template', ''),
                    bp.get('model_used', '')
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(blueprints)} 条爆款内容蓝图数据")

    def save_content_idea_bank(self, session_id: str, ideas: List[Dict]):
        """保存 LLM 生成的内容创意库"""
        if not self.is_available() or not ideas:
            return
        
        cursor = self.connection.cursor()
        
        for idea in ideas:
            # 确定 idea_type
            opp_type = idea.get('opportunity_type', '')
            if opp_type == 'unanswered_question':
                idea_type = 'question_to_answer'
            elif opp_type == 'hot_debate':
                idea_type = 'debate_to_join'
            else:
                idea_type = 'topic_to_cover'
            
            content_suggestion = idea.get('content_suggestion', {})
            
            cursor.execute(
                """
                INSERT INTO content_idea_bank 
                (session_id, source_tweet_id, idea_type, topic, user_intent, 
                 suggested_angle, suggested_title, model_used)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    str(idea.get('id', '')),
                    idea_type,
                    idea.get('core_topic', ''),
                    idea.get('user_intent', ''),
                    content_suggestion.get('angle', ''),
                    content_suggestion.get('title', ''),
                    idea.get('model_used', '')
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(ideas)} 条内容创意数据")

    def save_smart_reply_candidates(self, candidates: List[Dict]):
        """保存智能截流候选回复"""
        if not self.is_available() or not candidates:
            return
        
        cursor = self.connection.cursor()
        
        for cand in candidates:
            cursor.execute(
                """
                INSERT INTO smart_reply_candidates 
                (target_tweet_id, author_username, detected_signal, tweet_text, draft_reply_text)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    str(cand.get('target_tweet_id', '')),
                    cand.get('author_username', ''),
                    cand.get('detected_signal', ''),
                    cand.get('tweet_text', '')[:2000],
                    cand.get('draft_reply_text', '')
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(candidates)} 条智能截流候选数据")

    def update_smart_reply_status(self, candidate_id: int, status: str):
        """更新智能截流候选的状态"""
        if not self.is_available():
            return
        
        if status not in ('pending', 'posted', 'ignored'):
            logger.warning(f"无效的状态: {status}")
            return
        
        cursor = self.connection.cursor()
        cursor.execute(
            "UPDATE smart_reply_candidates SET status = %s WHERE id = %s",
            (status, candidate_id)
        )
        self.connection.commit()
        cursor.close()

    def save_post_features_enhanced(self, session_id: str, features_df: pd.DataFrame):
        """保存推文计算特征（升级版：含资产属性与Thread特征）"""
        if not self.is_available() or features_df.empty:
            return
        
        cursor = self.connection.cursor()
        
        for _, row in features_df.iterrows():
            cursor.execute(
                """
                INSERT INTO post_features 
                (session_id, tweet_id, conversation_id, utility_score, discussion_rate, 
                 virality_rate, is_question, topic_ids, sentiment_score, asset_quadrant,
                 thread_retention_rate, thread_length, funnel_signal, embedding_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    str(row.get('tweet_id', '')),
                    row.get('conversation_id'),
                    float(row.get('utility_score', 0) or 0),
                    float(row.get('discussion_rate', 0) or 0),
                    float(row.get('virality_rate', 0) or 0),
                    bool(row.get('is_question', False)),
                    json.dumps(row.get('topic_ids', []), ensure_ascii=False),
                    float(row.get('sentiment_score', 0) or 0) if pd.notna(row.get('sentiment_score')) else None,
                    row.get('asset_quadrant', 'other'),
                    float(row.get('thread_retention_rate', 0) or 0) if pd.notna(row.get('thread_retention_rate')) else None,
                    int(row.get('thread_length', 0) or 0) if pd.notna(row.get('thread_length')) else None,
                    row.get('funnel_signal'),
                    row.get('embedding_id')
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(features_df)} 条推文特征数据（升级版）")

    def save_content_efficiency(self, session_id: str, efficiency_df: pd.DataFrame):
        """保存内容效能统计数据"""
        if not self.is_available() or efficiency_df.empty:
            return
        
        cursor = self.connection.cursor()
        
        for _, row in efficiency_df.iterrows():
            cursor.execute(
                """
                INSERT INTO content_efficiency 
                (session_id, media_type, post_count, avg_views, avg_likes, 
                 avg_replies, avg_bookmarks, avg_utility_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    row.get('media_type', 'unknown'),
                    int(row.get('post_count', 0) or 0),
                    float(row.get('view_count', row.get('avg_views', 0)) or 0),
                    float(row.get('like_count', row.get('avg_likes', 0)) or 0),
                    float(row.get('reply_count', row.get('avg_replies', 0)) or 0),
                    float(row.get('bookmark_count', row.get('avg_bookmarks', 0)) or 0),
                    float(row.get('utility_score', row.get('avg_utility_score', 0)) or 0)
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(efficiency_df)} 条内容效能统计数据")

    def get_pending_smart_reply_candidates(self, limit: int = 20) -> List[Dict]:
        """获取待处理的智能截流候选"""
        if not self.is_available():
            return []
        
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT id, target_tweet_id, author_username, detected_signal, 
                   tweet_text, draft_reply_text, detected_at
            FROM smart_reply_candidates
            WHERE status = 'pending'
            ORDER BY detected_at DESC
            LIMIT %s
            """,
            (limit,)
        )
        rows = cursor.fetchall()
        cursor.close()
        
        columns = ['id', 'target_tweet_id', 'author_username', 'detected_signal', 
                   'tweet_text', 'draft_reply_text', 'detected_at']
        return [dict(zip(columns, row)) for row in rows]

    # =====================================================
    # Phase 2 LLM 候选列表查询方法（从数据库读取备选）
    # =====================================================
    
    def get_content_outliers_for_llm(self, limit: int = 50, session_id: str = None) -> List[Dict]:
        """
        获取内容异常点作为 LLM 分析候选（替代 list_posts_outliers.csv）
        
        用于：爆款内容逆向工程 (viral_content 任务)
        """
        if not self.is_available():
            return []
        
        cursor = self.connection.cursor()
        
        # 如果指定了 session_id，则查询该会话的数据；否则查询最近的数据
        if session_id:
            cursor.execute(
                """
                SELECT co.tweet_id as id, co.author, co.text, co.outlier_type, co.score,
                       pf.view_count, pf.like_count, pf.bookmark_count, pf.reply_count,
                       pf.utility_score, pf.asset_quadrant
                FROM content_outliers co
                LEFT JOIN post_features pf ON co.tweet_id = pf.tweet_id AND co.session_id = pf.session_id
                WHERE co.session_id = %s
                ORDER BY co.score DESC
                LIMIT %s
                """,
                (session_id, limit)
            )
        else:
            # 获取最近会话的数据
            cursor.execute(
                """
                SELECT co.tweet_id as id, co.author, co.text, co.outlier_type, co.score,
                       pf.view_count, pf.like_count, pf.bookmark_count, pf.reply_count,
                       pf.utility_score, pf.asset_quadrant
                FROM content_outliers co
                LEFT JOIN post_features pf ON co.tweet_id = pf.tweet_id
                WHERE co.session_id = (
                    SELECT session_id FROM analysis_sessions 
                    WHERE status = 'completed' 
                    ORDER BY completed_at DESC LIMIT 1
                )
                ORDER BY co.score DESC
                LIMIT %s
                """,
                (limit,)
            )
        
        rows = cursor.fetchall()
        cursor.close()
        
        columns = ['id', 'author', 'text', 'outlier_type', 'score',
                   'view_count', 'like_count', 'bookmark_count', 'reply_count',
                   'utility_score', 'asset_quadrant']
        return [dict(zip(columns, row)) for row in rows]

    def get_key_users_for_llm(self, limit: int = 50, session_id: str = None) -> List[Dict]:
        """
        获取关键用户作为 LLM 分析候选（替代 list_users_key_players.csv）
        
        用于：用户策略画像 (user_strategy 任务)
        """
        if not self.is_available():
            return []
        
        cursor = self.connection.cursor()
        
        if session_id:
            cursor.execute(
                """
                SELECT username, pagerank, betweenness, community_id, 
                       talkativity_ratio, professionalism_index, 
                       rising_star_velocity, avg_utility_score, category
                FROM user_metrics
                WHERE session_id = %s
                ORDER BY pagerank DESC
                LIMIT %s
                """,
                (session_id, limit)
            )
        else:
            cursor.execute(
                """
                SELECT username, pagerank, betweenness, community_id, 
                       talkativity_ratio, professionalism_index, 
                       rising_star_velocity, avg_utility_score, category
                FROM user_metrics
                WHERE session_id = (
                    SELECT session_id FROM analysis_sessions 
                    WHERE status = 'completed' 
                    ORDER BY completed_at DESC LIMIT 1
                )
                ORDER BY pagerank DESC
                LIMIT %s
                """,
                (limit,)
            )
        
        rows = cursor.fetchall()
        cursor.close()
        
        columns = ['username', 'pagerank', 'betweenness', 'community_id',
                   'talkativity_ratio', 'professionalism_index',
                   'rising_star_velocity', 'avg_utility_score', 'category']
        
        # 补充 bio 信息（需要从源数据库获取，这里返回空字符串作为占位）
        result = []
        for row in rows:
            d = dict(zip(columns, row))
            d['bio'] = ''  # bio 需要从源数据库获取
            d['name'] = ''  # name 需要从源数据库获取
            result.append(d)
        
        return result

    def get_strong_ties_for_llm(self, limit: int = 50, session_id: str = None) -> List[Dict]:
        """
        获取强互惠关系作为 LLM 分析候选（替代 list_interactions_strong_ties.csv）
        
        用于：关系内涵推理 (relationship 任务)
        """
        if not self.is_available():
            return []
        
        cursor = self.connection.cursor()
        
        if session_id:
            cursor.execute(
                """
                SELECT user_a, user_b, interaction_weight as weight, 
                       interaction_samples_json as interaction_samples, relationship_type
                FROM strong_ties
                WHERE session_id = %s
                ORDER BY interaction_weight DESC
                LIMIT %s
                """,
                (session_id, limit)
            )
        else:
            cursor.execute(
                """
                SELECT user_a, user_b, interaction_weight as weight, 
                       interaction_samples_json as interaction_samples, relationship_type
                FROM strong_ties
                WHERE session_id = (
                    SELECT session_id FROM analysis_sessions 
                    WHERE status = 'completed' 
                    ORDER BY completed_at DESC LIMIT 1
                )
                ORDER BY interaction_weight DESC
                LIMIT %s
                """,
                (limit,)
            )
        
        rows = cursor.fetchall()
        cursor.close()
        
        columns = ['user_a', 'user_b', 'weight', 'interaction_samples', 'relationship_type']
        return [dict(zip(columns, row)) for row in rows]

    def get_content_opportunities_for_llm(self, limit: int = 50, session_id: str = None) -> List[Dict]:
        """
        获取内容机会作为 LLM 分析候选（替代 list_content_opportunities.csv）
        
        用于：内容机会挖掘 (content_opportunity 任务)
        """
        if not self.is_available():
            return []
        
        cursor = self.connection.cursor()
        
        if session_id:
            cursor.execute(
                """
                SELECT co.tweet_id as id, co.author, co.text, co.outlier_type as opportunity_type,
                       pf.reply_count, pf.view_count, pf.is_question
                FROM content_outliers co
                LEFT JOIN post_features pf ON co.tweet_id = pf.tweet_id AND co.session_id = pf.session_id
                WHERE co.session_id = %s 
                  AND (co.outlier_type IN ('unanswered_question', 'hot_debate') 
                       OR pf.is_question = TRUE)
                ORDER BY pf.reply_count DESC
                LIMIT %s
                """,
                (session_id, limit)
            )
        else:
            cursor.execute(
                """
                SELECT co.tweet_id as id, co.author, co.text, co.outlier_type as opportunity_type,
                       pf.reply_count, pf.view_count, pf.is_question
                FROM content_outliers co
                LEFT JOIN post_features pf ON co.tweet_id = pf.tweet_id
                WHERE co.session_id = (
                    SELECT session_id FROM analysis_sessions 
                    WHERE status = 'completed' 
                    ORDER BY completed_at DESC LIMIT 1
                )
                AND (co.outlier_type IN ('unanswered_question', 'hot_debate') 
                     OR pf.is_question = TRUE)
                ORDER BY pf.reply_count DESC
                LIMIT %s
                """,
                (limit,)
            )
        
        rows = cursor.fetchall()
        cursor.close()
        
        columns = ['id', 'author', 'text', 'opportunity_type', 'reply_count', 'view_count', 'is_question']
        return [dict(zip(columns, row)) for row in rows]

    def get_viral_threads_for_llm(self, limit: int = 30, session_id: str = None) -> List[Dict]:
        """
        获取高留存 Thread 作为 LLM 分析候选（替代 list_threads_viral.csv）
        
        用于：Thread 结构分析 (thread 任务)
        """
        if not self.is_available():
            return []
        
        cursor = self.connection.cursor()
        
        if session_id:
            cursor.execute(
                """
                SELECT DISTINCT pf.conversation_id, pf.thread_length, pf.thread_retention_rate as retention_rate,
                       (SELECT author FROM content_outliers WHERE tweet_id = pf.tweet_id LIMIT 1) as author
                FROM post_features pf
                WHERE pf.session_id = %s 
                  AND pf.thread_length >= 2
                  AND pf.thread_retention_rate IS NOT NULL
                ORDER BY pf.thread_retention_rate DESC
                LIMIT %s
                """,
                (session_id, limit)
            )
        else:
            cursor.execute(
                """
                SELECT DISTINCT pf.conversation_id, pf.thread_length, pf.thread_retention_rate as retention_rate,
                       (SELECT author FROM content_outliers WHERE tweet_id = pf.tweet_id LIMIT 1) as author
                FROM post_features pf
                WHERE pf.session_id = (
                    SELECT session_id FROM analysis_sessions 
                    WHERE status = 'completed' 
                    ORDER BY completed_at DESC LIMIT 1
                )
                AND pf.thread_length >= 2
                AND pf.thread_retention_rate IS NOT NULL
                ORDER BY pf.thread_retention_rate DESC
                LIMIT %s
                """,
                (limit,)
            )
        
        rows = cursor.fetchall()
        cursor.close()
        
        columns = ['conversation_id', 'thread_length', 'retention_rate', 'author']
        return [dict(zip(columns, row)) for row in rows]

    def get_funnel_signals_for_llm(self, limit: int = 50, session_id: str = None) -> List[Dict]:
        """
        获取商业信号数据作为 LLM 分析候选（替代 stats_funnel_signals.csv）
        
        用于：商业模式解码 (monetization 任务)
        """
        if not self.is_available():
            return []
        
        cursor = self.connection.cursor()
        
        if session_id:
            cursor.execute(
                """
                SELECT pf.tweet_id, co.author, pf.funnel_signal, 
                       pf.view_count, pf.like_count, SUBSTRING(co.text, 1, 100) as text_preview
                FROM post_features pf
                JOIN content_outliers co ON pf.tweet_id = co.tweet_id AND pf.session_id = co.session_id
                WHERE pf.session_id = %s 
                  AND pf.funnel_signal IS NOT NULL
                ORDER BY pf.view_count DESC
                LIMIT %s
                """,
                (session_id, limit)
            )
        else:
            cursor.execute(
                """
                SELECT pf.tweet_id, co.author, pf.funnel_signal, 
                       pf.view_count, pf.like_count, SUBSTRING(co.text, 1, 100) as text_preview
                FROM post_features pf
                JOIN content_outliers co ON pf.tweet_id = co.tweet_id
                WHERE pf.session_id = (
                    SELECT session_id FROM analysis_sessions 
                    WHERE status = 'completed' 
                    ORDER BY completed_at DESC LIMIT 1
                )
                AND pf.funnel_signal IS NOT NULL
                ORDER BY pf.view_count DESC
                LIMIT %s
                """,
                (limit,)
            )
        
        rows = cursor.fetchall()
        cursor.close()
        
        columns = ['tweet_id', 'author', 'funnel_signal', 'view_count', 'like_count', 'text_preview']
        return [dict(zip(columns, row)) for row in rows]

    def get_latest_completed_session_id(self) -> Optional[str]:
        """获取最近完成的分析会话 ID"""
        if not self.is_available():
            return None
        
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT session_id FROM analysis_sessions 
            WHERE status = 'completed' 
            ORDER BY completed_at DESC 
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        cursor.close()
        
        return row[0] if row else None


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