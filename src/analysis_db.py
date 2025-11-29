#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析结果数据库模块
用于将分析结果存储到 ANALYSIS_DATABASE_URL 指定的数据库中

存储策略配置（通过环境变量控制）：
- ANALYSIS_STORAGE_MODE: 存储模式
  - 'snapshot': 快照模式，每次运行保存独立记录（默认）
  - 'upsert': 覆盖模式，相同数据只保留最新
  - 'hybrid': 混合模式，快照型数据保留历史，覆盖型数据只保留最新
  
- ANALYSIS_MAX_SESSIONS: 最大保留会话数（默认 10）
  - 自动清理超过此数量的旧会话数据
  - 设为 0 禁用自动清理
  
- ANALYSIS_SKIP_TABLES: 跳过保存的表（逗号分隔）
  - 例如: "activity_stats,content_efficiency" 将跳过这些表的保存

数据类型分类：
- 快照型（需要保留历史）：user_stats_history, llm_outputs, analysis_sessions
- 覆盖型（只需最新）：user_metrics, post_features, activity_stats, content_outliers, 
                       community_stats, strong_ties, content_efficiency, potential_new_users
"""

import os
import re
import json
import logging
import math
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================
# 辅助函数：处理 NaN 值
# =====================================================

def safe_float(value, default=0.0):
    """安全转换为 float，处理 NaN/None/无效值"""
    if value is None:
        return default
    if pd.isna(value):
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    """安全转换为 int，处理 NaN/None/无效值"""
    if value is None:
        return default
    if pd.isna(value):
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return int(f)
    except (ValueError, TypeError):
        return default


def safe_float_or_none(value):
    """安全转换为 float 或 None（用于可空字段）"""
    if value is None:
        return None
    if pd.isna(value):
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def safe_int_or_none(value):
    """安全转换为 int 或 None（用于可空字段）"""
    if value is None:
        return None
    if pd.isna(value):
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return int(f)
    except (ValueError, TypeError):
        return None


# =====================================================
# 存储策略常量
# =====================================================

# 存储模式
STORAGE_MODE_SNAPSHOT = 'snapshot'  # 快照模式：每次运行保存独立记录
STORAGE_MODE_UPSERT = 'upsert'      # 覆盖模式：相同数据只保留最新
STORAGE_MODE_HYBRID = 'hybrid'      # 混合模式：快照型保留，覆盖型更新

# 数据类型分类
SNAPSHOT_TABLES = {
    'user_stats_history',   # 用于时序分析，必须保留历史
    'llm_outputs',          # LLM 审计日志
    'analysis_sessions',    # 会话元数据
}

UPSERT_TABLES = {
    'user_metrics',         # 用户指标，变化小
    'post_features',        # 推文特征，相对稳定
    'activity_stats',       # 活跃度统计，小时热力图/周模式变化小
    'content_outliers',     # 高价值内容，可覆盖更新
    'community_stats',      # 社群统计
    'strong_ties',          # 强互惠关系
    'content_efficiency',   # 内容效能统计
    'potential_new_users',  # 潜在新用户
    'conversation_structures',  # 对话结构
}

# 默认配置
DEFAULT_MAX_SESSIONS = 10


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
        self.source_connection = None  # 源数据库连接（用于获取原始推文统计数据）
        self._source_db_initialized = False  # 懒加载标志
        
        # 加载存储策略配置
        self.storage_mode = os.getenv('ANALYSIS_STORAGE_MODE', STORAGE_MODE_HYBRID).lower()
        self.max_sessions = int(os.getenv('ANALYSIS_MAX_SESSIONS', DEFAULT_MAX_SESSIONS))
        self.skip_tables = self._parse_skip_tables(os.getenv('ANALYSIS_SKIP_TABLES', ''))
        
        logger.info(f"存储策略配置: mode={self.storage_mode}, max_sessions={self.max_sessions}, skip_tables={self.skip_tables}")
        
        self._connect()
        # 源数据库采用懒加载，只在 LLM 任务需要时才连接
    
    def _parse_skip_tables(self, skip_str: str) -> Set[str]:
        """解析跳过保存的表列表"""
        if not skip_str:
            return set()
        return {t.strip().lower() for t in skip_str.split(',') if t.strip()}
    
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
    
    def _get_source_database_config(self) -> Optional[Dict[str, Any]]:
        """获取源数据库配置（DATABASE_URL，存储采集的原始数据）"""
        db_uri = os.getenv('DATABASE_URL')
        
        if not db_uri:
            logger.info("未配置 DATABASE_URL，LLM 候选数据将无法补充统计字段")
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
        
        logger.warning(f"无法解析 DATABASE_URL: {db_uri}")
        return None
    
    def _connect_source_db(self):
        """建立源数据库连接"""
        config = self._get_source_database_config()
        if not config:
            return
        
        try:
            import pymysql
            
            if config.get('ssl') is None:
                config.pop('ssl', None)
            
            # 添加超时设置，避免在 CI 环境中无限等待
            config['connect_timeout'] = 10  # 连接超时 10 秒
            config['read_timeout'] = 30     # 读取超时 30 秒
            config['write_timeout'] = 30    # 写入超时 30 秒
            
            self.source_connection = pymysql.connect(**config)
            logger.info(f"源数据库连接成功: {config['host']}:{config['port']}/{config['database']}")
        except Exception as e:
            logger.warning(f"源数据库连接失败（LLM 候选数据将无法补充统计字段）: {e}")
            self.source_connection = None
    
    def _get_tweet_stats_from_source(self, tweet_ids: List[str]) -> Dict[str, Dict]:
        """
        从源数据库获取推文统计数据
        
        Args:
            tweet_ids: 推文 ID 列表
        
        Returns:
            Dict[tweet_id -> {view_count, like_count, bookmark_count, reply_count}]
        """
        if not tweet_ids:
            return {}
        
        # 懒加载：首次使用时才连接源数据库
        if not self._source_db_initialized:
            self._connect_source_db()
            self._source_db_initialized = True
        
        if not self.source_connection:
            return {}
        
        try:
            cursor = self.source_connection.cursor()
            # 构建 IN 查询
            placeholders = ','.join(['%s'] * len(tweet_ids))
            cursor.execute(
                f"""
                SELECT tweet_id, view_count, favorite_count, bookmark_count, reply_count
                FROM twitter_posts
                WHERE tweet_id IN ({placeholders})
                """,
                tweet_ids
            )
            rows = cursor.fetchall()
            cursor.close()
            
            return {
                str(row[0]): {
                    'view_count': row[1] or 0,
                    'like_count': row[2] or 0,
                    'bookmark_count': row[3] or 0,
                    'reply_count': row[4] or 0
                }
                for row in rows
            }
        except Exception as e:
            logger.warning(f"从源数据库获取推文统计失败: {e}")
            return {}
    
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
            
            # 添加超时设置，避免在 CI 环境中无限等待
            config['connect_timeout'] = 10  # 连接超时 10 秒
            config['read_timeout'] = 60     # 读取超时 60 秒
            config['write_timeout'] = 60    # 写入超时 60 秒
            
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
            # 跳过空语句
            if not statement:
                continue
            # 移除 SQL 注释行，只保留实际的 SQL 语句
            lines = statement.split('\n')
            sql_lines = [line for line in lines if not line.strip().startswith('--')]
            clean_statement = '\n'.join(sql_lines).strip()
            
            if clean_statement and 'CREATE TABLE' in clean_statement:
                try:
                    cursor.execute(clean_statement)
                except Exception as e:
                    logger.warning(f"创建表时出错（可能已存在）: {e}")
        
        self.connection.commit()
        cursor.close()
        logger.info("分析数据库表结构初始化完成")
    
    def _should_skip_table(self, table_name: str) -> bool:
        """检查是否应该跳过该表的保存"""
        return table_name.lower() in self.skip_tables
    
    def _should_use_upsert(self, table_name: str) -> bool:
        """
        判断该表是否应该使用覆盖模式
        
        Returns:
            True: 使用覆盖模式（先删后插或 UPSERT）
            False: 使用快照模式（直接插入）
        """
        if self.storage_mode == STORAGE_MODE_SNAPSHOT:
            return False
        elif self.storage_mode == STORAGE_MODE_UPSERT:
            return True
        else:  # hybrid 模式
            return table_name.lower() in UPSERT_TABLES
    
    def _delete_old_session_data(self, table_name: str, session_id: str):
        """
        删除指定 session 的旧数据（用于覆盖模式）
        
        注意：只删除同一 session_id 的旧数据，不影响其他 session
        """
        if not self.is_available():
            return
        
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"DELETE FROM {table_name} WHERE session_id = %s", (session_id,))
            deleted = cursor.rowcount
            if deleted > 0:
                logger.debug(f"覆盖模式：已删除 {table_name} 表中 session={session_id} 的 {deleted} 条旧数据")
        except Exception as e:
            logger.warning(f"删除旧数据时出错: {e}")
        finally:
            self.connection.commit()
            cursor.close()
    
    def cleanup_old_sessions(self, keep_count: int = None):
        """
        清理旧的会话数据，只保留最近 N 个会话
        
        Args:
            keep_count: 要保留的会话数，默认使用配置的 max_sessions
        """
        if not self.is_available():
            return
        
        keep_count = keep_count if keep_count is not None else self.max_sessions
        
        if keep_count <= 0:
            logger.info("自动清理已禁用 (max_sessions <= 0)")
            return
        
        cursor = self.connection.cursor()
        
        try:
            # 获取所有会话，按时间倒序
            cursor.execute(
                "SELECT session_id FROM analysis_sessions ORDER BY started_at DESC"
            )
            all_sessions = [row[0] for row in cursor.fetchall()]
            
            if len(all_sessions) <= keep_count:
                logger.info(f"当前会话数 ({len(all_sessions)}) <= 保留数 ({keep_count})，无需清理")
                return
            
            # 需要删除的会话
            sessions_to_delete = all_sessions[keep_count:]
            
            logger.info(f"开始清理 {len(sessions_to_delete)} 个旧会话...")
            
            # 需要清理的表（按依赖关系排序）
            tables_to_clean = [
                'user_metrics', 'post_features', 'activity_stats', 
                'content_outliers', 'community_stats', 'strong_ties',
                'content_efficiency', 'potential_new_users', 
                'conversation_structures', 'content_blueprints',
                'content_idea_bank', 'user_strategy_dossiers',
                'llm_outputs'  # llm_outputs 也可以清理旧会话的
            ]
            
            total_deleted = 0
            for session_id in sessions_to_delete:
                session_deleted = 0
                for table in tables_to_clean:
                    try:
                        cursor.execute(f"DELETE FROM {table} WHERE session_id = %s", (session_id,))
                        session_deleted += cursor.rowcount
                    except Exception as e:
                        logger.debug(f"清理 {table} 表时出错（可能表不存在）: {e}")
                
                # 删除会话记录本身
                cursor.execute("DELETE FROM analysis_sessions WHERE session_id = %s", (session_id,))
                total_deleted += session_deleted
                logger.debug(f"已清理会话 {session_id}，删除 {session_deleted} 条记录")
            
            self.connection.commit()
            logger.info(f"清理完成：删除了 {len(sessions_to_delete)} 个会话，共 {total_deleted} 条记录")
            
        except Exception as e:
            logger.error(f"清理旧会话时出错: {e}")
        finally:
            cursor.close()
    
    def get_session_count(self) -> int:
        """获取当前会话总数"""
        if not self.is_available():
            return 0
        
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM analysis_sessions")
        count = cursor.fetchone()[0]
        cursor.close()
        return count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            包含各表记录数、会话数等统计信息的字典
        """
        if not self.is_available():
            return {}
        
        cursor = self.connection.cursor()
        stats = {
            'storage_mode': self.storage_mode,
            'max_sessions': self.max_sessions,
            'tables': {}
        }
        
        tables = [
            'analysis_sessions', 'user_metrics', 'post_features',
            'activity_stats', 'content_outliers', 'community_stats',
            'strong_ties', 'user_stats_history', 'llm_outputs'
        ]
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats['tables'][table] = cursor.fetchone()[0]
            except:
                stats['tables'][table] = 0
        
        cursor.close()
        return stats
    
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
        if self.source_connection and self.source_connection.open:
            self.source_connection.close()
            logger.info("源数据库连接已关闭")
    
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
        """保存用户指标数据（使用批量插入优化）"""
        if not self.is_available() or users_df.empty:
            return
        
        table_name = 'user_metrics'
        if self._should_skip_table(table_name):
            logger.info(f"跳过保存 {table_name}（已配置跳过）")
            return
        
        # 覆盖模式：先删除旧数据
        if self._should_use_upsert(table_name):
            self._delete_old_session_data(table_name, session_id)
        
        cursor = self.connection.cursor()
        
        # 批量插入优化
        batch_size = 1000
        insert_sql = """
            INSERT INTO user_metrics 
            (session_id, username, pagerank, betweenness,
             in_degree, community_id, talkativity_ratio, professionalism_index,
             avg_reply_latency_seconds, rising_star_velocity, avg_utility_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        batch_data = []
        total_count = len(users_df)
        
        for idx, row in enumerate(users_df.itertuples(index=False)):
            batch_data.append((
                session_id,
                getattr(row, 'username', '') or '',
                safe_float(getattr(row, 'pagerank', 0), 0),
                safe_float(getattr(row, 'betweenness', 0), 0),
                safe_int(getattr(row, 'in_degree', 0), 0),
                safe_int_or_none(getattr(row, 'community_id', None)),
                safe_float(getattr(row, 'talkativity_ratio', 0), 0),
                safe_float(getattr(row, 'professionalism_index', 0), 0),
                safe_float_or_none(getattr(row, 'avg_reply_latency_seconds', None)),
                safe_float(getattr(row, 'rising_star_velocity', 0), 0),
                safe_float(getattr(row, 'avg_utility_score', 0), 0)
            ))
            
            if len(batch_data) >= batch_size:
                cursor.executemany(insert_sql, batch_data)
                self.connection.commit()
                batch_data = []
        
        # 插入剩余数据
        if batch_data:
            cursor.executemany(insert_sql, batch_data)
            self.connection.commit()
        
        cursor.close()
        logger.info(f"保存了 {total_count} 条用户指标数据（批量插入）")
    
    def save_community_stats(self, session_id: str, community_stats: List[Dict]):
        """保存社群统计数据"""
        if not self.is_available() or not community_stats:
            return
        
        table_name = 'community_stats'
        if self._should_skip_table(table_name):
            logger.info(f"跳过保存 {table_name}（已配置跳过）")
            return
        
        # 覆盖模式：先删除旧数据
        if self._should_use_upsert(table_name):
            self._delete_old_session_data(table_name, session_id)
        
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
                    safe_int(stat.get('community_id'), 0),
                    safe_int(stat.get('member_count'), 0),
                    safe_float(stat.get('avg_pagerank'), 0),
                    safe_float(stat.get('avg_betweenness'), 0),
                    safe_int(stat.get('total_followers'), 0),
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
        
        table_name = 'strong_ties'
        if self._should_skip_table(table_name):
            logger.info(f"跳过保存 {table_name}（已配置跳过）")
            return
        
        # 覆盖模式：先删除旧数据
        if self._should_use_upsert(table_name):
            self._delete_old_session_data(table_name, session_id)
        
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
                    safe_int(row.get('weight'), 0),
                    row.get('interaction_samples', '[]'),
                    'reciprocal'
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(ties_df)} 条强互惠关系数据")
    
    def save_post_features(self, session_id: str, features_df: pd.DataFrame):
        """保存推文计算特征

        性能优化：使用批量插入，每批 1000 条
        """
        if not self.is_available() or features_df.empty:
            return

        table_name = 'post_features'
        if self._should_skip_table(table_name):
            logger.info(f"跳过保存 {table_name}（已配置跳过）")
            return

        # 覆盖模式：先删除旧数据
        if self._should_use_upsert(table_name):
            self._delete_old_session_data(table_name, session_id)

        cursor = self.connection.cursor()

        # 批量插入优化
        batch_size = 1000
        insert_sql = """
            INSERT INTO post_features
            (session_id, tweet_id, conversation_id, utility_score, discussion_rate,
             is_question, topic_ids, sentiment_score, embedding_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        batch_data = []
        total_count = len(features_df)

        for idx, row in enumerate(features_df.itertuples(index=False)):
            batch_data.append((
                session_id,
                str(getattr(row, 'tweet_id', '') or ''),
                getattr(row, 'conversation_id', None),
                safe_float(getattr(row, 'utility_score', 0), 0),
                safe_float(getattr(row, 'discussion_rate', 0), 0),
                bool(getattr(row, 'is_question', False)),
                json.dumps(getattr(row, 'topic_ids', []) or [], ensure_ascii=False),
                safe_float_or_none(getattr(row, 'sentiment_score', None)),
                getattr(row, 'embedding_id', None)
            ))

            if len(batch_data) >= batch_size:
                cursor.executemany(insert_sql, batch_data)
                self.connection.commit()
                batch_data = []

        if batch_data:
            cursor.executemany(insert_sql, batch_data)
            self.connection.commit()

        cursor.close()
        logger.info(f"保存了 {total_count} 条推文特征数据（批量插入）")

    def save_conversation_structures(self, session_id: str, structure_df: pd.DataFrame):
        """保存对话拓扑结构（使用批量插入优化）"""
        if not self.is_available() or structure_df.empty:
            return

        table_name = 'conversation_structures'
        if self._should_skip_table(table_name):
            logger.info(f"跳过保存 {table_name}（已配置跳过）")
            return

        # 覆盖模式：先删除旧数据
        if self._should_use_upsert(table_name):
            self._delete_old_session_data(table_name, session_id)

        cursor = self.connection.cursor()

        # 批量插入优化
        batch_size = 1000
        insert_sql = """
            INSERT INTO conversation_structures
            (session_id, conversation_id, tweet_id, in_reply_to_tweet_id, depth)
            VALUES (%s, %s, %s, %s, %s)
        """

        batch_data = []
        total_count = len(structure_df)

        for idx, row in enumerate(structure_df.itertuples(index=False)):
            batch_data.append((
                session_id,
                str(getattr(row, 'conversation_id', '') or ''),
                str(getattr(row, 'tweet_id', '') or ''),
                getattr(row, 'in_reply_to_tweet_id', None),
                safe_int(getattr(row, 'depth', 0), 0)
            ))

            if len(batch_data) >= batch_size:
                cursor.executemany(insert_sql, batch_data)
                self.connection.commit()
                batch_data = []

        if batch_data:
            cursor.executemany(insert_sql, batch_data)
            self.connection.commit()

        cursor.close()
        logger.info(f"保存了 {total_count} 条对话结构数据（批量插入）")

    def save_content_outliers(self, session_id: str, outliers_df: pd.DataFrame):
        """保存高价值内容数据 (已优化为轻量级快照，使用批量插入)"""
        if not self.is_available() or outliers_df.empty:
            return

        table_name = 'content_outliers'
        if self._should_skip_table(table_name):
            logger.info(f"跳过保存 {table_name}（已配置跳过）")
            return

        # 覆盖模式：先删除旧数据
        if self._should_use_upsert(table_name):
            self._delete_old_session_data(table_name, session_id)

        cursor = self.connection.cursor()

        # 批量插入优化
        batch_size = 1000
        insert_sql = """
            INSERT INTO content_outliers
            (session_id, tweet_id, author, text, created_at, outlier_type, score)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        batch_data = []
        total_count = len(outliers_df)

        for idx, row in enumerate(outliers_df.itertuples(index=False)):
            # 确定 outlier_type
            outlier_type = getattr(row, 'outlier_type', None)
            if not outlier_type:
                utility_score = getattr(row, 'utility_score', 0) or 0
                opportunity_type = getattr(row, 'opportunity_type', None)
                if utility_score > 0.5:
                    outlier_type = 'high_utility'
                elif opportunity_type == 'unanswered_question':
                    outlier_type = 'unanswered_question'
                else:
                    outlier_type = 'high_traffic'

            text_val = str(getattr(row, 'text', '') or '')[:1000]
            score_val = safe_float(
                getattr(row, 'utility_score', None) or getattr(row, 'score', 0),
                0
            )

            batch_data.append((
                session_id,
                str(getattr(row, 'id', '') or ''),
                getattr(row, 'author', '') or '',
                text_val,
                getattr(row, 'created_at', None),
                outlier_type,
                score_val
            ))

            if len(batch_data) >= batch_size:
                cursor.executemany(insert_sql, batch_data)
                self.connection.commit()
                batch_data = []

        if batch_data:
            cursor.executemany(insert_sql, batch_data)
            self.connection.commit()

        cursor.close()
        logger.info(f"保存了 {total_count} 条高价值内容数据（批量插入）")
    
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
        
        table_name = 'activity_stats'
        if self._should_skip_table(table_name):
            logger.info(f"跳过保存 {table_name}（已配置跳过）")
            return
        
        # 覆盖模式：删除同一 session 和 stat_type 的旧数据
        if self._should_use_upsert(table_name):
            cursor = self.connection.cursor()
            try:
                cursor.execute(
                    f"DELETE FROM {table_name} WHERE session_id = %s AND stat_type = %s",
                    (session_id, stat_type)
                )
                deleted = cursor.rowcount
                if deleted > 0:
                    logger.debug(f"覆盖模式：已删除 {table_name} 表中 session={session_id}, type={stat_type} 的 {deleted} 条旧数据")
            except Exception as e:
                logger.warning(f"删除旧数据时出错: {e}")
            finally:
                self.connection.commit()
                cursor.close()
        
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
                    safe_int(stat.get('activity_count', stat.get('post_count', 0)), 0),
                    safe_float(stat.get('activity_percentage'), 0)
                )
            )
        
        self.connection.commit()
        cursor.close()
        logger.info(f"保存了 {len(stats_data)} 条 {stat_type} 统计数据")
    
    def save_potential_new_users(self, session_id: str, users_df: pd.DataFrame):
        """保存潜在新用户数据（使用批量插入优化）"""
        if not self.is_available() or users_df.empty:
            return
        
        table_name = 'potential_new_users'
        if self._should_skip_table(table_name):
            logger.info(f"跳过保存 {table_name}（已配置跳过）")
            return
        
        # 覆盖模式：先删除旧数据
        if self._should_use_upsert(table_name):
            self._delete_old_session_data(table_name, session_id)
        
        cursor = self.connection.cursor()
        
        # 批量插入优化
        batch_size = 1000
        insert_sql = """
            INSERT INTO potential_new_users 
            (session_id, username, weighted_reply_score, reply_count, avg_replier_pagerank)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        batch_data = []
        total_count = len(users_df)
        
        for idx, row in enumerate(users_df.itertuples(index=False)):
            # 处理列名可能是 Username 或 username 的情况
            username = getattr(row, 'Username', None) or getattr(row, 'username', '') or ''
            batch_data.append((
                session_id,
                username,
                safe_float(getattr(row, 'WeightedReplyScore', 0), 0),
                safe_int(getattr(row, 'ReplyCount', 0), 0),
                safe_float(getattr(row, 'AvgReplierPageRank', 0), 0)
            ))
            
            if len(batch_data) >= batch_size:
                cursor.executemany(insert_sql, batch_data)
                self.connection.commit()
                batch_data = []
        
        # 插入剩余数据
        if batch_data:
            cursor.executemany(insert_sql, batch_data)
            self.connection.commit()
        
        cursor.close()
        logger.info(f"保存了 {total_count} 条潜在新用户数据（批量插入）")

    # =====================================================
    # 新增保存方法 (基于 RESULT_DB_RECOMMENDATIONS.md)
    # =====================================================
    
    def save_user_stats_history(self, users_df: pd.DataFrame):
        """
        保存用户历史快照数据（用于时序分析，使用批量插入优化）
        建议每天运行一次快照任务
        """
        if not self.is_available() or users_df.empty:
            return
        
        cursor = self.connection.cursor()
        
        # 批量插入优化
        batch_size = 1000
        insert_sql = """
            INSERT INTO user_stats_history 
            (username, followers_count, following_count, tweets_count, listed_count)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        batch_data = []
        total_count = len(users_df)
        
        for idx, row in enumerate(users_df.itertuples(index=False)):
            batch_data.append((
                getattr(row, 'username', '') or '',
                safe_int(getattr(row, 'followers_count', 0), 0),
                safe_int(getattr(row, 'following_count', 0), 0),
                safe_int(getattr(row, 'tweets_count', 0), 0),
                safe_int(getattr(row, 'listed_count', 0), 0)
            ))
            
            if len(batch_data) >= batch_size:
                cursor.executemany(insert_sql, batch_data)
                self.connection.commit()
                batch_data = []
        
        # 插入剩余数据
        if batch_data:
            cursor.executemany(insert_sql, batch_data)
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
        """保存推文计算特征（升级版：含资产属性与Thread特征）

        性能优化：使用批量插入，每批 1000 条
        """
        if not self.is_available() or features_df.empty:
            return

        table_name = 'post_features'
        if self._should_skip_table(table_name):
            logger.info(f"跳过保存 {table_name}（已配置跳过）")
            return

        # 覆盖模式：先删除旧数据
        if self._should_use_upsert(table_name):
            self._delete_old_session_data(table_name, session_id)

        cursor = self.connection.cursor()

        # 批量插入优化：准备所有数据
        batch_size = 1000
        insert_sql = """
            INSERT INTO post_features
            (session_id, tweet_id, conversation_id, utility_score, discussion_rate,
             virality_rate, is_question, topic_ids, sentiment_score, asset_quadrant,
             thread_retention_rate, thread_length, funnel_signal, embedding_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        # 准备批量数据
        batch_data = []
        total_count = len(features_df)

        for idx, row in enumerate(features_df.itertuples(index=False)):
            # 使用 getattr 访问 namedtuple 属性，提供默认值
            batch_data.append((
                session_id,
                str(getattr(row, 'tweet_id', '') or ''),
                getattr(row, 'conversation_id', None),
                safe_float(getattr(row, 'utility_score', 0), 0),
                safe_float(getattr(row, 'discussion_rate', 0), 0),
                safe_float(getattr(row, 'virality_rate', 0), 0),
                bool(getattr(row, 'is_question', False)),
                json.dumps(getattr(row, 'topic_ids', []) or [], ensure_ascii=False),
                safe_float_or_none(getattr(row, 'sentiment_score', None)),
                getattr(row, 'asset_quadrant', 'other') or 'other',
                safe_float_or_none(getattr(row, 'thread_retention_rate', None)),
                safe_int_or_none(getattr(row, 'thread_length', None)),
                getattr(row, 'funnel_signal', None),
                getattr(row, 'embedding_id', None)
            ))

            # 每 batch_size 条执行一次批量插入
            if len(batch_data) >= batch_size:
                cursor.executemany(insert_sql, batch_data)
                self.connection.commit()
                logger.debug(f"已插入 {idx + 1}/{total_count} 条推文特征")
                batch_data = []

        # 插入剩余数据
        if batch_data:
            cursor.executemany(insert_sql, batch_data)
            self.connection.commit()

        cursor.close()
        logger.info(f"保存了 {total_count} 条推文特征数据（升级版，批量插入）")

    def save_content_efficiency(self, session_id: str, efficiency_df: pd.DataFrame):
        """保存内容效能统计数据"""
        if not self.is_available() or efficiency_df.empty:
            return
        
        table_name = 'content_efficiency'
        if self._should_skip_table(table_name):
            logger.info(f"跳过保存 {table_name}（已配置跳过）")
            return
        
        # 覆盖模式：先删除旧数据
        if self._should_use_upsert(table_name):
            self._delete_old_session_data(table_name, session_id)
        
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
                    safe_int(row.get('post_count'), 0),
                    safe_float(row.get('view_count', row.get('avg_views', 0)), 0),
                    safe_float(row.get('like_count', row.get('avg_likes', 0)), 0),
                    safe_float(row.get('reply_count', row.get('avg_replies', 0)), 0),
                    safe_float(row.get('bookmark_count', row.get('avg_bookmarks', 0)), 0),
                    safe_float(row.get('utility_score', row.get('avg_utility_score', 0)), 0)
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
        # 注意: post_features 表不存储 view_count/like_count 等统计字段,
        # 这些字段需要从 CSV 文件或源数据库获取
        if session_id:
            cursor.execute(
                """
                SELECT co.tweet_id as id, co.author, co.text, co.outlier_type, co.score,
                       pf.utility_score, pf.asset_quadrant, pf.discussion_rate, pf.virality_rate
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
                       pf.utility_score, pf.asset_quadrant, pf.discussion_rate, pf.virality_rate
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
                   'utility_score', 'asset_quadrant', 'discussion_rate', 'virality_rate']
        
        # 先构建基础结果
        result = [dict(zip(columns, row)) for row in rows]
        
        # 从源数据库补充统计字段
        if result:
            tweet_ids = [str(r['id']) for r in result]
            stats_map = self._get_tweet_stats_from_source(tweet_ids)
            
            for r in result:
                tweet_id = str(r['id'])
                if tweet_id in stats_map:
                    r.update(stats_map[tweet_id])
                else:
                    # 默认值
                    r['view_count'] = 0
                    r['like_count'] = 0
                    r['bookmark_count'] = 0
                    r['reply_count'] = 0
        
        return result

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
        
        # 注意: post_features 表不存储 view_count/reply_count 等统计字段
        if session_id:
            cursor.execute(
                """
                SELECT co.tweet_id as id, co.author, co.text, co.outlier_type as opportunity_type,
                       pf.is_question, pf.discussion_rate, pf.utility_score
                FROM content_outliers co
                LEFT JOIN post_features pf ON co.tweet_id = pf.tweet_id AND co.session_id = pf.session_id
                WHERE co.session_id = %s 
                  AND (co.outlier_type IN ('unanswered_question', 'hot_debate') 
                       OR pf.is_question = TRUE)
                ORDER BY co.score DESC
                LIMIT %s
                """,
                (session_id, limit)
            )
        else:
            cursor.execute(
                """
                SELECT co.tweet_id as id, co.author, co.text, co.outlier_type as opportunity_type,
                       pf.is_question, pf.discussion_rate, pf.utility_score
                FROM content_outliers co
                LEFT JOIN post_features pf ON co.tweet_id = pf.tweet_id
                WHERE co.session_id = (
                    SELECT session_id FROM analysis_sessions 
                    WHERE status = 'completed' 
                    ORDER BY completed_at DESC LIMIT 1
                )
                AND (co.outlier_type IN ('unanswered_question', 'hot_debate') 
                     OR pf.is_question = TRUE)
                ORDER BY co.score DESC
                LIMIT %s
                """,
                (limit,)
            )
        
        rows = cursor.fetchall()
        cursor.close()
        
        columns = ['id', 'author', 'text', 'opportunity_type', 'is_question', 'discussion_rate', 'utility_score']
        
        # 先构建基础结果
        result = [dict(zip(columns, row)) for row in rows]
        
        # 从源数据库补充统计字段
        if result:
            tweet_ids = [str(r['id']) for r in result]
            stats_map = self._get_tweet_stats_from_source(tweet_ids)
            
            for r in result:
                tweet_id = str(r['id'])
                if tweet_id in stats_map:
                    r['reply_count'] = stats_map[tweet_id].get('reply_count', 0)
                    r['view_count'] = stats_map[tweet_id].get('view_count', 0)
                else:
                    r['reply_count'] = 0
                    r['view_count'] = 0
        
        return result

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
        
        # 注意: post_features 表不存储 view_count/like_count 等统计字段
        if session_id:
            cursor.execute(
                """
                SELECT pf.tweet_id, co.author, pf.funnel_signal, 
                       pf.utility_score, pf.virality_rate, SUBSTRING(co.text, 1, 100) as text_preview
                FROM post_features pf
                JOIN content_outliers co ON pf.tweet_id = co.tweet_id AND pf.session_id = co.session_id
                WHERE pf.session_id = %s 
                  AND pf.funnel_signal IS NOT NULL
                ORDER BY pf.utility_score DESC
                LIMIT %s
                """,
                (session_id, limit)
            )
        else:
            cursor.execute(
                """
                SELECT pf.tweet_id, co.author, pf.funnel_signal, 
                       pf.utility_score, pf.virality_rate, SUBSTRING(co.text, 1, 100) as text_preview
                FROM post_features pf
                JOIN content_outliers co ON pf.tweet_id = co.tweet_id
                WHERE pf.session_id = (
                    SELECT session_id FROM analysis_sessions 
                    WHERE status = 'completed' 
                    ORDER BY completed_at DESC LIMIT 1
                )
                AND pf.funnel_signal IS NOT NULL
                ORDER BY pf.utility_score DESC
                LIMIT %s
                """,
                (limit,)
            )
        
        rows = cursor.fetchall()
        cursor.close()
        
        columns = ['tweet_id', 'author', 'funnel_signal', 'utility_score', 'virality_rate', 'text_preview']
        
        # 先构建基础结果
        result = [dict(zip(columns, row)) for row in rows]
        
        # 从源数据库补充统计字段
        if result:
            tweet_ids = [str(r['tweet_id']) for r in result]
            stats_map = self._get_tweet_stats_from_source(tweet_ids)
            
            for r in result:
                tweet_id = str(r['tweet_id'])
                if tweet_id in stats_map:
                    r['view_count'] = stats_map[tweet_id].get('view_count', 0)
                    r['like_count'] = stats_map[tweet_id].get('like_count', 0)
                else:
                    r['view_count'] = 0
                    r['like_count'] = 0
        
        return result

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