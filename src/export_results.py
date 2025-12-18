#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析结果导出脚本
从 ANALYSIS_DATABASE_URL 数据库中导出分析结果到本地文件

功能特性:
    - 交互式菜单选择导出内容
    - 支持多种导出格式 (CSV, JSON, Markdown)
    - 彩色终端输出，友好的用户体验
    - 自动生成可读的分析报告

使用方法:
    python -m src.export_results                    # 交互式菜单
    python -m src.export_results --session SESSION_ID --output ./exports
    python -m src.export_results --latest --output ./exports
    python -m src.export_results --list             # 列出所有会话
    python -m src.export_results --all              # 导出所有数据
    python -m src.export_results --format markdown  # 指定导出格式
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =====================================================
# 终端颜色和样式
# =====================================================
class Colors:
    """终端颜色定义"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @classmethod
    def disable(cls):
        """在不支持颜色的终端中禁用颜色"""
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.YELLOW = ''
        cls.RED = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.UNDERLINE = ''


def print_banner():
    """打印欢迎横幅"""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        🔍  X_deepdive 分析结果导出工具  📊                        ║
║                                                                  ║
║   从分析数据库中提取并格式化输出所有深度分析结果                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
{Colors.ENDC}
"""
    print(banner)


def print_section(title: str, icon: str = "📁"):
    """打印分节标题"""
    print(f"\n{Colors.YELLOW}{Colors.BOLD}{icon} {title}{Colors.ENDC}")
    print(f"{Colors.YELLOW}{'─' * 60}{Colors.ENDC}")


def print_success(message: str):
    """打印成功消息"""
    print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")


def print_error(message: str):
    """打印错误消息"""
    print(f"{Colors.RED}❌ {message}{Colors.ENDC}")


def print_info(message: str):
    """打印信息消息"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.ENDC}")


def print_warning(message: str):
    """打印警告消息"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")


# =====================================================
# 配置加载
# =====================================================
def load_config():
    """加载配置文件"""
    try:
        from .config import config
        return config
    except ImportError:
        # 作为独立脚本运行时
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.config import config
        return config


# =====================================================
# 数据库连接
# =====================================================
def get_analysis_connection():
    """获取分析数据库连接"""
    # 优先从配置文件读取
    try:
        cfg = load_config()
        db_config = cfg.get_analysis_database_config()
        if db_config:
            import pymysql
            # 添加超时设置，避免在 CI 环境中无限等待
            db_config['connect_timeout'] = 10
            db_config['read_timeout'] = 60
            db_config['write_timeout'] = 60
            connection = pymysql.connect(**db_config)
            print_success(f"连接到分析数据库: {db_config['host']}:{db_config['port']}/{db_config['database']}")
            return connection
    except Exception as e:
        logger.debug(f"从配置文件读取失败: {e}")
    
    # 回退到环境变量
    db_uri = os.getenv('ANALYSIS_DATABASE_URL')
    
    if not db_uri:
        print_error("未配置分析数据库连接")
        print_info("请在 config.ini 中配置 ANALYSIS_DATABASE_URL 或设置环境变量")
        print_info("示例: export ANALYSIS_DATABASE_URL='mysql://user:pass@host:port/dbname'")
        return None
    
    import re
    pattern = r'mysql://([^:]+):([^@]+)@([^:]+):(\d+)/([^?]+)(\?.*)?'
    match = re.match(pattern, db_uri)
    
    if not match:
        print_error(f"无法解析数据库连接字符串")
        return None
    
    user, password, host, port, database, params = match.groups()
    
    try:
        import pymysql
        config = {
            'host': host,
            'port': int(port),
            'user': user,
            'password': password,
            'database': database,
            'charset': 'utf8mb4',
            'autocommit': True,
        }
        
        # 自动配置 SSL (TiDB Cloud 强制要求)
        if (params and 'ssl-mode=REQUIRED' in params) or 'tidbcloud.com' in host:
            config['ssl'] = {}
            # 尝试查找系统CA证书
            possible_paths = [
                '/etc/ssl/certs/ca-certificates.crt',  # Debian/Ubuntu
                '/etc/pki/tls/certs/ca-bundle.crt',    # Fedora/RHEL
                '/etc/ssl/cert.pem',                   # macOS
                '/usr/local/etc/openssl/cert.pem',     # macOS Homebrew
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config['ssl']['ca'] = path
                    break
        
        # 添加超时设置，避免在 CI 环境中无限等待
        config['connect_timeout'] = 10
        config['read_timeout'] = 60
        config['write_timeout'] = 60
        
        connection = pymysql.connect(**config)
        print_success(f"连接到分析数据库: {host}:{port}/{database}")
        return connection
    except ImportError:
        print_error("未安装 pymysql，请运行: pip install pymysql")
        return None
    except Exception as e:
        print_error(f"数据库连接失败: {e}")
        return None


# =====================================================
# 会话管理
# =====================================================
def list_sessions(connection) -> pd.DataFrame:
    """列出所有分析会话"""
    query = """
    SELECT 
        session_id,
        started_at,
        completed_at,
        status,
        TIMESTAMPDIFF(SECOND, started_at, COALESCE(completed_at, NOW())) as duration_seconds
    FROM analysis_sessions
    ORDER BY started_at DESC
    LIMIT 20
    """
    
    df = pd.read_sql(query, connection)
    return df


def get_latest_session(connection) -> Optional[str]:
    """获取最新完成的会话ID"""
    query = """
    SELECT session_id 
    FROM analysis_sessions 
    WHERE status = 'completed'
    ORDER BY completed_at DESC 
    LIMIT 1
    """
    
    df = pd.read_sql(query, connection)
    if df.empty:
        return None
    return df.iloc[0]['session_id']


def get_session_info(connection, session_id: str) -> Optional[Dict]:
    """获取会话详细信息"""
    query = """
    SELECT * FROM analysis_sessions WHERE session_id = %s
    """
    df = pd.read_sql(query, connection, params=[session_id])
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def get_session_stats(connection, session_id: str) -> Dict[str, int]:
    """获取会话数据统计"""
    stats = {}
    tables = [
        ('user_metrics', '用户指标'),
        ('community_stats', '社群统计'),
        ('strong_ties', '强互惠关系'),
        ('content_outliers', '高价值内容'),
        ('activity_stats', '活跃度统计'),
        ('potential_new_users', '潜在新用户'),
        ('llm_outputs', 'LLM输出记录'),
        ('user_strategy_dossiers', '用户策略画像'),
        ('content_blueprints', '爆款内容蓝图'),
        ('content_idea_bank', '内容创意库'),
        ('post_features', '推文特征'),
        ('content_efficiency', '内容效能')
    ]
    
    cursor = connection.cursor()
    for table, name in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE session_id = %s", [session_id])
            count = cursor.fetchone()[0]
            stats[name] = count
        except Exception:
            stats[name] = 0
    cursor.close()
    
    return stats


# =====================================================
# 数据导出函数
# =====================================================
def export_user_metrics(connection, session_id: str, output_dir: str, fmt: str = 'csv') -> Tuple[pd.DataFrame, List[str]]:
    """
    导出用户指标数据
    
    Returns:
        Tuple[DataFrame, List[str]]: 数据框和生成的文件列表
    """
    query = """
    SELECT 
        um.username,
        um.pagerank,
        um.betweenness,
        um.in_degree,
        um.community_id,
        um.talkativity_ratio,
        um.professionalism_index,
        um.avg_reply_latency_seconds,
        um.rising_star_velocity,
        um.avg_utility_score,
        um.category,
        um.analysis_timestamp
    FROM user_metrics um
    WHERE um.session_id = %s
    ORDER BY um.pagerank DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    files_created = []
    
    if not df.empty:
        # 完整数据
        filepath = os.path.join(output_dir, f'all_users_with_metrics.{fmt}')
        _save_dataframe(df, filepath, fmt)
        files_created.append(filepath)
        print_success(f"导出用户指标: {os.path.basename(filepath)} ({len(df)} 条)")
        
        # 权威枢纽清单 (Top PageRank)
        authorities = df.nlargest(50, 'pagerank')
        filepath = os.path.join(output_dir, f'watchlist_authorities.{fmt}')
        _save_dataframe(authorities, filepath, fmt)
        files_created.append(filepath)
        print_info(f"  └─ 权威枢纽 Top 50: {os.path.basename(filepath)}")
        
        # 破圈者清单 (Top Betweenness)
        connectors = df.nlargest(50, 'betweenness')
        filepath = os.path.join(output_dir, f'watchlist_connectors.{fmt}')
        _save_dataframe(connectors, filepath, fmt)
        files_created.append(filepath)
        print_info(f"  └─ 破圈者 Top 50: {os.path.basename(filepath)}")
        
        # 崛起新星清单 (Top Rising Star Velocity)
        rising_stars = df[df['rising_star_velocity'] > 0].nlargest(50, 'rising_star_velocity')
        if not rising_stars.empty:
            filepath = os.path.join(output_dir, f'watchlist_rising_stars.{fmt}')
            _save_dataframe(rising_stars, filepath, fmt)
            files_created.append(filepath)
            print_info(f"  └─ 崛起新星 Top 50: {os.path.basename(filepath)}")
    
    return df, files_created


def export_community_stats(connection, session_id: str, output_dir: str, fmt: str = 'csv') -> Tuple[pd.DataFrame, List[str]]:
    """导出社群统计数据"""
    query = """
    SELECT 
        community_id,
        member_count,
        avg_pagerank,
        avg_betweenness,
        total_followers,
        top_members_json,
        topic_keywords_json,
        analysis_timestamp
    FROM community_stats
    WHERE session_id = %s
    ORDER BY member_count DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    files_created = []
    
    if not df.empty:
        filepath = os.path.join(output_dir, f'community_stats.{fmt}')
        _save_dataframe(df, filepath, fmt)
        files_created.append(filepath)
        print_success(f"导出社群统计: {os.path.basename(filepath)} ({len(df)} 个社群)")
    
    return df, files_created


def export_strong_ties(connection, session_id: str, output_dir: str, fmt: str = 'csv') -> Tuple[pd.DataFrame, List[str]]:
    """导出强互惠关系数据"""
    query = """
    SELECT 
        user_a,
        user_b,
        interaction_weight,
        interaction_samples_json,
        relationship_type,
        analysis_timestamp
    FROM strong_ties
    WHERE session_id = %s
    ORDER BY interaction_weight DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    files_created = []
    
    if not df.empty:
        filepath = os.path.join(output_dir, f'list_interactions_strong_ties.{fmt}')
        _save_dataframe(df, filepath, fmt)
        files_created.append(filepath)
        print_success(f"导出强互惠关系: {os.path.basename(filepath)} ({len(df)} 对)")
    
    return df, files_created


def export_content_outliers(connection, session_id: str, output_dir: str, fmt: str = 'csv') -> Tuple[pd.DataFrame, List[str]]:
    """导出高价值内容数据"""
    query = """
    SELECT 
        tweet_id,
        author,
        text,
        created_at,
        outlier_type,
        score,
        analysis_timestamp
    FROM content_outliers
    WHERE session_id = %s
    ORDER BY score DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    files_created = []
    
    if not df.empty:
        # 所有异常内容
        filepath = os.path.join(output_dir, f'list_posts_outliers.{fmt}')
        _save_dataframe(df, filepath, fmt)
        files_created.append(filepath)
        print_success(f"导出高价值内容: {os.path.basename(filepath)} ({len(df)} 条)")
        
        # 按类型分类导出
        for outlier_type in df['outlier_type'].dropna().unique():
            type_df = df[df['outlier_type'] == outlier_type]
            type_name = _get_outlier_type_name(outlier_type)
            filepath = os.path.join(output_dir, f'list_posts_{outlier_type}.{fmt}')
            _save_dataframe(type_df, filepath, fmt)
            files_created.append(filepath)
            print_info(f"  └─ {type_name}: {os.path.basename(filepath)} ({len(type_df)} 条)")
    
    return df, files_created


def export_activity_stats(connection, session_id: str, output_dir: str, fmt: str = 'csv') -> Tuple[pd.DataFrame, List[str]]:
    """导出活跃度统计数据"""
    query = """
    SELECT 
        stat_type,
        time_key,
        activity_count,
        activity_percentage,
        analysis_timestamp
    FROM activity_stats
    WHERE session_id = %s
    ORDER BY stat_type, CAST(time_key AS UNSIGNED)
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    files_created = []
    
    if not df.empty:
        # 按统计类型分别导出
        for stat_type in df['stat_type'].unique():
            type_df = df[df['stat_type'] == stat_type]
            type_name = _get_stat_type_name(stat_type)
            filepath = os.path.join(output_dir, f'stats_{stat_type}.{fmt}')
            _save_dataframe(type_df, filepath, fmt)
            files_created.append(filepath)
            print_success(f"导出{type_name}: {os.path.basename(filepath)}")
    
    return df, files_created


def export_potential_new_users(connection, session_id: str, output_dir: str, fmt: str = 'csv') -> Tuple[pd.DataFrame, List[str]]:
    """导出潜在新用户数据"""
    query = """
    SELECT 
        username,
        weighted_reply_score,
        reply_count,
        avg_replier_pagerank,
        analysis_timestamp
    FROM potential_new_users
    WHERE session_id = %s
    ORDER BY weighted_reply_score DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    files_created = []
    
    if not df.empty:
        filepath = os.path.join(output_dir, f'watchlist_potential_new_users.{fmt}')
        _save_dataframe(df, filepath, fmt)
        files_created.append(filepath)
        print_success(f"导出潜在新用户: {os.path.basename(filepath)} ({len(df)} 个)")
    
    return df, files_created


def export_llm_outputs(connection, session_id: str, output_dir: str, fmt: str = 'json') -> Tuple[pd.DataFrame, List[str]]:
    """导出 LLM 输出记录"""
    query = """
    SELECT 
        task_type,
        target_id,
        model_used,
        prompt_tokens,
        completion_tokens,
        total_cost,
        parsed_output,
        created_at
    FROM llm_outputs
    WHERE session_id = %s
    ORDER BY created_at DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    files_created = []
    
    if not df.empty:
        # 按任务类型分组导出
        llm_insights_dir = os.path.join(output_dir, 'llm_insights')
        os.makedirs(llm_insights_dir, exist_ok=True)
        
        for task_type in df['task_type'].unique():
            type_df = df[df['task_type'] == task_type]
            
            # 提取并合并 parsed_output
            outputs = []
            for _, row in type_df.iterrows():
                parsed = row['parsed_output']
                if parsed:
                    if isinstance(parsed, str):
                        try:
                            parsed = json.loads(parsed)
                        except:
                            pass
                    if isinstance(parsed, dict):
                        parsed['_model_used'] = row['model_used']
                        parsed['_created_at'] = str(row['created_at'])
                        outputs.append(parsed)
            
            if outputs:
                task_name = _get_task_type_name(task_type)
                filepath = os.path.join(llm_insights_dir, f'{task_type}.json')
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(outputs, f, ensure_ascii=False, indent=2, default=str)
                files_created.append(filepath)
                print_info(f"  └─ {task_name}: {len(outputs)} 条")
        
        print_success(f"导出 LLM 洞察到 llm_insights/ 目录")
    
    return df, files_created


def export_user_strategy_dossiers(connection, session_id: str, output_dir: str, fmt: str = 'json') -> Tuple[pd.DataFrame, List[str]]:
    """导出用户策略画像"""
    query = """
    SELECT 
        username,
        core_identity,
        growth_tactics,
        monetization_model,
        content_style_summary,
        actionable_takeaways,
        model_used,
        created_at
    FROM user_strategy_dossiers
    WHERE session_id = %s
    ORDER BY created_at DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    files_created = []
    
    if not df.empty:
        llm_insights_dir = os.path.join(output_dir, 'llm_insights')
        os.makedirs(llm_insights_dir, exist_ok=True)
        
        filepath = os.path.join(llm_insights_dir, 'User_Strategy_Dossiers.json')
        _save_json_from_df(df, filepath, ['growth_tactics', 'actionable_takeaways'])
        files_created.append(filepath)
        print_success(f"导出用户策略画像: {os.path.basename(filepath)} ({len(df)} 条)")
    
    return df, files_created


def export_content_blueprints(connection, session_id: str, output_dir: str, fmt: str = 'json') -> Tuple[pd.DataFrame, List[str]]:
    """导出爆款内容蓝图"""
    query = """
    SELECT 
        source_tweet_id,
        quadrant,
        hook_style,
        body_structure,
        readability_features,
        emotional_tone,
        call_to_action,
        why_viral,
        replication_template,
        model_used,
        created_at
    FROM content_blueprints
    WHERE session_id = %s
    ORDER BY created_at DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    files_created = []
    
    if not df.empty:
        llm_insights_dir = os.path.join(output_dir, 'llm_insights')
        os.makedirs(llm_insights_dir, exist_ok=True)
        
        filepath = os.path.join(llm_insights_dir, 'viral_deconstruction.json')
        _save_json_from_df(df, filepath, ['hook_style', 'readability_features', 'call_to_action'])
        files_created.append(filepath)
        print_success(f"导出爆款内容蓝图: {os.path.basename(filepath)} ({len(df)} 条)")
    
    return df, files_created


def export_content_idea_bank(connection, session_id: str, output_dir: str, fmt: str = 'json') -> Tuple[pd.DataFrame, List[str]]:
    """导出内容创意库"""
    query = """
    SELECT 
        source_tweet_id,
        idea_type,
        topic,
        user_intent,
        suggested_angle,
        suggested_title,
        status,
        model_used,
        created_at
    FROM content_idea_bank
    WHERE session_id = %s
    ORDER BY created_at DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    files_created = []
    
    if not df.empty:
        llm_insights_dir = os.path.join(output_dir, 'llm_insights')
        os.makedirs(llm_insights_dir, exist_ok=True)
        
        filepath = os.path.join(llm_insights_dir, 'Content_Idea_Bank.json')
        _save_json_from_df(df, filepath, [])
        files_created.append(filepath)
        print_success(f"导出内容创意库: {os.path.basename(filepath)} ({len(df)} 条)")
    
    return df, files_created


def export_post_features(connection, session_id: str, output_dir: str, fmt: str = 'csv') -> Tuple[pd.DataFrame, List[str]]:
    """导出推文特征数据"""
    query = """
    SELECT 
        tweet_id,
        conversation_id,
        utility_score,
        discussion_rate,
        virality_rate,
        is_question,
        sentiment_score,
        asset_quadrant,
        thread_retention_rate,
        thread_length,
        funnel_signal,
        analysis_timestamp
    FROM post_features
    WHERE session_id = %s
    ORDER BY utility_score DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    files_created = []
    
    if not df.empty:
        filepath = os.path.join(output_dir, f'post_features.{fmt}')
        _save_dataframe(df, filepath, fmt)
        files_created.append(filepath)
        print_success(f"导出推文特征: {os.path.basename(filepath)} ({len(df)} 条)")
        
        # 内容资产四象限分析
        if 'asset_quadrant' in df.columns:
            quadrant_stats = df['asset_quadrant'].value_counts().to_dict()
            if quadrant_stats:
                print_info(f"  └─ 内容四象限分布: {quadrant_stats}")
    
    return df, files_created


def export_content_efficiency(connection, session_id: str, output_dir: str, fmt: str = 'csv') -> Tuple[pd.DataFrame, List[str]]:
    """导出内容效能统计"""
    query = """
    SELECT 
        media_type,
        post_count,
        avg_views,
        avg_likes,
        avg_replies,
        avg_bookmarks,
        avg_utility_score,
        analysis_timestamp
    FROM content_efficiency
    WHERE session_id = %s
    ORDER BY post_count DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    files_created = []
    
    if not df.empty:
        filepath = os.path.join(output_dir, f'stats_content_efficiency.{fmt}')
        _save_dataframe(df, filepath, fmt)
        files_created.append(filepath)
        print_success(f"导出内容效能统计: {os.path.basename(filepath)} ({len(df)} 种媒体类型)")
    
    return df, files_created


# =====================================================
# 辅助函数
# =====================================================
def _save_dataframe(df: pd.DataFrame, filepath: str, fmt: str):
    """保存 DataFrame 到文件"""
    if fmt == 'csv':
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
    elif fmt == 'json':
        df.to_json(filepath, orient='records', force_ascii=False, indent=2)
    elif fmt == 'markdown':
        with open(filepath.replace('.markdown', '.md'), 'w', encoding='utf-8') as f:
            f.write(df.to_markdown(index=False))
    else:
        df.to_csv(filepath, index=False, encoding='utf-8-sig')


def _save_json_from_df(df: pd.DataFrame, filepath: str, json_columns: List[str]):
    """将 DataFrame 保存为 JSON，自动解析 JSON 字段"""
    records = []
    for _, row in df.iterrows():
        record = row.to_dict()
        for col in json_columns:
            if col in record and record[col]:
                if isinstance(record[col], str):
                    try:
                        record[col] = json.loads(record[col])
                    except:
                        pass
        records.append(record)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)


def _get_outlier_type_name(outlier_type: str) -> str:
    """获取异常类型的中文名称"""
    names = {
        'high_utility': '高干货内容',
        'high_traffic': '高流量内容',
        'high_discussion': '高讨论内容',
        'unanswered_question': '未回答问题',
        'hot_debate': '热议话题'
    }
    return names.get(outlier_type, outlier_type)


def _get_stat_type_name(stat_type: str) -> str:
    """获取统计类型的中文名称"""
    names = {
        'hourly_heatmap': '小时活跃热力图',
        'daily_trend': '日活跃趋势',
        'weekly_pattern': '周活跃模式'
    }
    return names.get(stat_type, stat_type)


def _get_task_type_name(task_type: str) -> str:
    """获取任务类型的中文名称"""
    names = {
        'viral_deconstruction': '爆款内容拆解',
        'user_profiling': '用户策略画像',
        'relationship_insight': '关系洞察',
        'content_opportunity': '内容机会挖掘',
        'thread_analysis': 'Thread分析',
        'monetization_analysis': '变现模式分析'
    }
    return names.get(task_type, task_type)


def generate_summary_report(session_info: Dict, stats: Dict, output_dir: str, files_created: List[str]) -> str:
    """生成会话摘要报告 (Markdown 格式)"""
    report_lines = []
    
    # 标题
    session_id = session_info.get('session_id', 'Unknown')
    report_lines.append(f"# 🔍 X_deepdive 分析结果报告\n")
    report_lines.append(f"**会话 ID**: `{session_id}`\n")
    report_lines.append(f"**导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 会话信息
    report_lines.append("\n## 📊 会话信息\n")
    report_lines.append(f"| 属性 | 值 |")
    report_lines.append(f"|------|-----|")
    report_lines.append(f"| 开始时间 | {session_info.get('started_at', 'N/A')} |")
    report_lines.append(f"| 完成时间 | {session_info.get('completed_at', 'N/A')} |")
    report_lines.append(f"| 状态 | {session_info.get('status', 'N/A')} |")
    
    # 数据统计
    report_lines.append("\n## 📈 数据统计\n")
    report_lines.append(f"| 数据类型 | 数量 |")
    report_lines.append(f"|----------|------|")
    for name, count in stats.items():
        if count > 0:
            report_lines.append(f"| {name} | {count:,} |")
    
    # 导出文件清单
    report_lines.append("\n## 📁 导出文件清单\n")
    
    # 按目录分组
    csv_files = [f for f in files_created if f.endswith('.csv')]
    json_files = [f for f in files_created if f.endswith('.json')]
    other_files = [f for f in files_created if not f.endswith('.csv') and not f.endswith('.json')]
    
    if csv_files:
        report_lines.append("\n### CSV 文件 (结构化数据)\n")
        for f in csv_files:
            report_lines.append(f"- `{os.path.basename(f)}`")
    
    if json_files:
        report_lines.append("\n### JSON 文件 (LLM 洞察)\n")
        for f in json_files:
            report_lines.append(f"- `{os.path.basename(f)}`")
    
    if other_files:
        report_lines.append("\n### 其他文件\n")
        for f in other_files:
            report_lines.append(f"- `{os.path.basename(f)}`")
    
    # 数据说明
    report_lines.append("\n## 📖 数据说明\n")
    report_lines.append("""
### 用户分析文件

- **`all_users_with_metrics.csv`**: 所有用户的完整指标数据
- **`watchlist_authorities.csv`**: 权威枢纽 Top 50 (按 PageRank 排序)
- **`watchlist_connectors.csv`**: 破圈者 Top 50 (按 Betweenness 排序)
- **`watchlist_rising_stars.csv`**: 崛起新星 Top 50 (按增长速度排序)
- **`watchlist_potential_new_users.csv`**: 值得关注的潜在新用户

### 内容分析文件

- **`list_posts_outliers.csv`**: 高价值异常内容汇总
- **`list_posts_high_utility.csv`**: 高干货内容 (收藏/点赞比高)
- **`list_posts_unanswered_question.csv`**: 未被充分回答的问题 (内容机会)

### 社群分析文件

- **`community_stats.csv`**: 社群统计数据
- **`list_interactions_strong_ties.csv`**: 强互惠关系对

### LLM 洞察文件 (llm_insights/)

- **`User_Strategy_Dossiers.json`**: 成功用户的策略画像
- **`Content_Blueprints.json`**: 爆款内容的逆向工程分析
- **`Content_Idea_Bank.json`**: 内容创意和选题库

详细字段说明请参考 `docs/EXPORT_RESULTS_GUIDE.md`
""")
    
    report_content = '\n'.join(report_lines)
    
    # 保存报告
    report_path = os.path.join(output_dir, 'EXPORT_SUMMARY.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_path


# =====================================================
# 交互式菜单
# =====================================================
def show_interactive_menu(connection, session_id: str) -> List[str]:
    """显示交互式导出菜单"""
    print_section("选择要导出的数据类型", "📋")
    
    options = [
        ('1', '用户指标数据', 'user_metrics', '包含 PageRank, Betweenness, 社群ID 等'),
        ('2', '社群统计数据', 'community', '各社群的成员数、核心成员等'),
        ('3', '强互惠关系', 'ties', '高频互动的用户对'),
        ('4', '高价值内容', 'outliers', '干货内容、热议话题、未回答问题等'),
        ('5', '活跃度统计', 'activity', '按小时/天的活跃分布'),
        ('6', '潜在新用户', 'potential', '值得关注的新用户'),
        ('7', 'LLM 洞察报告', 'llm', '策略画像、爆款蓝图、创意库'),
        ('8', '推文特征数据', 'features', '内容四象限、Thread分析等'),
        ('9', '内容效能统计', 'efficiency', '不同媒体类型的效能对比'),
        ('A', '全部导出', 'all', '导出以上所有数据'),
    ]
    
    for opt, name, _, desc in options:
        print(f"  {Colors.CYAN}[{opt}]{Colors.ENDC} {name}")
        print(f"      {Colors.BLUE}{desc}{Colors.ENDC}")
    
    print(f"\n  {Colors.CYAN}[Q]{Colors.ENDC} 退出")
    
    print(f"\n{Colors.YELLOW}请输入选项 (多选用逗号分隔，如 1,2,3): {Colors.ENDC}", end='')
    
    try:
        choice = input().strip().upper()
    except (EOFError, KeyboardInterrupt):
        return []
    
    if choice == 'Q':
        return []
    
    if choice == 'A':
        return ['all']
    
    selected = []
    for c in choice.split(','):
        c = c.strip()
        for opt, _, key, _ in options:
            if c == opt:
                selected.append(key)
                break
    
    return selected if selected else ['all']


def display_session_selector(connection) -> Optional[str]:
    """显示会话选择器"""
    sessions_df = list_sessions(connection)
    
    if sessions_df.empty:
        print_error("没有找到任何分析会话")
        return None
    
    print_section("选择要导出的分析会话", "📅")
    
    print(f"\n{'序号':<4} {'会话ID':<20} {'开始时间':<20} {'状态':<12}")
    print("-" * 60)
    
    for idx, row in sessions_df.iterrows():
        status_color = Colors.GREEN if row['status'] == 'completed' else Colors.YELLOW
        print(f"{idx+1:<4} {row['session_id']:<20} {str(row['started_at']):<20} {status_color}{row['status']:<12}{Colors.ENDC}")
    
    print(f"\n{Colors.YELLOW}输入序号选择会话 (直接回车选择最新的): {Colors.ENDC}", end='')
    
    try:
        choice = input().strip()
    except (EOFError, KeyboardInterrupt):
        return None
    
    if not choice:
        # 返回最新的已完成会话
        completed = sessions_df[sessions_df['status'] == 'completed']
        if completed.empty:
            return sessions_df.iloc[0]['session_id']
        return completed.iloc[0]['session_id']
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(sessions_df):
            return sessions_df.iloc[idx]['session_id']
    except ValueError:
        # 可能直接输入了会话ID
        if choice in sessions_df['session_id'].values:
            return choice
    
    print_warning("无效的选择，使用最新会话")
    return sessions_df.iloc[0]['session_id']


def select_output_format() -> str:
    """选择输出格式"""
    print(f"\n{Colors.YELLOW}选择导出格式 [csv/json] (默认 csv): {Colors.ENDC}", end='')
    
    try:
        choice = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return 'csv'
    
    return choice if choice in ['csv', 'json'] else 'csv'


# =====================================================
# 主导出函数
# =====================================================
def export_all(session_id: str, output_dir: str, fmt: str = 'csv', 
               export_types: Optional[List[str]] = None) -> bool:
    """
    导出所有数据
    
    Args:
        session_id: 会话ID
        output_dir: 输出目录
        fmt: 导出格式 (csv/json)
        export_types: 要导出的数据类型列表，None 表示全部
    """
    connection = get_analysis_connection()
    if not connection:
        print_error("无法连接到分析数据库")
        return False
    
    try:
        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_subdir = os.path.join(output_dir, f'export_{session_id}_{timestamp}')
        os.makedirs(output_subdir, exist_ok=True)
        
        print_section(f"开始导出分析会话: {session_id}", "🚀")
        print_info(f"输出目录: {output_subdir}")
        print_info(f"导出格式: {fmt.upper()}")
        
        all_files_created = []
        export_all_types = export_types is None or 'all' in export_types
        
        # 获取会话信息和统计
        session_info = get_session_info(connection, session_id) or {'session_id': session_id}
        stats = get_session_stats(connection, session_id)
        
        # 按类型导出
        if export_all_types or 'user_metrics' in export_types:
            print_section("用户指标数据", "👥")
            _, files = export_user_metrics(connection, session_id, output_subdir, fmt)
            all_files_created.extend(files)
        
        if export_all_types or 'community' in export_types:
            print_section("社群统计数据", "🏘️")
            _, files = export_community_stats(connection, session_id, output_subdir, fmt)
            all_files_created.extend(files)
        
        if export_all_types or 'ties' in export_types:
            print_section("强互惠关系", "🤝")
            _, files = export_strong_ties(connection, session_id, output_subdir, fmt)
            all_files_created.extend(files)
        
        if export_all_types or 'outliers' in export_types:
            print_section("高价值内容", "💎")
            _, files = export_content_outliers(connection, session_id, output_subdir, fmt)
            all_files_created.extend(files)
        
        if export_all_types or 'activity' in export_types:
            print_section("活跃度统计", "📈")
            _, files = export_activity_stats(connection, session_id, output_subdir, fmt)
            all_files_created.extend(files)
        
        if export_all_types or 'potential' in export_types:
            print_section("潜在新用户", "🌟")
            _, files = export_potential_new_users(connection, session_id, output_subdir, fmt)
            all_files_created.extend(files)
        
        if export_all_types or 'llm' in export_types:
            print_section("LLM 洞察报告", "🤖")
            _, files = export_llm_outputs(connection, session_id, output_subdir)
            all_files_created.extend(files)
            _, files = export_user_strategy_dossiers(connection, session_id, output_subdir)
            all_files_created.extend(files)
            _, files = export_content_blueprints(connection, session_id, output_subdir)
            all_files_created.extend(files)
            _, files = export_content_idea_bank(connection, session_id, output_subdir)
            all_files_created.extend(files)
        
        if export_all_types or 'features' in export_types:
            print_section("推文特征数据", "📝")
            _, files = export_post_features(connection, session_id, output_subdir, fmt)
            all_files_created.extend(files)
        
        if export_all_types or 'efficiency' in export_types:
            print_section("内容效能统计", "📊")
            _, files = export_content_efficiency(connection, session_id, output_subdir, fmt)
            all_files_created.extend(files)
        
        # 生成汇总报告
        print_section("生成导出报告", "📋")
        report_path = generate_summary_report(session_info, stats, output_subdir, all_files_created)
        print_success(f"生成汇总报告: {os.path.basename(report_path)}")
        
        # 打印完成信息
        print(f"\n{Colors.GREEN}{Colors.BOLD}")
        print("═" * 60)
        print(f"  ✅ 导出完成!")
        print(f"  📁 输出目录: {output_subdir}")
        print(f"  📄 文件数量: {len(all_files_created) + 1}")
        print("═" * 60)
        print(f"{Colors.ENDC}")
        
        return True
    
    except Exception as e:
        print_error(f"导出失败: {e}")
        logger.exception("导出过程中发生错误")
        return False
    
    finally:
        connection.close()


# =====================================================
# 主函数
# =====================================================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='X_deepdive 分析结果导出工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.export_results                    # 交互式菜单
  python -m src.export_results --list             # 列出所有会话
  python -m src.export_results --latest           # 导出最新会话
  python -m src.export_results --session 20231201_120000 --output ./my_exports
  python -m src.export_results --all --format json
        """
    )
    parser.add_argument('--session', type=str, help='指定要导出的会话ID')
    parser.add_argument('--latest', action='store_true', help='导出最新完成的会话')
    parser.add_argument('--list', action='store_true', help='列出所有分析会话')
    parser.add_argument('--all', action='store_true', help='非交互式导出所有数据')
    parser.add_argument('--output', type=str, default='./exports', help='输出目录 (默认: ./exports)')
    parser.add_argument('--format', type=str, choices=['csv', 'json'], default='csv', 
                       help='导出格式 (默认: csv)')
    parser.add_argument('--no-color', action='store_true', help='禁用彩色输出')
    
    args = parser.parse_args()
    
    # 处理颜色
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()
    
    # 打印欢迎横幅
    print_banner()
    
    # 连接数据库
    connection = get_analysis_connection()
    if not connection:
        print_error("无法连接到分析数据库")
        print_info("请检查 config.ini 中的 ANALYSIS_DATABASE_URL 配置")
        print_info("或设置环境变量: export ANALYSIS_DATABASE_URL='mysql://user:pass@host:port/dbname'")
        sys.exit(1)
    
    try:
        # 列出会话模式
        if args.list:
            print_section("分析会话列表", "📅")
            sessions = list_sessions(connection)
            if sessions.empty:
                print_warning("没有找到任何分析会话")
            else:
                print(sessions.to_string(index=False))
                print(f"\n{Colors.BLUE}共 {len(sessions)} 个会话{Colors.ENDC}")
            return
        
        # 确定会话ID
        session_id = args.session
        
        if args.latest:
            session_id = get_latest_session(connection)
            if not session_id:
                print_error("没有找到已完成的分析会话")
                sys.exit(1)
            print_info(f"使用最新会话: {session_id}")
        
        # 交互式模式
        if not session_id and not args.all:
            session_id = display_session_selector(connection)
            if not session_id:
                print_info("已取消操作")
                return
        
        # 如果仍然没有会话ID，尝试获取最新的
        if not session_id:
            session_id = get_latest_session(connection)
            if not session_id:
                print_error("没有找到任何可用的分析会话")
                sys.exit(1)
        
        # 显示会话统计
        stats = get_session_stats(connection, session_id)
        print_section(f"会话 {session_id} 数据概览", "📊")
        for name, count in stats.items():
            if count > 0:
                print(f"  {name}: {Colors.CYAN}{count:,}{Colors.ENDC} 条")
        
        # 非交互式全量导出
        if args.all:
            connection.close()
            success = export_all(session_id, args.output, args.format, ['all'])
            sys.exit(0 if success else 1)
        
        # 交互式选择导出类型
        export_types = show_interactive_menu(connection, session_id)
        if not export_types:
            print_info("已取消操作")
            return
        
        # 选择导出格式
        fmt = select_output_format() if not args.format else args.format
        
        connection.close()
        
        # 执行导出
        success = export_all(session_id, args.output, fmt, export_types)
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}操作已取消{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print_error(f"操作失败: {e}")
        logger.exception("发生错误")
        sys.exit(1)
    finally:
        if connection and connection.open:
            connection.close()


if __name__ == '__main__':
    main()
