#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析结果导出脚本
从 ANALYSIS_DATABASE_URL 数据库中导出分析结果到本地文件

使用方法:
    python -m src.export_results --session SESSION_ID --output ./exports
    python -m src.export_results --latest --output ./exports
    python -m src.export_results --list  # 列出所有会话
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_analysis_connection():
    """获取分析数据库连接"""
    db_uri = os.getenv('ANALYSIS_DATABASE_URL')
    
    if not db_uri:
        logger.error("未配置 ANALYSIS_DATABASE_URL 环境变量")
        return None
    
    import re
    pattern = r'mysql://([^:]+):([^@]+)@([^:]+):(\d+)/([^?]+)(\?.*)?'
    match = re.match(pattern, db_uri)
    
    if not match:
        logger.error(f"无法解析 ANALYSIS_DATABASE_URL: {db_uri}")
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
        
        if params and 'ssl-mode=REQUIRED' in params:
            config['ssl'] = {'ssl_mode': 'REQUIRED'}
        
        connection = pymysql.connect(**config)
        logger.info(f"连接到分析数据库: {host}:{port}/{database}")
        return connection
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        return None


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


def export_user_metrics(connection, session_id: str, output_dir: str):
    """导出用户指标数据"""
    query = """
    SELECT 
        username, name, bio, followers_count, following_count, tweets_count,
        verified, verified_type, created_at, account_age_days,
        pagerank, betweenness, in_degree, community_id,
        talkativity_ratio, professionalism_index, avg_reply_latency_seconds,
        rising_star_velocity, avg_utility_score, category
    FROM user_metrics
    WHERE session_id = %s
    ORDER BY pagerank DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    
    if not df.empty:
        # 完整数据
        filepath = os.path.join(output_dir, 'all_users_with_metrics.csv')
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"导出用户指标: {filepath} ({len(df)} 条)")
        
        # 权威枢纽清单
        authorities = df.sort_values('pagerank', ascending=False)
        filepath = os.path.join(output_dir, 'watchlist_authorities.csv')
        authorities.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"导出权威枢纽清单: {filepath}")
        
        # 破圈者清单
        connectors = df.sort_values('betweenness', ascending=False)
        filepath = os.path.join(output_dir, 'watchlist_connectors.csv')
        connectors.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"导出破圈者清单: {filepath}")
        
        # 崛起新星清单
        rising_stars = df[df['account_age_days'] < 730].sort_values('rising_star_velocity', ascending=False)
        filepath = os.path.join(output_dir, 'watchlist_rising_stars.csv')
        rising_stars.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"导出崛起新星清单: {filepath}")
    
    return df


def export_community_stats(connection, session_id: str, output_dir: str):
    """导出社群统计数据"""
    query = """
    SELECT 
        community_id, member_count, avg_pagerank, avg_betweenness,
        total_followers, top_members_json, topic_keywords_json
    FROM community_stats
    WHERE session_id = %s
    ORDER BY member_count DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    
    if not df.empty:
        filepath = os.path.join(output_dir, 'community_stats.csv')
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"导出社群统计: {filepath} ({len(df)} 个社群)")
    
    return df


def export_strong_ties(connection, session_id: str, output_dir: str):
    """导出强互惠关系数据"""
    query = """
    SELECT 
        user_a, user_b, interaction_weight, interaction_samples_json, relationship_type
    FROM strong_ties
    WHERE session_id = %s
    ORDER BY interaction_weight DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    
    if not df.empty:
        filepath = os.path.join(output_dir, 'list_interactions_strong_ties.csv')
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"导出强互惠关系: {filepath} ({len(df)} 对)")
    
    return df


def export_content_outliers(connection, session_id: str, output_dir: str):
    """导出高价值内容数据"""
    query = """
    SELECT 
        tweet_id, author, text, created_at, content_type, media_type,
        view_count, like_count, bookmark_count, reply_count, retweet_count,
        utility_score, discussion_rate, outlier_type
    FROM content_outliers
    WHERE session_id = %s
    ORDER BY utility_score DESC, view_count DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    
    if not df.empty:
        # 所有异常内容
        filepath = os.path.join(output_dir, 'list_posts_outliers.csv')
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"导出高价值内容: {filepath} ({len(df)} 条)")
        
        # 内容机会
        opportunities = df[df['outlier_type'].isin(['unanswered_question', 'hot_debate'])]
        if not opportunities.empty:
            filepath = os.path.join(output_dir, 'list_content_opportunities.csv')
            opportunities.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"导出内容机会: {filepath} ({len(opportunities)} 条)")
    
    return df


def export_activity_stats(connection, session_id: str, output_dir: str):
    """导出活跃度统计数据"""
    query = """
    SELECT 
        stat_type, time_key as hour, activity_count, activity_percentage
    FROM activity_stats
    WHERE session_id = %s AND stat_type = 'hourly_heatmap'
    ORDER BY CAST(time_key AS UNSIGNED)
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    
    if not df.empty:
        filepath = os.path.join(output_dir, 'stats_activity_heatmap.csv')
        df[['hour', 'activity_count', 'activity_percentage']].to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"导出活跃度热力图: {filepath}")
    
    return df


def export_potential_new_users(connection, session_id: str, output_dir: str):
    """导出潜在新用户数据"""
    query = """
    SELECT 
        username as Username, 
        weighted_reply_score as WeightedReplyScore,
        reply_count as ReplyCount,
        avg_replier_pagerank as AvgReplierPageRank
    FROM potential_new_users
    WHERE session_id = %s
    ORDER BY weighted_reply_score DESC
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    
    if not df.empty:
        filepath = os.path.join(output_dir, 'watchlist_potential_new_users.csv')
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"导出潜在新用户: {filepath} ({len(df)} 个)")
    
    return df


def export_session_summary(connection, session_id: str, output_dir: str):
    """导出会话摘要"""
    query = """
    SELECT * FROM analysis_sessions WHERE session_id = %s
    """
    
    df = pd.read_sql(query, connection, params=[session_id])
    
    if not df.empty:
        summary = df.iloc[0].to_dict()
        
        # 统计各表数据量
        stats = {}
        tables = ['user_metrics', 'community_stats', 'strong_ties', 'content_outliers', 
                  'activity_stats', 'potential_new_users']
        
        cursor = connection.cursor()
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE session_id = %s", [session_id])
            stats[table] = cursor.fetchone()[0]
        cursor.close()
        
        summary['export_stats'] = stats
        summary['export_time'] = datetime.now().isoformat()
        
        filepath = os.path.join(output_dir, 'session_summary.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"导出会话摘要: {filepath}")
    
    return df


def export_all(session_id: str, output_dir: str):
    """导出所有数据"""
    connection = get_analysis_connection()
    if not connection:
        logger.error("无法连接到分析数据库")
        return False
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'=' * 60}")
        print(f"导出分析会话: {session_id}")
        print(f"输出目录: {output_dir}")
        print(f"{'=' * 60}\n")
        
        # 导出各类数据
        export_session_summary(connection, session_id, output_dir)
        export_user_metrics(connection, session_id, output_dir)
        export_community_stats(connection, session_id, output_dir)
        export_strong_ties(connection, session_id, output_dir)
        export_content_outliers(connection, session_id, output_dir)
        export_activity_stats(connection, session_id, output_dir)
        export_potential_new_users(connection, session_id, output_dir)
        
        print(f"\n{'=' * 60}")
        print(f"导出完成！所有文件保存在: {output_dir}")
        print(f"{'=' * 60}")
        
        return True
    
    except Exception as e:
        logger.error(f"导出失败: {e}")
        return False
    
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description='导出分析结果到本地文件')
    parser.add_argument('--session', type=str, help='要导出的会话ID')
    parser.add_argument('--latest', action='store_true', help='导出最新完成的会话')
    parser.add_argument('--list', action='store_true', help='列出所有会话')
    parser.add_argument('--output', type=str, default='./exports', help='输出目录 (默认: ./exports)')
    
    args = parser.parse_args()
    
    connection = get_analysis_connection()
    if not connection:
        logger.error("无法连接到分析数据库，请检查 ANALYSIS_DATABASE_URL 环境变量")
        sys.exit(1)
    
    try:
        if args.list:
            # 列出所有会话
            sessions = list_sessions(connection)
            if sessions.empty:
                print("没有找到任何分析会话")
            else:
                print("\n最近的分析会话:")
                print("-" * 80)
                print(sessions.to_string(index=False))
                print("-" * 80)
            return
        
        # 确定要导出的会话ID
        session_id = args.session
        
        if args.latest or not session_id:
            session_id = get_latest_session(connection)
            if not session_id:
                logger.error("没有找到已完成的分析会话")
                sys.exit(1)
            logger.info(f"使用最新会话: {session_id}")
        
        connection.close()
        
        # 执行导出
        success = export_all(session_id, args.output)
        sys.exit(0 if success else 1)
    
    except Exception as e:
        logger.error(f"操作失败: {e}")
        sys.exit(1)
    
    finally:
        if connection and connection.open:
            connection.close()


if __name__ == '__main__':
    main()
