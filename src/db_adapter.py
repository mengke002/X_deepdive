"""
数据库适配器模块
从 MySQL 数据库加载 Twitter 数据，适配分析脚本所需的数据格式
"""
import os
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import pandas as pd

# 抑制 pandas 关于非 SQLAlchemy 连接的警告
warnings.filterwarnings('ignore', message='.*pandas only supports SQLAlchemy.*')

logger = logging.getLogger(__name__)


class DatabaseAdapter:
    """数据库适配器，负责从数据库加载数据并转换为分析脚本所需的格式"""
    
    def __init__(self):
        self.connection = None
        self._connect()
    
    def _get_database_config(self) -> Dict[str, Any]:
        """获取数据库配置"""
        
        def _configure_ssl(config, force_ssl=False):
            """配置SSL并自动查找CA证书"""
            if force_ssl:
                config['ssl'] = {}
                # 尝试查找系统CA证书
                # GitHub Actions (Ubuntu) 通常在 /etc/ssl/certs/ca-certificates.crt
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
            return config

        db_uri = os.getenv('DATABASE_URL')
        
        if db_uri:
            import re
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
                
                force_ssl = (params and 'ssl-mode=REQUIRED' in params) or 'tidbcloud.com' in host
                return _configure_ssl(config, force_ssl)
        
        # 从单独的环境变量读取
        db_host = os.getenv('DB_HOST', 'localhost')
        config = {
            'host': db_host,
            'port': int(os.getenv('DB_PORT', '3306')),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'twitter'),
            'charset': 'utf8mb4',
            'autocommit': True,
        }
        
        force_ssl = os.getenv('DB_SSL', 'false').lower() == 'true' or 'tidbcloud.com' in db_host
        return _configure_ssl(config, force_ssl)
    
    def _connect(self):
        """建立数据库连接"""
        try:
            import pymysql
            config = self._get_database_config()
            
            # 移除 None 值的 ssl 配置
            if config.get('ssl') is None:
                config.pop('ssl', None)
            
            # 添加超时设置，避免在 CI 环境中无限等待
            config['connect_timeout'] = 10  # 连接超时 10 秒
            config['read_timeout'] = 60     # 读取超时 60 秒
            config['write_timeout'] = 60    # 写入超时 60 秒
            
            self.connection = pymysql.connect(**config)
            logger.info(f"数据库连接成功: {config['host']}:{config['port']}/{config['database']}")
        except ImportError:
            logger.error("未安装 pymysql，请运行: pip install pymysql")
            raise
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def _ensure_connection(self):
        """确保数据库连接有效"""
        if self.connection is None or not self.connection.open:
            self._connect()
    
    def close(self):
        """关闭数据库连接"""
        if self.connection and self.connection.open:
            self.connection.close()
            logger.info("数据库连接已关闭")
    
    def get_all_users(self) -> pd.DataFrame:
        """
        获取所有用户数据
        
        Returns:
            DataFrame with columns matching the original Following CSV format:
            User ID, Name, Username, Bio, Tweets Count, Followers Count, Following Count,
            Favourites Count, Media Count, Professional, Location, Website, Verified,
            Is Blue Verified, Verified Type, Created At, Profile URL, Avatar URL, Profile Banner URL
        """
        self._ensure_connection()
        
        query = """
        SELECT 
            user_id AS `User ID`,
            name AS `Name`,
            username AS `Username`,
            bio AS `Bio`,
            tweets_count AS `Tweets Count`,
            followers_count AS `Followers Count`,
            following_count AS `Following Count`,
            favourites_count AS `Favourites Count`,
            media_count AS `Media Count`,
            professional AS `Professional`,
            location AS `Location`,
            website AS `Website`,
            verified AS `Verified`,
            is_blue_verified AS `Is Blue Verified`,
            verified_type AS `Verified Type`,
            DATE_FORMAT(twitter_created_at, '%Y-%m-%d %H:%i:%s') AS `Created At`,
            profile_url AS `Profile URL`,
            avatar_url AS `Avatar URL`,
            banner_url AS `Profile Banner URL`
        FROM twitter_users
        """
        
        df = pd.read_sql(query, self.connection)
        logger.info(f"从数据库加载了 {len(df)} 个用户")
        return df
    
    def get_followings(self) -> List[Tuple[str, pd.DataFrame]]:
        """
        获取所有关注关系数据，按 source_user 分组
        模拟原有的 Following CSV 文件结构
        
        Returns:
            List of (source_username, DataFrame) tuples
            每个 DataFrame 包含 source_user 关注的所有用户信息
        """
        self._ensure_connection()
        
        # 获取所有有关注关系的 source 用户
        source_users_query = """
        SELECT DISTINCT u.username
        FROM twitter_followings f
        JOIN twitter_users u ON f.source_user_id = u.id
        """
        
        source_users_df = pd.read_sql(source_users_query, self.connection)
        source_users = source_users_df['username'].tolist()
        
        results = []
        
        for source_user in source_users:
            query = """
            SELECT 
                t.user_id AS `User ID`,
                t.name AS `Name`,
                t.username AS `Username`,
                t.bio AS `Bio`,
                t.tweets_count AS `Tweets Count`,
                t.followers_count AS `Followers Count`,
                t.following_count AS `Following Count`,
                t.favourites_count AS `Favourites Count`,
                t.media_count AS `Media Count`,
                t.professional AS `Professional`,
                t.location AS `Location`,
                t.website AS `Website`,
                t.verified AS `Verified`,
                t.is_blue_verified AS `Is Blue Verified`,
                t.verified_type AS `Verified Type`,
                DATE_FORMAT(t.twitter_created_at, '%%Y-%%m-%%d %%H:%%i:%%s') AS `Created At`,
                t.profile_url AS `Profile URL`,
                t.avatar_url AS `Avatar URL`,
                t.banner_url AS `Profile Banner URL`
            FROM twitter_followings f
            JOIN twitter_users s ON f.source_user_id = s.id
            JOIN twitter_users t ON f.target_user_id = t.id
            WHERE s.username = %s
            """
            
            df = pd.read_sql(query, self.connection, params=[source_user])
            if not df.empty:
                results.append((source_user, df))
        
        logger.info(f"从数据库加载了 {len(results)} 个用户的关注列表")
        return results
    
    def get_posts_with_replies(self) -> List[Tuple[str, pd.DataFrame]]:
        """
        获取所有推文和回复数据，按作者分组
        重要：利用 in_reply_to_tweet_id 重建对话关系
        
        Returns:
            List of (author_username, DataFrame) tuples
            每个 DataFrame 包含该用户的推文及其上下文（被回复的推文）
        """
        self._ensure_connection()
        
        # 获取所有有发帖记录的用户
        authors_query = """
        SELECT DISTINCT u.username
        FROM twitter_posts p
        JOIN twitter_users u ON p.user_id = u.id
        """
        
        authors_df = pd.read_sql(authors_query, self.connection)
        authors = authors_df['username'].tolist()
        
        results = []
        
        for author in authors:
            # 获取该用户的所有推文，包括其回复的上下文
            # 使用 conversation_id 和 in_reply_to_tweet_id 重建对话流
            query = """
            WITH user_posts AS (
                -- 获取该用户的所有推文
                SELECT 
                    p.id,
                    p.tweet_id,
                    p.text,
                    p.language,
                    p.type,
                    u.name AS author_name,
                    u.username AS author_username,
                    p.view_count,
                    p.reply_count,
                    p.retweet_count,
                    p.quote_count,
                    p.favorite_count,
                    p.bookmark_count,
                    p.published_at,
                    p.tweet_url,
                    p.source,
                    p.hashtags,
                    p.urls,
                    p.media_type,
                    p.media_urls,
                    p.in_reply_to_tweet_id,
                    p.conversation_id,
                    1 as is_target_user
                FROM twitter_posts p
                JOIN twitter_users u ON p.user_id = u.id
                WHERE u.username = %s
            ),
            replied_posts AS (
                -- 获取被回复的推文（用于上下文）
                SELECT 
                    p.id,
                    p.tweet_id,
                    p.text,
                    p.language,
                    p.type,
                    u.name AS author_name,
                    u.username AS author_username,
                    p.view_count,
                    p.reply_count,
                    p.retweet_count,
                    p.quote_count,
                    p.favorite_count,
                    p.bookmark_count,
                    p.published_at,
                    p.tweet_url,
                    p.source,
                    p.hashtags,
                    p.urls,
                    p.media_type,
                    p.media_urls,
                    p.in_reply_to_tweet_id,
                    p.conversation_id,
                    0 as is_target_user
                FROM twitter_posts p
                JOIN twitter_users u ON p.user_id = u.id
                WHERE p.tweet_id IN (
                    SELECT DISTINCT in_reply_to_tweet_id 
                    FROM user_posts 
                    WHERE in_reply_to_tweet_id IS NOT NULL
                )
            )
            SELECT * FROM user_posts
            UNION ALL
            SELECT * FROM replied_posts
            ORDER BY conversation_id, published_at
            """
            
            df = pd.read_sql(query, self.connection, params=[author])
            
            if not df.empty:
                # 转换为原有的 CSV 格式
                df_formatted = self._format_posts_dataframe(df)
                results.append((author, df_formatted))
        
        logger.info(f"从数据库加载了 {len(results)} 个用户的推文数据")
        return results
    
    def _format_posts_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将数据库查询结果转换为原有的 Replies CSV 格式
        重要：按 conversation_id 和 published_at 排序后，
        利用 in_reply_to_tweet_id 关系还原"上一行规则"
        """
        # 重命名列以匹配原有格式
        column_mapping = {
            'tweet_id': 'ID',
            'text': 'Text',
            'language': 'Language',
            'type': 'Type',
            'author_name': 'Author Name',
            'author_username': 'Author Username',
            'view_count': 'View Count',
            'reply_count': 'Reply Count',
            'retweet_count': 'Retweet Count',
            'quote_count': 'Quote Count',
            'favorite_count': 'Favorite Count',
            'bookmark_count': 'Bookmark Count',
            'published_at': 'Created At',
            'tweet_url': 'Tweet URL',
            'source': 'Source',
            'hashtags': 'hashtags',
            'urls': 'urls',
            'media_type': 'media_type',
            'media_urls': 'media_urls'
        }
        
        df_formatted = df.rename(columns=column_mapping)
        
        # 将 Type 转换为原有格式 (Tweet -> Origin, Reply -> Reply)
        type_mapping = {
            'Tweet': 'Origin',
            'Reply': 'Reply',
            'Retweet': 'Retweet',
            'Quoted': 'Quote'
        }
        df_formatted['Type'] = df_formatted['Type'].map(type_mapping).fillna('Origin')
        
        # 按对话和时间排序，构建对话流
        # 关键：需要将被回复的推文放在回复之前，模拟原有的 CSV 结构
        sorted_rows = []
        
        # 按 conversation_id 分组
        for conv_id, conv_df in df_formatted.groupby('conversation_id', dropna=False):
            conv_df = conv_df.sort_values('Created At')
            
            # 构建 tweet_id -> row 的映射
            tweet_map = {row.get('ID'): row for _, row in conv_df.iterrows()}
            
            # 已添加的推文
            added_tweets = set()
            
            for _, row in conv_df.iterrows():
                tweet_id = row.get('ID')
                reply_to = df.loc[df['tweet_id'] == tweet_id, 'in_reply_to_tweet_id'].values
                reply_to_id = reply_to[0] if len(reply_to) > 0 else None
                
                # 如果是回复，先添加被回复的推文（如果还未添加）
                if pd.notna(reply_to_id) and reply_to_id not in added_tweets:
                    if reply_to_id in tweet_map:
                        sorted_rows.append(tweet_map[reply_to_id])
                        added_tweets.add(reply_to_id)
                
                # 添加当前推文
                if tweet_id not in added_tweets:
                    sorted_rows.append(row)
                    added_tweets.add(tweet_id)
        
        if sorted_rows:
            result_df = pd.DataFrame(sorted_rows)
        else:
            result_df = df_formatted
        
        # 只保留需要的列
        required_columns = [
            'ID', 'Text', 'Language', 'Type', 'Author Name', 'Author Username',
            'View Count', 'Reply Count', 'Retweet Count', 'Quote Count',
            'Favorite Count', 'Bookmark Count', 'Created At', 'Tweet URL',
            'Source', 'hashtags', 'urls', 'media_type', 'media_urls'
        ]
        
        existing_columns = [col for col in required_columns if col in result_df.columns]
        return result_df[existing_columns]
    
    def get_all_posts(self) -> pd.DataFrame:
        """
        获取所有推文数据（不按用户分组）
        用于全局内容分析
        """
        self._ensure_connection()
        
        query = """
        SELECT 
            p.tweet_id AS `ID`,
            p.text AS `Text`,
            p.language AS `Language`,
            p.type AS `Type`,
            u.name AS `Author Name`,
            u.username AS `Author Username`,
            p.view_count AS `View Count`,
            p.reply_count AS `Reply Count`,
            p.retweet_count AS `Retweet Count`,
            p.quote_count AS `Quote Count`,
            p.favorite_count AS `Favorite Count`,
            p.bookmark_count AS `Bookmark Count`,
            DATE_FORMAT(p.published_at, '%Y-%m-%d %H:%i:%s') AS `Created At`,
            p.tweet_url AS `Tweet URL`,
            p.source AS `Source`,
            p.hashtags AS `hashtags`,
            p.urls AS `urls`,
            p.media_type AS `media_type`,
            p.media_urls AS `media_urls`,
            p.in_reply_to_tweet_id,
            p.conversation_id
        FROM twitter_posts p
        JOIN twitter_users u ON p.user_id = u.id
        ORDER BY p.published_at DESC
        """
        
        df = pd.read_sql(query, self.connection)
        
        # 转换 Type
        type_mapping = {
            'Tweet': 'Origin',
            'Reply': 'Reply',
            'Retweet': 'Retweet',
            'Quoted': 'Quote'
        }
        df['Type'] = df['Type'].map(type_mapping).fillna('Origin')
        
        logger.info(f"从数据库加载了 {len(df)} 条推文")
        return df
    
    def get_reply_relationships(self) -> pd.DataFrame:
        """
        获取所有回复关系数据
        利用 in_reply_to_tweet_id 直接获取回复关系，无需"上一行规则"
        
        Returns:
            DataFrame with columns: source_user, target_user, timestamp, text, weight
        """
        self._ensure_connection()
        
        query = """
        SELECT 
            u1.username AS source_user,
            u2.username AS target_user,
            p1.published_at AS timestamp,
            p1.text AS text,
            -- 互动权重计算
            (COALESCE(p1.view_count, 0) * 0.01 + 
             COALESCE(p1.reply_count, 0) * 3.0 + 
             COALESCE(p1.retweet_count, 0) * 4.0 + 
             COALESCE(p1.favorite_count, 0) * 2.0) AS weight
        FROM twitter_posts p1
        JOIN twitter_posts p2 ON p1.in_reply_to_tweet_id = p2.tweet_id
        JOIN twitter_users u1 ON p1.user_id = u1.id
        JOIN twitter_users u2 ON p2.user_id = u2.id
        WHERE p1.in_reply_to_tweet_id IS NOT NULL
          AND u1.username != u2.username  -- 排除自己回复自己
        ORDER BY p1.published_at
        """
        
        df = pd.read_sql(query, self.connection)
        logger.info(f"从数据库加载了 {len(df)} 条回复关系")
        return df
    
    def get_interaction_pairs(self) -> List[Dict]:
        """
        获取所有互动对数据，用于互惠性分析
        
        Returns:
            List of dicts with: source, target, timestamp, text
        """
        df = self.get_reply_relationships()
        
        return df[['source_user', 'target_user', 'timestamp', 'text']].rename(
            columns={'source_user': 'source', 'target_user': 'target'}
        ).to_dict('records')
    
    def get_core_users(self) -> set:
        """
        获取核心用户集合（有发帖记录或有关注关系的用户）
        """
        self._ensure_connection()
        
        query = """
        SELECT DISTINCT username FROM (
            -- 有发帖记录的用户
            SELECT u.username 
            FROM twitter_posts p
            JOIN twitter_users u ON p.user_id = u.id
            
            UNION
            
            -- 有关注关系的用户（关注者）
            SELECT u.username
            FROM twitter_followings f
            JOIN twitter_users u ON f.source_user_id = u.id
        ) AS core_users
        """
        
        df = pd.read_sql(query, self.connection)
        core_users = set(df['username'].tolist())
        logger.info(f"识别到 {len(core_users)} 个核心用户")
        return core_users
    
    def get_stats(self) -> Dict[str, int]:
        """获取数据库统计信息"""
        self._ensure_connection()
        
        stats = {}
        
        # 用户数
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM twitter_users")
        stats['users_count'] = cursor.fetchone()[0]
        
        # 推文数
        cursor.execute("SELECT COUNT(*) FROM twitter_posts")
        stats['posts_count'] = cursor.fetchone()[0]
        
        # 关注关系数
        cursor.execute("SELECT COUNT(*) FROM twitter_followings")
        stats['followings_count'] = cursor.fetchone()[0]
        
        # 回复数
        cursor.execute("SELECT COUNT(*) FROM twitter_posts WHERE in_reply_to_tweet_id IS NOT NULL")
        stats['replies_count'] = cursor.fetchone()[0]
        
        cursor.close()
        
        return stats


def is_database_available() -> bool:
    """检查数据库是否可用"""
    db_url = os.getenv('DATABASE_URL')
    db_host = os.getenv('DB_HOST')
    
    if not db_url and not db_host:
        return False
    
    try:
        adapter = DatabaseAdapter()
        adapter.close()
        return True
    except Exception as e:
        logger.warning(f"数据库不可用: {e}")
        return False


def get_db_adapter() -> Optional[DatabaseAdapter]:
    """获取数据库适配器实例"""
    try:
        return DatabaseAdapter()
    except Exception as e:
        logger.error(f"创建数据库适配器失败: {e}")
        return None
