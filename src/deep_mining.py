#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度挖掘脚本 (Phase 1 Completion)
继承自 MacroNetworkAnalyzer，补充社群发现、行为指纹分析和内容效能解码。

数据源：从 MySQL 数据库读取数据（通过 DATABASE_URL 或 DB_* 环境变量配置）

新增功能（基于 ANALYSIS_MINING_PLAN.md）：
- 内容资产四象限分析 (Library, Controversy, News, Cult)
- Thread 留存率分析
- 商业信号解码 (Newsletter, Product, Content, Community)
- 增强的候选清单生成
"""

import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain
import numpy as np
import os
import json
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from tqdm import tqdm
from urllib.parse import urlparse
import warnings

# 导入基础分析器
from .macro_analysis import MacroNetworkAnalyzer
from .analysis_db import get_analysis_db_adapter

warnings.filterwarnings('ignore')

# 商业信号域名映射
FUNNEL_DOMAIN_MAP = {
    # Newsletter 平台
    'newsletter': ['substack.com', 'beehiiv.com', 'convertkit.com', 'mailchimp.com', 
                   'buttondown.email', 'revue.co', 'getrevue.co', 'ghost.io'],
    # 产品/付费平台
    'product': ['gumroad.com', 'lemonsqueezy.com', 'patreon.com', 'ko-fi.com', 
                'buymeacoffee.com', 'stripe.com', 'paddle.com', 'stan.store'],
    # 内容平台
    'content': ['youtube.com', 'youtu.be', 'spotify.com', 'apple.com/podcast',
                'anchor.fm', 'medium.com', 'dev.to', 'hashnode.dev'],
    # 社群平台
    'community': ['discord.gg', 'discord.com', 'slack.com', 'circle.so',
                  'skool.com', 'mighty.com', 'telegram.me', 't.me'],
    # 个人网站/博客
    'website': ['github.io', 'notion.so', 'carrd.co', 'linktree.com', 'bio.link']
}

class DeepMiner(MacroNetworkAnalyzer):
    def __init__(self, output_dir='output'):
        super().__init__()
        self.output_dir = output_dir
        self.community_map = {}
        self.behavior_stats = defaultdict(lambda: {
            'reply_count': 0, 'origin_count': 0, 'retweet_count': 0,
            'sources': Counter(), 'active_hours': Counter(),
            'latencies': [], 'media_types': Counter(),
            'total_views': 0, 'total_likes': 0, 'total_bookmarks': 0,
            'total_replies_received': 0, 'total_retweets_received': 0
        })
        self.content_stats = [] # 存储每条帖子的详细数据用于内容分析
        self.interaction_pairs = [] # 存储互动对
        self.thread_stats = defaultdict(list)  # 存储 Thread 相关数据
        self.analysis_db = get_analysis_db_adapter()  # 分析数据库适配器
        self.session_id = None  # 当前分析会话 ID

    def run_deep_analysis(self):
        """运行完整的深度挖掘流程"""
        print("\n" + "=" * 60)
        print("开始深度挖掘分析 (Phase 1 Completion)")
        print("=" * 60)

        # 0. 创建分析会话
        if self.analysis_db:
            self.session_id = self.analysis_db.create_session({
                'type': 'deep_mining',
                'output_dir': self.output_dir
            })
            print(f"  ✓ 创建分析会话: {self.session_id}")

        # 1. 复用宏观分析构建网络 (G_combined, G_dynamic, users_profile)
        self.run_full_analysis()

        # 2. 社群发现 (Louvain)
        self._detect_communities()

        # 3. 行为与内容深度扫描 (需重新读取 Replies 文件以获取细节)
        self._scan_behavior_and_content()

        # 4. 内容资产四象限分析 (新增)
        self._analyze_content_assets()

        # 5. Thread 留存率分析 (新增)
        self._analyze_thread_retention()

        # 6. 计算并合并所有指标
        self._calculate_advanced_metrics()

        # 7. 生成 Phase 2 所需的候选清单
        self._generate_phase2_candidates()

        # 8. 生成并导出统计报告 (Phase 1 Outputs)
        self._generate_statistical_reports()

        # 9. 保存分析结果到数据库 (新增)
        self._save_to_analysis_db()

        # 10. 完成会话并清理旧数据
        if self.analysis_db and self.session_id:
            stats = {
                'users_analyzed': len(self.users_profile),
                'posts_analyzed': len(self.content_stats),
                'communities_found': len(set(self.community_map.values())) if self.community_map else 0
            }
            self.analysis_db.complete_session(self.session_id, stats)
            
            # 自动清理旧会话数据
            self.analysis_db.cleanup_old_sessions()
            
            # 输出存储统计信息
            storage_stats = self.analysis_db.get_storage_stats()
            print(f"\n[存储统计] 模式: {storage_stats.get('storage_mode', 'N/A')}, "
                  f"当前会话数: {storage_stats.get('tables', {}).get('analysis_sessions', 0)}")

        print("\n" + "=" * 60)
        print("深度挖掘完成！所有清单与统计报告已生成。")
        print("=" * 60)

    def _generate_statistical_reports(self):
        """生成 Phase 1 要求的具体统计报告产出物"""
        print("\n[Deep Mining] 生成统计报告 (Fingerprints & Content DNA)...")

        # 1. 活跃节律 - 小时热力图 (Hourly Heatmap)
        # 聚合所有用户的活跃小时分布
        total_hours = Counter()
        for user, stats in self.behavior_stats.items():
            total_hours.update(stats['active_hours'])

        heatmap_data = []
        total_activity = sum(total_hours.values())
        for hour in range(24):
            heatmap_data.append({
                'hour': hour,
                'activity_count': total_hours[hour],
                'activity_percentage': total_hours[hour] / total_activity if total_activity > 0 else 0
            })
        pd.DataFrame(heatmap_data).to_csv(f'{self.output_dir}/stats_hourly_heatmap.csv', index=False)
        print(f"  ✓ stats_hourly_heatmap.csv")

        # 2. 日活跃趋势 (Daily Trend)
        if self.content_stats:
            df_content = pd.DataFrame(self.content_stats)
            df_content['created_at'] = pd.to_datetime(df_content['created_at'], errors='coerce')
            df_content['date'] = df_content['created_at'].dt.date
            
            daily_data = df_content.groupby('date').agg({
                'id': 'count',
                'view_count': 'sum',
                'like_count': 'sum',
                'reply_count': 'sum'
            }).rename(columns={'id': 'post_count'}).reset_index()
            daily_data = daily_data.sort_values('date')
            daily_data.to_csv(f'{self.output_dir}/stats_daily_trend.csv', index=False)
            print(f"  ✓ stats_daily_trend.csv")

        # 3. 周活跃模式 (Weekly Pattern)
        if self.content_stats:
            df_content = pd.DataFrame(self.content_stats)
            df_content['created_at'] = pd.to_datetime(df_content['created_at'], errors='coerce')
            df_content['day_of_week'] = df_content['created_at'].dt.dayofweek  # 0=Monday, 6=Sunday
            df_content['day_name'] = df_content['created_at'].dt.day_name()
            
            weekly_data = df_content.groupby(['day_of_week', 'day_name']).agg({
                'id': 'count',
                'view_count': 'mean',
                'like_count': 'mean',
                'reply_count': 'mean'
            }).rename(columns={'id': 'post_count'}).reset_index()
            weekly_data = weekly_data.sort_values('day_of_week')
            weekly_data.to_csv(f'{self.output_dir}/stats_weekly_pattern.csv', index=False)
            print(f"  ✓ stats_weekly_pattern.csv")

        # 4. 内容格式效能 (Content Efficiency by Media Type)
        # 聚合不同 media_type 的平均表现
        if self.content_stats:
            df_content = pd.DataFrame(self.content_stats)
            efficiency = df_content.groupby('media_type').agg({
                'view_count': 'mean',
                'like_count': 'mean',
                'reply_count': 'mean',
                'bookmark_count': 'mean',
                'utility_score': 'mean',
                'id': 'count'
            }).rename(columns={'id': 'post_count'}).reset_index()

            efficiency.to_csv(f'{self.output_dir}/stats_content_efficiency.csv', index=False)
            print(f"  ✓ stats_content_efficiency.csv")

        # 5. 流量漏斗 (Traffic Funnel)
        # 计算全局转化率
        if self.content_stats:
            df_content = pd.DataFrame(self.content_stats)
            total_views = df_content['view_count'].sum()
            funnel = {
                'total_views': total_views,
                'total_likes': df_content['like_count'].sum(),
                'total_replies': df_content['reply_count'].sum(),
                'total_retweets': df_content.get('retweet_count', pd.Series([0]*len(df_content))).sum(),
                'view_to_like_rate': df_content['like_count'].sum() / total_views if total_views > 0 else 0,
                'view_to_reply_rate': df_content['reply_count'].sum() / total_views if total_views > 0 else 0,
            }
            pd.DataFrame([funnel]).to_csv(f'{self.output_dir}/stats_traffic_funnel.csv', index=False)
            print(f"  ✓ stats_traffic_funnel.csv")

    def _detect_communities(self):
        """使用 Louvain 算法进行社群发现"""
        print("\n[Deep Mining] 执行社群发现 (Louvain)...")

        # Louvain 最好在无向图上运行，或者将有向图转换为无向图
        G_undirected = self.G_dynamic.to_undirected()

        if G_undirected.number_of_edges() == 0:
            print("  警告: 动态网络无边，跳过社群发现")
            return

        partition = community_louvain.best_partition(G_undirected)
        self.community_map = partition

        # 将社群ID添加到用户画像
        for user, comm_id in partition.items():
            if user in self.users_profile:
                self.users_profile[user]['community_id'] = comm_id

        num_communities = len(set(partition.values()))
        print(f"  ✓ 发现 {num_communities} 个社群")
        
        # 生成社群统计文件
        self._generate_community_stats()

    def _generate_community_stats(self):
        """生成社群统计文件 community_stats.csv"""
        if not self.community_map:
            print("  警告: 无社群数据，跳过社群统计生成")
            return
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 按社群聚合统计
        community_data = defaultdict(lambda: {
            'member_count': 0,
            'members': [],
            'total_pagerank': 0,
            'total_betweenness': 0,
            'total_followers': 0
        })
        
        for user, comm_id in self.community_map.items():
            community_data[comm_id]['member_count'] += 1
            community_data[comm_id]['members'].append(user)
            
            if user in self.users_profile:
                profile = self.users_profile[user]
                community_data[comm_id]['total_pagerank'] += profile.get('pagerank', 0)
                community_data[comm_id]['total_betweenness'] += profile.get('betweenness', 0)
                community_data[comm_id]['total_followers'] += profile.get('followers_count', 0)
        
        # 构建输出数据
        stats_list = []
        for comm_id, data in community_data.items():
            member_count = data['member_count']
            # 获取 top 成员（按 pagerank 排序）
            members_with_pr = []
            for m in data['members']:
                if m in self.users_profile:
                    members_with_pr.append((m, self.users_profile[m].get('pagerank', 0)))
                else:
                    members_with_pr.append((m, 0))
            members_with_pr.sort(key=lambda x: x[1], reverse=True)
            top_members = [m[0] for m in members_with_pr[:10]]
            
            stats_list.append({
                'community_id': comm_id,
                'member_count': member_count,
                'avg_pagerank': data['total_pagerank'] / member_count if member_count > 0 else 0,
                'avg_betweenness': data['total_betweenness'] / member_count if member_count > 0 else 0,
                'total_followers': data['total_followers'],
                'top_members_json': json.dumps(top_members, ensure_ascii=False),
                'topic_keywords_json': '[]'  # 后续可以通过 LLM 分析补充
            })
        
        df_stats = pd.DataFrame(stats_list)
        df_stats = df_stats.sort_values('member_count', ascending=False)
        df_stats.to_csv(f'{self.output_dir}/community_stats.csv', index=False, encoding='utf-8-sig')
        print(f"  ✓ community_stats.csv ({len(df_stats)} 个社群)")

    def _scan_behavior_and_content(self):
        """扫描推文数据，提取行为指纹和内容指标"""
        print("\n[Deep Mining] 扫描行为指纹与内容基因...")
        self._scan_behavior_and_content_from_db()

    def _scan_behavior_and_content_from_db(self):
        """
        从数据库扫描行为指纹和内容指标
        
        性能优化说明：
        - 使用向量化操作替代 iterrows()，在十万级数据量下性能提升 10-50 倍
        - 使用 groupby + agg 聚合替代逐行累加
        - 使用 itertuples() 替代 iterrows()（在必须遍历的场景下）
        """
        # 获取所有推文数据
        df = self.db_adapter.get_all_posts()
        
        if df.empty:
            print("  警告: 未找到任何推文数据")
            return
        
        print(f"  处理 {len(df)} 条推文（向量化优化模式）...")
        
        # ========================================
        # 1. 向量化预处理
        # ========================================
        
        # 转换时间
        df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')
        
        # 提取小时（向量化）
        df['hour'] = df['Created At'].dt.hour
        
        # 填充缺失值（向量化）
        df['Author Username'] = df['Author Username'].fillna('')
        df['Type'] = df['Type'].fillna('')
        df['Source'] = df['Source'].fillna('')
        df['media_type'] = df['media_type'].fillna('none') if 'media_type' in df.columns else 'none'
        df['Text'] = df['Text'].fillna('')
        df['Urls'] = df['Urls'].fillna('') if 'Urls' in df.columns else ''
        df['Conversation ID'] = df['Conversation ID'].fillna('') if 'Conversation ID' in df.columns else ''
        
        # 数值列填充（向量化）
        numeric_cols = ['View Count', 'Favorite Count', 'Bookmark Count', 'Reply Count', 'Retweet Count', 'Quote Count']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0
        
        # 计算 Utility Score（向量化）
        df['utility_score'] = np.where(
            df['Favorite Count'] > 0,
            df['Bookmark Count'] / df['Favorite Count'],
            0
        )
        
        # 判断是否为问句（向量化）
        df['is_question'] = df['Text'].str.contains(r'[?？]', regex=True, na=False)
        
        # 过滤有效用户
        df_valid = df[df['Author Username'] != ''].copy()
        
        # ========================================
        # 2. 向量化聚合行为统计
        # ========================================
        print("  聚合用户行为统计...")
        
        # 2.1 活跃小时统计
        hour_counts = df_valid.groupby(['Author Username', 'hour']).size().reset_index(name='count')
        for row in hour_counts.itertuples(index=False):
            user, hour, count = row
            self.behavior_stats[user]['active_hours'][hour] += count
        
        # 2.2 Source 统计
        source_counts = df_valid.groupby(['Author Username', 'Source']).size().reset_index(name='count')
        for row in source_counts.itertuples(index=False):
            user, source, count = row
            if source:
                self.behavior_stats[user]['sources'][source] += count
        
        # 2.3 推文类型统计
        type_counts = df_valid.groupby(['Author Username', 'Type']).size().reset_index(name='count')
        for row in type_counts.itertuples(index=False):
            user, t_type, count = row
            if t_type == 'Reply':
                self.behavior_stats[user]['reply_count'] += count
            elif t_type == 'Origin':
                self.behavior_stats[user]['origin_count'] += count
            elif t_type == 'Retweet':
                self.behavior_stats[user]['retweet_count'] += count
        
        # 2.4 media_type 统计
        if 'media_type' in df_valid.columns:
            media_counts = df_valid.groupby(['Author Username', 'media_type']).size().reset_index(name='count')
            for row in media_counts.itertuples(index=False):
                user, media_type, count = row
                self.behavior_stats[user]['media_types'][media_type] += count
        
        # 2.5 数值聚合
        user_agg = df_valid.groupby('Author Username').agg({
            'View Count': 'sum',
            'Favorite Count': 'sum',
            'Bookmark Count': 'sum',
            'Reply Count': 'sum',
            'Retweet Count': 'sum'
        }).reset_index()
        
        for row in user_agg.itertuples(index=False):
            user = row[0]
            self.behavior_stats[user]['total_views'] = row[1]
            self.behavior_stats[user]['total_likes'] = row[2]
            self.behavior_stats[user]['total_bookmarks'] = row[3]
            self.behavior_stats[user]['total_replies_received'] = row[4]
            self.behavior_stats[user]['total_retweets_received'] = row[5]
        
        # ========================================
        # 3. 构建 content_stats（使用 itertuples 替代 iterrows）
        # ========================================
        print("  构建内容特征数据...")
        
        # 预先选择需要的列，减少内存访问
        content_columns = ['ID', 'Author Username', 'Text', 'Created At', 'Type', 
                          'View Count', 'Favorite Count', 'Bookmark Count', 
                          'Reply Count', 'Retweet Count', 'Quote Count',
                          'utility_score', 'is_question', 'Source', 'Urls', 'Conversation ID']
        
        if 'media_type' in df_valid.columns:
            content_columns.append('media_type')
        
        # 使用 itertuples 遍历（比 iterrows 快 10-100 倍）
        for row in tqdm(df_valid[content_columns].itertuples(index=False), 
                       total=len(df_valid), desc="  构建内容特征"):
            # 解包（注意顺序与 content_columns 对应）
            tweet_id = row[0]
            user = row[1]
            text = row[2]
            created_at = row[3]
            t_type = row[4]
            view_count = row[5]
            like_count = row[6]
            bookmark_count = row[7]
            reply_count_val = row[8]
            retweet_count = row[9]
            quote_count = row[10]
            utility_score = row[11]
            is_question = row[12]
            source = row[13]
            urls = row[14]
            conv_id = row[15]
            media_type = row[16] if len(row) > 16 else 'none'
            
            if not user:
                continue
            
            self.content_stats.append({
                'id': tweet_id,
                'author': user,
                'text': text,
                'created_at': created_at,
                'type': t_type,
                'media_type': media_type,
                'view_count': view_count,
                'like_count': like_count,
                'bookmark_count': bookmark_count,
                'reply_count': reply_count_val,
                'retweet_count': retweet_count,
                'utility_score': utility_score,
                'is_question': is_question,
                'source': source,
                'urls': urls,
                'conversation_id': conv_id,
                'quote_count': quote_count
            })
            
            # 记录 Thread 数据
            if conv_id:
                self.thread_stats[conv_id].append({
                    'tweet_id': tweet_id,
                    'view_count': view_count,
                    'created_at': created_at,
                    'author': user
                })
        
        # ========================================
        # 4. 处理回复关系（使用 itertuples）
        # ========================================
        reply_df = self.db_adapter.get_reply_relationships()
        
        if not reply_df.empty:
            print(f"  处理 {len(reply_df)} 条回复关系...")
            
            # 使用 itertuples 替代 iterrows
            for row in tqdm(reply_df.itertuples(index=False), total=len(reply_df), desc="  分析互动关系"):
                self.interaction_pairs.append({
                    'source': row.source_user,
                    'target': row.target_user,
                    'timestamp': row.timestamp,
                    'text': row.text
                })

    def _analyze_content_assets(self):
        """
        内容资产四象限分析 (基于 ANALYSIS_MINING_PLAN.md)
        - Library: 高书签/低点赞 (真正的复利资产)
        - Controversy: 高引用/高传播 (争议性内容)
        - News: 高浏览/低互动 (纯资讯)
        - Cult: 高互动/低浏览 (圈层内容)
        """
        print("\n[Deep Mining] 内容资产四象限分析...")
        
        if not self.content_stats:
            print("  警告: 无内容数据，跳过四象限分析")
            return
        
        df = pd.DataFrame(self.content_stats)
        
        # 计算传播力 (Virality Rate)
        df['virality_rate'] = (df['retweet_count'] + df.get('quote_count', 0)) / (df['view_count'] + 1)
        
        # 计算干货率 (Utility Rate) - 已有
        # df['utility_score'] 已计算
        
        # 计算讨论率 (Discussion Rate)
        df['discussion_rate'] = df['reply_count'] / (df['view_count'] + 1)
        
        # 引用/转发比（争议指标）
        df['controversy_ratio'] = df.get('quote_count', 0) / (df['retweet_count'] + 1)
        
        # 互动/浏览比
        df['engagement_rate'] = (df['like_count'] + df['reply_count'] + df['retweet_count']) / (df['view_count'] + 1)
        
        # 设定阈值（基于分位数）
        utility_threshold = df[df['like_count'] > 5]['utility_score'].quantile(0.75) if len(df[df['like_count'] > 5]) > 0 else 0.5
        virality_threshold = df['virality_rate'].quantile(0.75)
        engagement_threshold = df['engagement_rate'].quantile(0.75)
        
        # 四象限分类
        asset_quadrants = []
        for idx, row in df.iterrows():
            if row['utility_score'] > utility_threshold and row['like_count'] > 5:
                quadrant = 'library'  # 高书签/点赞 - 复利资产
            elif row.get('controversy_ratio', 0) > 0.5 or row['virality_rate'] > virality_threshold:
                quadrant = 'controversy'  # 高争议
            elif row['view_count'] > df['view_count'].median() and row['engagement_rate'] < engagement_threshold:
                quadrant = 'news'  # 高浏览低互动
            elif row['engagement_rate'] > engagement_threshold and row['view_count'] < df['view_count'].median():
                quadrant = 'cult'  # 高互动低浏览
            else:
                quadrant = 'other'
            asset_quadrants.append(quadrant)
        
        df['asset_quadrant'] = asset_quadrants
        
        # 更新 content_stats
        for i, quad in enumerate(asset_quadrants):
            if i < len(self.content_stats):
                self.content_stats[i]['asset_quadrant'] = quad
                self.content_stats[i]['virality_rate'] = df.iloc[i]['virality_rate']
                self.content_stats[i]['discussion_rate'] = df.iloc[i]['discussion_rate']
        
        # 统计各象限分布
        quadrant_counts = df['asset_quadrant'].value_counts()
        print(f"  ✓ 四象限分布:")
        for q, count in quadrant_counts.items():
            print(f"    - {q}: {count} ({count/len(df)*100:.1f}%)")

    def _analyze_thread_retention(self):
        """
        Thread 留存率分析 (基于 ANALYSIS_MINING_PLAN.md)
        计算 Thread 首楼与后续楼层的浏览量衰减曲线
        """
        print("\n[Deep Mining] Thread 留存率分析...")
        
        if not self.thread_stats:
            print("  警告: 无 Thread 数据，跳过留存率分析")
            return
        
        thread_retention_data = []
        
        for conv_id, tweets in self.thread_stats.items():
            if len(tweets) < 2:
                continue  # 单条推文不算 Thread
            
            # 按时间排序
            sorted_tweets = sorted(tweets, key=lambda x: x.get('created_at') or datetime.min)
            
            first_view = sorted_tweets[0].get('view_count', 0)
            if first_view <= 0:
                continue
            
            last_view = sorted_tweets[-1].get('view_count', 0)
            retention_rate = last_view / first_view if first_view > 0 else 0
            
            thread_retention_data.append({
                'conversation_id': conv_id,
                'thread_length': len(sorted_tweets),
                'first_view': first_view,
                'last_view': last_view,
                'retention_rate': retention_rate,
                'author': sorted_tweets[0].get('author', '')
            })
        
        if thread_retention_data:
            df_threads = pd.DataFrame(thread_retention_data)
            
            # 统计最佳 Thread 长度
            avg_retention_by_length = df_threads.groupby('thread_length')['retention_rate'].mean()
            best_length = avg_retention_by_length.idxmax() if len(avg_retention_by_length) > 0 else 0
            
            print(f"  ✓ 分析了 {len(df_threads)} 个 Thread")
            print(f"    - 平均留存率: {df_threads['retention_rate'].mean():.2%}")
            print(f"    - 最佳 Thread 长度: {best_length} 条")
            
            # 保存高留存率 Thread
            high_retention = df_threads[df_threads['retention_rate'] > df_threads['retention_rate'].median()]
            high_retention.to_csv(f'{self.output_dir}/list_threads_viral.csv', index=False, encoding='utf-8-sig')
            print(f"  ✓ list_threads_viral.csv ({len(high_retention)} 条)")
            
            # 更新 content_stats 中的 Thread 特征
            thread_map = {t['conversation_id']: t for t in thread_retention_data}
            for i, stat in enumerate(self.content_stats):
                conv_id = stat.get('conversation_id', '')
                if conv_id in thread_map:
                    self.content_stats[i]['thread_retention_rate'] = thread_map[conv_id]['retention_rate']
                    self.content_stats[i]['thread_length'] = thread_map[conv_id]['thread_length']

    def _extract_funnel_signal(self, urls_str: str) -> str:
        """
        商业信号解码：从 URL 中提取变现漏斗信号
        """
        if not urls_str or pd.isna(urls_str):
            return None
        
        try:
            # URL 可能是逗号分隔的字符串或 JSON 数组
            if urls_str.startswith('['):
                urls = json.loads(urls_str)
            else:
                urls = [u.strip() for u in urls_str.split(',')]
        except:
            urls = [urls_str]
        
        for url in urls:
            if not url:
                continue
            try:
                domain = urlparse(str(url)).netloc.lower()
                # 去除 www 前缀
                domain = domain.replace('www.', '')
                
                for signal_type, domains in FUNNEL_DOMAIN_MAP.items():
                    for d in domains:
                        if d in domain:
                            return signal_type
            except:
                continue
        
        return None

    def _calculate_advanced_metrics(self):
        """计算高级指标并更新 users_profile"""
        print("\n[Deep Mining] 计算高级指标...")

        for user, stats in self.behavior_stats.items():
            if user not in self.users_profile:
                continue

            profile = self.users_profile[user]

            # 1. 互动/广播比率 (Talkativity)
            total_tweets = stats['reply_count'] + stats['origin_count'] + stats['retweet_count']
            if total_tweets > 0:
                profile['talkativity_ratio'] = stats['reply_count'] / total_tweets
            else:
                profile['talkativity_ratio'] = 0

            # 2. 专业度指数 (Professionalism)
            # Typefully/Hypefury 等视为专业
            pro_tools = ['Typefully', 'Hypefury', 'Buffer', 'SocialPilot']
            pro_count = sum(stats['sources'][s] for s in stats['sources'] if any(t in str(s) for t in pro_tools))
            if total_tweets > 0:
                profile['professionalism_index'] = pro_count / total_tweets
            else:
                profile['professionalism_index'] = 0

            # 3. 回复时效性 (Latency)
            if stats['latencies']:
                profile['avg_reply_latency_seconds'] = np.mean(stats['latencies'])
            else:
                profile['avg_reply_latency_seconds'] = None

            # 4. 潜龙在渊指数 (Rising Star Velocity) - 已在 macro 中部分计算，这里强化
            # PageRank / Account Age Days
            # 确保 keys 存在
            pr = profile.get('pagerank', 0)
            created_at = profile.get('created_at', '')
            days_exist = 1
            if created_at:
                try:
                    d = datetime.strptime(str(created_at), '%Y-%m-%d %H:%M:%S')
                    days_exist = (datetime.now() - d).days
                except:
                    pass
            if days_exist < 1: days_exist = 1

            profile['rising_star_velocity'] = pr / days_exist * 10000 # 放大系数方便查看

            # 5. 干货生产力 (Utility Provider)
            # Avg Utility Score of their posts
            # 这里简单用总收藏/总点赞估算
            if stats['total_likes'] > 0:
                profile['avg_utility_score'] = stats['total_bookmarks'] / stats['total_likes']
            else:
                profile['avg_utility_score'] = 0

    def _generate_phase2_candidates(self):
        """生成 Phase 2 所需的清单"""
        print("\n[Deep Mining] 生成 Phase 2 候选清单...")
        os.makedirs(self.output_dir, exist_ok=True)

        # 清单 1: list_posts_outliers.csv (异常价值帖子) 及各类型单独文件
        # 筛选标准：干货指数 Top 50 OR 浏览量 Top 50 OR 讨论率极高
        df_content = pd.DataFrame(self.content_stats)
        df_users = pd.DataFrame(self.users_profile.values()) if self.users_profile else pd.DataFrame()
        
        if not df_content.empty:
            # 排除无效内容
            df_content = df_content[df_content['text'].str.len() > 10]

            # 计算讨论率
            df_content['discussion_rate'] = df_content['reply_count'] / (df_content['view_count'] + 1)

            # 策略A：高干货 (Utility > 0.5 且 Like > 5)
            high_utility = df_content[(df_content['utility_score'] > 0.5) & (df_content['like_count'] > 5)].copy()
            high_utility['outlier_type'] = 'high_utility'
            high_utility = high_utility.head(50)
            
            # 单独导出高干货内容
            if not high_utility.empty:
                high_utility.to_csv(f'{self.output_dir}/list_posts_high_utility.csv', index=False, encoding='utf-8-sig')
                print(f"  ✓ list_posts_high_utility.csv ({len(high_utility)} 条)")

            # 策略B：高流量
            high_traffic = df_content.sort_values('view_count', ascending=False).head(50).copy()
            high_traffic['outlier_type'] = 'high_traffic'
            
            # 单独导出高流量内容
            high_traffic.to_csv(f'{self.output_dir}/list_posts_high_traffic.csv', index=False, encoding='utf-8-sig')
            print(f"  ✓ list_posts_high_traffic.csv ({len(high_traffic)} 条)")

            # 策略C：高讨论 (Reply/View 高)
            high_discuss = df_content.sort_values('discussion_rate', ascending=False).head(50).copy()
            high_discuss['outlier_type'] = 'high_discussion'
            
            # 单独导出高讨论内容
            high_discuss.to_csv(f'{self.output_dir}/list_posts_high_discussion.csv', index=False, encoding='utf-8-sig')
            print(f"  ✓ list_posts_high_discussion.csv ({len(high_discuss)} 条)")

            # 合并所有异常内容并去重
            outliers = pd.concat([high_utility, high_traffic, high_discuss]).drop_duplicates(subset=['id', 'text'])
            outliers.to_csv(f'{self.output_dir}/list_posts_outliers.csv', index=False, encoding='utf-8-sig')
            print(f"  ✓ list_posts_outliers.csv ({len(outliers)} 条)")

        # 清单 2: list_users_key_players.csv (关键角色)
        # 包含：PageRank Top, Betweenness Top, Rising Star Top, Professional Top
        if not df_users.empty:
            # 收集可用的排序结果
            player_dfs = []
            
            # 检查每个排序列是否存在
            if 'pagerank' in df_users.columns:
                player_dfs.append(df_users.sort_values('pagerank', ascending=False).head(50))
            if 'betweenness' in df_users.columns:
                player_dfs.append(df_users.sort_values('betweenness', ascending=False).head(50))
            if 'rising_star_velocity' in df_users.columns:
                player_dfs.append(df_users.sort_values('rising_star_velocity', ascending=False).head(30))
            if 'avg_utility_score' in df_users.columns:
                player_dfs.append(df_users.sort_values('avg_utility_score', ascending=False).head(20))
            
            # 如果没有任何可用的排序列，跳过此清单
            if not player_dfs:
                print("  ⚠️ 跳过 list_users_key_players.csv（无可用排序指标）")
            else:
                key_players = pd.concat(player_dfs).drop_duplicates(subset=['username'])

                # 必须保留关键指标列
                cols = ['username', 'name', 'bio', 'pagerank', 'betweenness', 'rising_star_velocity',
                        'professionalism_index', 'talkativity_ratio', 'avg_utility_score', 'community_id']
                # 确保列存在
                existing_cols = [c for c in cols if c in df_users.columns]
                key_players[existing_cols].to_csv(f'{self.output_dir}/list_users_key_players.csv', index=False, encoding='utf-8-sig')
                print(f"  ✓ list_users_key_players.csv ({len(key_players)} 人)")

        # 清单 3: list_interactions_strong_ties.csv (强互惠关系)
        # 互惠性：A->B 且 B->A
        # 分析 self.interaction_pairs
        if self.interaction_pairs:
            df_inter = pd.DataFrame(self.interaction_pairs)
            # 统计两两互动次数
            pair_counts = df_inter.groupby(['source', 'target']).size().reset_index(name='count')

            # 寻找互惠对
            strong_ties = []
            # 为了效率，转为字典查找
            pair_dict = {}
            for _, row in pair_counts.iterrows():
                pair_dict[(row['source'], row['target'])] = row['count']

            processed_pairs = set()

            for (u, v), count_uv in pair_dict.items():
                if (v, u) in pair_dict and (u, v) not in processed_pairs and (v, u) not in processed_pairs:
                    count_vu = pair_dict[(v, u)]
                    total_interaction = count_uv + count_vu
                    strong_ties.append({
                        'user_a': u,
                        'user_b': v,
                        'weight': total_interaction,
                        'type': 'reciprocal'
                    })
                    processed_pairs.add((u, v))

            df_ties = pd.DataFrame(strong_ties).sort_values('weight', ascending=False).head(100)

            # 提取这些关系的对话样本 (这里简化，只存关系对元数据，Phase 2 LLM 再去根据关系对找文本?
            # 或者这里直接把文本附上？为了 Phase 2 方便，我们应该附上最近的几条互动文本)

            # 构建快速查找 map
            # key: (u, v), value: list of texts
            inter_texts = defaultdict(list)
            for _, row in df_inter.iterrows():
                inter_texts[(row['source'], row['target'])].append(row['text'])

            # 补充文本到 strong_ties
            final_ties = []
            for _, row in df_ties.iterrows():
                u, v = row['user_a'], row['user_b']
                texts = inter_texts[(u, v)] + inter_texts[(v, u)]
                # 截取前5条文本作为样本
                final_ties.append({
                    'user_a': u,
                    'user_b': v,
                    'weight': row['weight'],
                    'interaction_samples': json.dumps(texts[:5], ensure_ascii=False)
                })

            pd.DataFrame(final_ties).to_csv(f'{self.output_dir}/list_interactions_strong_ties.csv', index=False, encoding='utf-8-sig')
            print(f"  ✓ list_interactions_strong_ties.csv ({len(final_ties)} 对)")

        # 清单 4: list_content_opportunities.csv (内容机会)
        # 筛选标准：是问句，回复数为0，但作者影响力尚可（不是垃圾号）；或者 回复数极高（争议）
        if not df_content.empty and not df_users.empty:
            # 机会 A: 待回答的高价值问题 (Question, Reply=0, Author PageRank Top 50%)
            # 先获取作者 PageRank
            if 'pagerank' in df_users.columns:
                median_pr = df_users['pagerank'].median()

                # map author pagerank
                author_pr = df_users.set_index('username')['pagerank'].to_dict()
                df_content['author_pr'] = df_content['author'].map(author_pr).fillna(0)

                unanswered_questions = df_content[
                    (df_content['is_question']) &
                    (df_content['reply_count'] == 0) &
                    (df_content['author_pr'] > median_pr)
                ].copy()
                unanswered_questions['outlier_type'] = 'unanswered_question'
                unanswered_questions['opportunity_type'] = 'unanswered_question'
                unanswered_questions = unanswered_questions.head(50)
                
                # 单独导出未回答问题
                if not unanswered_questions.empty:
                    unanswered_questions.to_csv(f'{self.output_dir}/list_posts_unanswered_question.csv', index=False, encoding='utf-8-sig')
                    print(f"  ✓ list_posts_unanswered_question.csv ({len(unanswered_questions)} 条)")

                # 机会 B: 激烈讨论 (Reply > 10)
                hot_debates = df_content[df_content['reply_count'] > 10].copy()
                hot_debates['outlier_type'] = 'hot_debate'
                hot_debates['opportunity_type'] = 'hot_debate'
                hot_debates = hot_debates.head(50)
                
                # 单独导出热议话题
                if not hot_debates.empty:
                    hot_debates.to_csv(f'{self.output_dir}/list_posts_hot_debate.csv', index=False, encoding='utf-8-sig')
                    print(f"  ✓ list_posts_hot_debate.csv ({len(hot_debates)} 条)")

                # 合并内容机会
                opportunities = pd.concat([unanswered_questions, hot_debates])
                opportunities.to_csv(f'{self.output_dir}/list_content_opportunities.csv', index=False, encoding='utf-8-sig')
                print(f"  ✓ list_content_opportunities.csv ({len(opportunities)} 条)")

        # 新增清单 5: list_posts_assets.csv (按四象限分类的内容资产)
        if not df_content.empty and 'asset_quadrant' in df_content.columns:
            # 为每个象限选择 Top 内容
            assets_by_quadrant = []
            for quadrant in ['library', 'controversy', 'news', 'cult']:
                q_content = df_content[df_content['asset_quadrant'] == quadrant]
                if not q_content.empty:
                    # 按不同指标排序
                    if quadrant == 'library':
                        top = q_content.sort_values('utility_score', ascending=False).head(20)
                    elif quadrant == 'controversy':
                        top = q_content.sort_values('virality_rate', ascending=False).head(20)
                    elif quadrant == 'news':
                        top = q_content.sort_values('view_count', ascending=False).head(20)
                    else:  # cult
                        top = q_content.sort_values('discussion_rate', ascending=False).head(20)
                    assets_by_quadrant.append(top)
            
            if assets_by_quadrant:
                all_assets = pd.concat(assets_by_quadrant)
                all_assets.to_csv(f'{self.output_dir}/list_posts_assets.csv', index=False, encoding='utf-8-sig')
                print(f"  ✓ list_posts_assets.csv ({len(all_assets)} 条)")

        # 新增清单 6: 商业信号统计
        if not df_content.empty:
            # 提取每条内容的商业信号
            funnel_signals = []
            for _, row in df_content.iterrows():
                signal = self._extract_funnel_signal(row.get('urls', ''))
                if signal:
                    funnel_signals.append({
                        'tweet_id': row.get('id'),
                        'author': row.get('author'),
                        'funnel_signal': signal,
                        'view_count': row.get('view_count'),
                        'like_count': row.get('like_count'),
                        'text_preview': str(row.get('text', ''))[:100]
                    })
            
            if funnel_signals:
                df_funnel = pd.DataFrame(funnel_signals)
                df_funnel.to_csv(f'{self.output_dir}/stats_funnel_signals.csv', index=False, encoding='utf-8-sig')
                print(f"  ✓ stats_funnel_signals.csv ({len(df_funnel)} 条含商业信号)")
                
                # 统计各类变现渠道分布
                signal_counts = df_funnel['funnel_signal'].value_counts()
                print(f"    商业信号分布:")
                for sig, count in signal_counts.items():
                    print(f"      - {sig}: {count}")

    def _save_to_analysis_db(self):
        """将分析结果保存到分析数据库"""
        if not self.analysis_db or not self.session_id:
            print("\n[Deep Mining] 分析数据库未配置，跳过数据库保存")
            return
        
        print("\n[Deep Mining] 保存分析结果到数据库...")
        
        try:
            # 1. 保存用户指标
            if self.users_profile:
                print("  [1/6] 保存用户指标...")
                df_users = pd.DataFrame(self.users_profile.values())
                self.analysis_db.save_user_metrics(self.session_id, df_users)
                print(f"    ✓ 用户指标: {len(df_users)} 条")
            
            # 2. 保存用户历史快照 (用于时序分析)
            if self.users_profile:
                print("  [2/6] 保存用户历史快照...")
                df_users = pd.DataFrame(self.users_profile.values())
                if 'followers_count' in df_users.columns:
                    self.analysis_db.save_user_stats_history(df_users)
                    print(f"    ✓ 用户历史快照: {len(df_users)} 条")
            
            # 3. 保存推文特征 (升级版)
            if self.content_stats:
                print(f"  [3/6] 保存推文特征 ({len(self.content_stats)} 条)...")
                df_features = pd.DataFrame(self.content_stats)
                df_features['tweet_id'] = df_features['id']
                self.analysis_db.save_post_features_enhanced(self.session_id, df_features)
                print(f"    ✓ 推文特征: {len(df_features)} 条")
            
            # 4. 保存强互惠关系
            print("  [4/6] 保存强互惠关系...")
            if self.interaction_pairs:
                df_inter = pd.DataFrame(self.interaction_pairs)
                pair_counts = df_inter.groupby(['source', 'target']).size().reset_index(name='count')
                pair_dict = {}
                for _, row in pair_counts.iterrows():
                    pair_dict[(row['source'], row['target'])] = row['count']
                
                # 构建互动文本快速查找 map: key=(u, v), value=list of texts
                inter_texts = defaultdict(list)
                for _, row in df_inter.iterrows():
                    inter_texts[(row['source'], row['target'])].append(row.get('text', ''))
                
                strong_ties = []
                processed_pairs = set()
                for (u, v), count_uv in pair_dict.items():
                    if (v, u) in pair_dict and (u, v) not in processed_pairs and (v, u) not in processed_pairs:
                        count_vu = pair_dict[(v, u)]
                        # 提取双向互动文本作为样本（取前5条）
                        texts = inter_texts[(u, v)] + inter_texts[(v, u)]
                        # 过滤空文本
                        texts = [t for t in texts if t and str(t).strip()]
                        strong_ties.append({
                            'user_a': u,
                            'user_b': v,
                            'weight': count_uv + count_vu,
                            'interaction_samples': json.dumps(texts[:5], ensure_ascii=False) if texts else '[]'
                        })
                        processed_pairs.add((u, v))
                
                if strong_ties:
                    df_ties = pd.DataFrame(strong_ties)
                    self.analysis_db.save_strong_ties(self.session_id, df_ties)
                    print(f"    ✓ 强互惠关系: {len(strong_ties)} 条")
                else:
                    print("    - 无强互惠关系数据")
            else:
                print("    - 无互动数据")
            
            # 4.5 保存社群统计
            print("  [5/6] 保存社群统计...")
            if self.community_map:
                # 按社群聚合统计
                community_data = defaultdict(lambda: {
                    'member_count': 0,
                    'members': [],
                    'total_pagerank': 0,
                    'total_betweenness': 0,
                    'total_followers': 0
                })
                
                for user, comm_id in self.community_map.items():
                    community_data[comm_id]['member_count'] += 1
                    community_data[comm_id]['members'].append(user)
                    
                    if user in self.users_profile:
                        profile = self.users_profile[user]
                        community_data[comm_id]['total_pagerank'] += profile.get('pagerank', 0)
                        community_data[comm_id]['total_betweenness'] += profile.get('betweenness', 0)
                        community_data[comm_id]['total_followers'] += profile.get('followers_count', 0)
                
                community_stats = []
                for comm_id, data in community_data.items():
                    member_count = data['member_count']
                    # 获取 top 成员（按 pagerank 排序）
                    members_with_pr = []
                    for m in data['members']:
                        if m in self.users_profile:
                            members_with_pr.append((m, self.users_profile[m].get('pagerank', 0)))
                        else:
                            members_with_pr.append((m, 0))
                    members_with_pr.sort(key=lambda x: x[1], reverse=True)
                    top_members = [m[0] for m in members_with_pr[:10]]
                    
                    community_stats.append({
                        'community_id': comm_id,
                        'member_count': member_count,
                        'avg_pagerank': data['total_pagerank'] / member_count if member_count > 0 else 0,
                        'avg_betweenness': data['total_betweenness'] / member_count if member_count > 0 else 0,
                        'total_followers': data['total_followers'],
                        'top_members': top_members,
                        'topic_keywords': []
                    })
                
                if community_stats:
                    self.analysis_db.save_community_stats(self.session_id, community_stats)
                    print(f"    ✓ 社群统计: {len(community_stats)} 个社群")
            else:
                print("    - 无社群数据")
            
            # 5. 保存内容异常点
            print("  [6/6] 保存内容异常点和活跃度统计...")
            if self.content_stats:
                df_content = pd.DataFrame(self.content_stats)
                # 筛选高价值内容
                high_value = df_content[
                    (df_content['utility_score'] > 0.5) | 
                    (df_content['view_count'] > df_content['view_count'].median()) |
                    (df_content['is_question'] == True)
                ].head(200)
                if not high_value.empty:
                    self.analysis_db.save_content_outliers(self.session_id, high_value)
            
            # 6. 保存活跃度统计（小时热力图）
            total_hours = Counter()
            for user, stats in self.behavior_stats.items():
                total_hours.update(stats['active_hours'])
            
            heatmap_data = []
            total = sum(total_hours.values())
            for hour in range(24):
                heatmap_data.append({
                    'hour': hour,
                    'activity_count': total_hours[hour],
                    'activity_percentage': total_hours[hour] / total if total > 0 else 0
                })
            if heatmap_data:
                self.analysis_db.save_activity_stats(self.session_id, heatmap_data, 'hourly_heatmap')
            
            # 7. 保存日活跃趋势
            if self.content_stats:
                df_content = pd.DataFrame(self.content_stats)
                df_content['created_at'] = pd.to_datetime(df_content['created_at'], errors='coerce')
                df_content['date'] = df_content['created_at'].dt.date
                
                daily_data = df_content.groupby('date').agg({
                    'id': 'count'
                }).rename(columns={'id': 'post_count'}).reset_index()
                
                daily_stats = []
                for _, row in daily_data.iterrows():
                    daily_stats.append({
                        'date': str(row['date']),
                        'post_count': row['post_count']
                    })
                if daily_stats:
                    self.analysis_db.save_activity_stats(self.session_id, daily_stats, 'daily_trend')
            
            # 8. 保存周活跃模式
            if self.content_stats:
                df_content = pd.DataFrame(self.content_stats)
                df_content['created_at'] = pd.to_datetime(df_content['created_at'], errors='coerce')
                df_content['day_of_week'] = df_content['created_at'].dt.dayofweek
                
                weekly_data = df_content.groupby('day_of_week').agg({
                    'id': 'count'
                }).rename(columns={'id': 'post_count'}).reset_index()
                
                weekly_stats = []
                for _, row in weekly_data.iterrows():
                    weekly_stats.append({
                        'day_of_week': row['day_of_week'],
                        'post_count': row['post_count']
                    })
                if weekly_stats:
                    self.analysis_db.save_activity_stats(self.session_id, weekly_stats, 'weekly_pattern')
            
            # 9. 保存内容效能统计
            if self.content_stats:
                df_content = pd.DataFrame(self.content_stats)
                efficiency = df_content.groupby('media_type').agg({
                    'view_count': 'mean',
                    'like_count': 'mean',
                    'reply_count': 'mean',
                    'bookmark_count': 'mean',
                    'utility_score': 'mean',
                    'id': 'count'
                }).rename(columns={'id': 'post_count'}).reset_index()
                if not efficiency.empty:
                    self.analysis_db.save_content_efficiency(self.session_id, efficiency)
            
            print("  ✓ 分析结果已保存到数据库")
            
        except Exception as e:
            print(f"  ✗ 保存到数据库时出错: {e}")


def main():
    """命令行入口，支持细粒度任务拆分"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Deep Mining - 社交网络深度挖掘分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 完整运行所有任务
  python -m src.deep_mining
  
  # 只运行网络分析
  python -m src.deep_mining --tasks network
  
  # 只运行内容分析
  python -m src.deep_mining --tasks content
  
  # 组合运行多个任务
  python -m src.deep_mining --tasks network,community,content
  
可用任务:
  network    - 网络图构建 + PageRank + Betweenness (无LLM)
  community  - Louvain 社群发现 (无LLM)
  behavior   - 行为指纹扫描 (无LLM)
  content    - 内容四象限 + Thread 分析 (无LLM)
  all        - 全部任务 (默认)
        """
    )
    
    parser.add_argument(
        '--tasks',
        type=str,
        default='all',
        help='要执行的任务列表，逗号分隔 (network,community,behavior,content,all)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='输出目录 (默认: output)'
    )
    
    args = parser.parse_args()
    
    # 解析任务列表
    task_list = [t.strip() for t in args.tasks.split(',')]
    
    # 如果包含 'all'，运行所有任务
    if 'all' in task_list:
        task_list = ['network', 'community', 'behavior', 'content']
    
    # 验证任务名称
    valid_tasks = {'network', 'community', 'behavior', 'content'}
    invalid_tasks = set(task_list) - valid_tasks
    if invalid_tasks:
        print(f"❌ 错误: 无效的任务名称: {', '.join(invalid_tasks)}")
        print(f"✅ 有效任务: {', '.join(valid_tasks)}")
        return
    
    print("\n" + "=" * 60)
    print("Deep Mining 任务配置")
    print("=" * 60)
    print(f"执行任务: {', '.join(task_list)}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    
    # 创建 DeepMiner 实例
    miner = DeepMiner(output_dir=args.output_dir)
    
    # 创建分析会话
    if miner.analysis_db:
        miner.session_id = miner.analysis_db.create_session({
            'type': 'deep_mining',
            'output_dir': miner.output_dir,
            'tasks': task_list
        })
        print(f"  ✓ 创建分析会话: {miner.session_id}")
    
    # 基础步骤：加载数据并构建网络（所有任务都需要）
    print("\n[准备阶段] 加载数据...")
    miner.load_all_data()
    miner.build_combined_network()
    
    # 执行选定的任务
    if 'network' in task_list:
        print("\n" + "=" * 60)
        print("执行任务: 网络指标计算")
        print("=" * 60)
        miner.calculate_network_metrics()
        miner.identify_external_influencers()
        miner.generate_watchlists(output_dir=miner.output_dir)
        miner.save_network_data(output_dir=miner.output_dir)
    
    if 'community' in task_list:
        print("\n" + "=" * 60)
        print("执行任务: 社群发现")
        print("=" * 60)
        miner._detect_communities()
    
    if 'behavior' in task_list:
        print("\n" + "=" * 60)
        print("执行任务: 行为指纹扫描")
        print("=" * 60)
        miner._scan_behavior_and_content()
    
    if 'content' in task_list:
        print("\n" + "=" * 60)
        print("执行任务: 内容分析")
        print("=" * 60)
        # 如果还没运行 behavior 任务，需要先扫描
        if 'behavior' not in task_list:
            print("  注意: 内容分析依赖行为扫描，自动执行...")
            miner._scan_behavior_and_content()
        
        miner._analyze_content_assets()
        miner._analyze_thread_retention()
        miner._calculate_advanced_metrics()
        miner._generate_phase2_candidates()
        miner._generate_statistical_reports()
    
    # 保存到数据库
    if miner.analysis_db and miner.session_id:
        print("\n[保存结果] 写入分析数据库...")
        miner._save_to_analysis_db()
        
        # 完成会话
        stats = {
            'tasks_executed': task_list,
            'users_analyzed': len(miner.users_profile),
            'posts_analyzed': len(miner.content_stats),
            'communities_found': len(set(miner.community_map.values())) if miner.community_map else 0
        }
        miner.analysis_db.complete_session(miner.session_id, stats)
        
        # 自动清理旧会话数据
        miner.analysis_db.cleanup_old_sessions()
        
        # 输出存储统计信息
        storage_stats = miner.analysis_db.get_storage_stats()
        print(f"\n[存储统计] 模式: {storage_stats.get('storage_mode', 'N/A')}, "
              f"当前会话数: {storage_stats.get('tables', {}).get('analysis_sessions', 0)}")
    
    print("\n" + "=" * 60)
    print("✅ 深度挖掘完成！")
    print("=" * 60)
    print(f"输出目录: {miner.output_dir}")
    print(f"执行的任务: {', '.join(task_list)}")

if __name__ == '__main__':
    main()
