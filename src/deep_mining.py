#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度挖掘脚本 (Phase 1 Completion)
继承自 MacroNetworkAnalyzer，补充社群发现、行为指纹分析和内容效能解码。
"""

import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain
import numpy as np
import os
import glob
import json
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from tqdm import tqdm
import warnings

# 导入基础分析器
from .macro_analysis import MacroNetworkAnalyzer

warnings.filterwarnings('ignore')

class DeepMiner(MacroNetworkAnalyzer):
    def __init__(self, followers_dir='X_followers', replies_dir='X_replies', output_dir='output'):
        super().__init__(followers_dir, replies_dir)
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

    def run_deep_analysis(self):
        """运行完整的深度挖掘流程"""
        print("\n" + "=" * 60)
        print("开始深度挖掘分析 (Phase 1 Completion)")
        print("=" * 60)

        # 1. 复用宏观分析构建网络 (G_combined, G_dynamic, users_profile)
        self.run_full_analysis()

        # 2. 社群发现 (Louvain)
        self._detect_communities()

        # 3. 行为与内容深度扫描 (需重新读取 Replies 文件以获取细节)
        self._scan_behavior_and_content()

        # 4. 计算并合并所有指标
        self._calculate_advanced_metrics()

        # 5. 生成 Phase 2 所需的 4 个候选清单
        self._generate_phase2_candidates()

        # 6. 生成并导出统计报告 (Phase 1 Outputs)
        self._generate_statistical_reports()

        print("\n" + "=" * 60)
        print("深度挖掘完成！所有清单与统计报告已生成。")
        print("=" * 60)

    def _generate_statistical_reports(self):
        """生成 Phase 1 要求的具体统计报告产出物"""
        print("\n[Deep Mining] 生成统计报告 (Fingerprints & Content DNA)...")

        # 1. 活跃节律 (Activity Heatmap)
        # 聚合所有用户的活跃小时分布
        total_hours = Counter()
        for user, stats in self.behavior_stats.items():
            total_hours.update(stats['active_hours'])

        heatmap_data = []
        for hour in range(24):
            heatmap_data.append({
                'hour': hour,
                'activity_count': total_hours[hour],
                'activity_percentage': total_hours[hour] / sum(total_hours.values()) if sum(total_hours.values()) > 0 else 0
            })
        pd.DataFrame(heatmap_data).to_csv(f'{self.output_dir}/stats_activity_heatmap.csv', index=False)
        print(f"  ✓ stats_activity_heatmap.csv")

        # 2. 内容格式效能 (Content Efficiency by Media Type)
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

        # 3. 流量漏斗 (Traffic Funnel)
        # 计算全局转化率
        if self.content_stats:
            df_content = pd.DataFrame(self.content_stats)
            total_views = df_content['view_count'].sum()
            funnel = {
                'total_views': total_views,
                'total_likes': df_content['like_count'].sum(),
                'total_replies': df_content['reply_count'].sum(),
                'total_retweets': df_content.get('retweet_count', pd.Series([0]*len(df_content))).sum(), # content_stats 可能没存 retweet? 检查下
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

    def _scan_behavior_and_content(self):
        """扫描回复文件，提取行为指纹和内容指标"""
        print("\n[Deep Mining] 扫描行为指纹与内容基因...")

        reply_files = glob.glob(f"{self.replies_dir}/TwExport_*_Replies.csv")

        for file_path in tqdm(reply_files, desc="  深度扫描文件"):
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                df.columns = df.columns.str.strip()

                # 转换时间
                df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')

                # 上一行规则辅助列
                df['prev_author'] = df['Author Username'].shift(1)
                df['prev_created_at'] = df['Created At'].shift(1)
                df['prev_text'] = df['Text'].shift(1) # 用于上下文

                for i, row in df.iterrows():
                    user = row['Author Username']
                    if pd.isna(user): continue

                    # --- 行为统计 ---

                    # 1. 活跃节律 (Hour)
                    if pd.notna(row['Created At']):
                        hour = row['Created At'].hour
                        self.behavior_stats[user]['active_hours'][hour] += 1

                    # 2. 专业度 (Source)
                    if pd.notna(row['Source']):
                        self.behavior_stats[user]['sources'][row['Source']] += 1

                    # 3. 互动类型 (Reply vs Origin)
                    # 注意：CSV中 Type='Reply' 表示这是回复，Type='Origin' 表示这是原创
                    # 但有时 Reply 也是 Origin (Thread)，这里按 Type 字段判断
                    t_type = row['Type']
                    if t_type == 'Reply':
                        self.behavior_stats[user]['reply_count'] += 1

                        # 4. 回复延迟 (Latency) - 仅对 Reply 有效
                        # 规则：当前行是 Reply，上一行是 Target
                        target_user = row['prev_author']
                        if pd.notna(target_user) and pd.notna(row['Created At']) and pd.notna(row['prev_created_at']):
                            # 确保不是自言自语 (Thread)
                            if user != target_user:
                                latency = (row['Created At'] - row['prev_created_at']).total_seconds()
                                if latency > 0:
                                    self.behavior_stats[user]['latencies'].append(latency)

                                # 记录互动对用于互惠性分析
                                # (Source, Target, Type, Timestamp, Text)
                                self.interaction_pairs.append({
                                    'source': user,
                                    'target': target_user,
                                    'timestamp': row['Created At'],
                                    'text': row['Text']
                                })

                    elif t_type == 'Origin':
                        self.behavior_stats[user]['origin_count'] += 1
                    elif t_type == 'Retweet':
                        self.behavior_stats[user]['retweet_count'] += 1

                    # --- 内容统计 (针对该推文本身的数据) ---
                    # 我们关心的是这条推文的表现
                    view_count = float(row.get('View Count', 0)) if pd.notna(row.get('View Count')) else 0
                    like_count = float(row.get('Favorite Count', 0)) if pd.notna(row.get('Favorite Count')) else 0
                    bookmark_count = float(row.get('Bookmark Count', 0)) if pd.notna(row.get('Bookmark Count')) else 0
                    reply_count = float(row.get('Reply Count', 0)) if pd.notna(row.get('Reply Count')) else 0
                    retweet_count = float(row.get('Retweet Count', 0)) if pd.notna(row.get('Retweet Count')) else 0

                    media_type = row.get('media_type', 'none')
                    if pd.isna(media_type): media_type = 'none'

                    self.behavior_stats[user]['media_types'][media_type] += 1
                    self.behavior_stats[user]['total_views'] += view_count
                    self.behavior_stats[user]['total_likes'] += like_count
                    self.behavior_stats[user]['total_bookmarks'] += bookmark_count
                    self.behavior_stats[user]['total_replies_received'] += reply_count
                    self.behavior_stats[user]['total_retweets_received'] += retweet_count

                    # 收集每一条帖子用于 "list_posts_outliers" 和 "content_opportunities"
                    # 计算 Utility Score (Bookmark / Like)
                    utility_score = 0
                    if like_count > 0:
                        utility_score = bookmark_count / like_count

                    # 内容机会：是问句吗？
                    text = str(row.get('Text', ''))
                    is_question = '?' in text or '？' in text

                    self.content_stats.append({
                        'id': row.get('ID', ''), # 假设有ID列，如果没有可能需要用索引
                        'author': user,
                        'text': text,
                        'created_at': row.get('Created At'),
                        'type': t_type,
                        'media_type': media_type,
                        'view_count': view_count,
                        'like_count': like_count,
                        'bookmark_count': bookmark_count,
                        'reply_count': reply_count, # 这条帖子收到的回复数
                        'utility_score': utility_score,
                        'is_question': is_question,
                        'source': row.get('Source', '')
                    })

            except Exception as e:
                # print(f"  Error scanning {file_path}: {e}")
                pass

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
        """生成 Phase 2 所需的 4 个清单"""
        print("\n[Deep Mining] 生成 Phase 2 候选清单...")
        os.makedirs(self.output_dir, exist_ok=True)

        # 清单 1: list_posts_outliers.csv (异常价值帖子)
        # 筛选标准：干货指数 Top 50 OR 浏览量 Top 50 OR 讨论率极高
        df_content = pd.DataFrame(self.content_stats)
        if not df_content.empty:
            # 排除无效内容
            df_content = df_content[df_content['text'].str.len() > 10]

            # 策略A：高干货 (Utility > 0.5 且 Like > 5)
            high_utility = df_content[(df_content['utility_score'] > 0.5) & (df_content['like_count'] > 5)]

            # 策略B：高流量
            high_traffic = df_content.sort_values('view_count', ascending=False).head(50)

            # 策略C：高讨论 (Reply/View 高)
            df_content['discussion_rate'] = df_content['reply_count'] / (df_content['view_count'] + 1)
            high_discuss = df_content.sort_values('discussion_rate', ascending=False).head(50)

            # 合并并去重
            outliers = pd.concat([high_utility.head(50), high_traffic, high_discuss]).drop_duplicates(subset=['id', 'text'])
            outliers.to_csv(f'{self.output_dir}/list_posts_outliers.csv', index=False, encoding='utf-8-sig')
            print(f"  ✓ list_posts_outliers.csv ({len(outliers)} 条)")

        # 清单 2: list_users_key_players.csv (关键角色)
        # 包含：PageRank Top, Betweenness Top, Rising Star Top, Professional Top
        df_users = pd.DataFrame(self.users_profile.values())
        if not df_users.empty:
            key_players = pd.concat([
                df_users.sort_values('pagerank', ascending=False).head(50),
                df_users.sort_values('betweenness', ascending=False).head(50),
                df_users.sort_values('rising_star_velocity', ascending=False).head(30),
                df_users.sort_values('avg_utility_score', ascending=False).head(20)
            ]).drop_duplicates(subset=['username'])

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
        if not df_content.empty:
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
                unanswered_questions['opportunity_type'] = 'unanswered_question'

                # 机会 B: 激烈讨论 (Reply > 10)
                hot_debates = df_content[df_content['reply_count'] > 10].copy()
                hot_debates['opportunity_type'] = 'hot_debate'

                opportunities = pd.concat([unanswered_questions.head(50), hot_debates.head(50)])
                opportunities.to_csv(f'{self.output_dir}/list_content_opportunities.csv', index=False, encoding='utf-8-sig')
                print(f"  ✓ list_content_opportunities.csv ({len(opportunities)} 条)")

if __name__ == '__main__':
    miner = DeepMiner()
    miner.run_deep_analysis()
