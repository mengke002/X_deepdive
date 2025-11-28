#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发现潜在新用户脚本
基于数据库中的互动关系，发现尚未被采集但在网络中活跃的用户
"""

import pandas as pd
import os
from collections import defaultdict
import logging
from .db_adapter import get_db_adapter

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NewUserDiscoverer:
    """
    发现潜在的新用户，这些用户尚未被采集数据但在互动网络中表现活跃。

    核心逻辑：
    - 使用数据库中的 in_reply_to_tweet_id 精确识别回复关系
    - 找到被核心用户回复但自身不是核心用户的外部用户
    - 使用回复者的PageRank作为权重进行排序
    """
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, 'all_users_with_metrics.csv')
        self.output_file = os.path.join(output_dir, 'watchlist_potential_new_users.csv')

        self.core_users = set()  # 核心用户集合（已采集数据的用户）
        self.all_network_users = set()  # 整个网络中的所有用户
        self.user_pagerank = {}
        
        # 初始化数据库适配器
        self.db_adapter = get_db_adapter()
        if not self.db_adapter:
            raise RuntimeError("数据库连接失败，请检查 DATABASE_URL 或 DB_* 环境变量配置")

    def _load_existing_users_and_metrics(self):
        """加载核心用户集合和PageRank分数"""
        logging.info(f"Loading existing user metrics from {self.metrics_file}...")
        if not os.path.exists(self.metrics_file):
            logging.error(f"Metrics file not found: {self.metrics_file}. Please run macro_analysis.py first.")
            return False

        # 加载所有网络用户的PageRank
        df_metrics = pd.read_csv(self.metrics_file, encoding='utf-8-sig')
        self.all_network_users = set(df_metrics['username'].str.lower())
        self.user_pagerank = df_metrics.set_index('username')['pagerank'].to_dict()
        logging.info(f"Loaded {len(self.all_network_users)} users from network metrics.")

        # 从数据库获取核心用户
        self.core_users = self.db_adapter.get_core_users()
        self.core_users = {u.lower() for u in self.core_users}
        logging.info(f"Identified {len(self.core_users)} core users (users with collected data).")
        return True

    def _process_replies(self):
        """
        使用数据库中的回复关系，发现被核心用户回复的外部用户

        核心逻辑：
        1. 使用 in_reply_to_tweet_id 精确识别回复关系
        2. 如果回复者是核心用户，而被回复者不是核心用户，则后者是潜在新用户
        3. 使用回复者的PageRank作为权重累加到被回复者的分数上
        """
        logging.info("Processing reply relationships from database...")
        
        # 获取回复关系
        reply_df = self.db_adapter.get_reply_relationships()
        
        if reply_df.empty:
            logging.warning("No reply relationships found in database.")
            return None

        external_user_scores = defaultdict(float)
        external_user_reply_count = defaultdict(int)
        total_external_found = 0

        for _, row in reply_df.iterrows():
            source_user = row['source_user']  # 回复者
            target_user = row['target_user']  # 被回复者

            if pd.notna(source_user) and pd.notna(target_user):
                source_user_lower = source_user.lower()
                target_user_lower = target_user.lower()

                # 核心判断：回复者是核心用户，被回复者不是核心用户
                if (source_user_lower in self.core_users and
                    target_user_lower not in self.core_users):

                    # 获取回复者的PageRank作为权重
                    author_pagerank = self.user_pagerank.get(source_user, 0)

                    if author_pagerank > 0:
                        external_user_scores[target_user] += author_pagerank
                        external_user_reply_count[target_user] += 1
                        total_external_found += 1

        logging.info(f"发现 {total_external_found} 次对外部用户的回复")
        return external_user_scores, external_user_reply_count

    def discover_and_rank(self):
        """Main method to run the discovery and ranking process."""
        if not self._load_existing_users_and_metrics():
            return

        result = self._process_replies()

        if not result or not result[0]:
            logging.info("No new external users were discovered.")
            return

        external_user_scores, external_user_reply_count = result

        # Convert to DataFrame for sorting and saving
        data = []
        for user in external_user_scores:
            data.append({
                'Username': user,
                'WeightedReplyScore': external_user_scores[user],
                'ReplyCount': external_user_reply_count[user],
                'AvgReplierPageRank': external_user_scores[user] / external_user_reply_count[user] if external_user_reply_count[user] > 0 else 0
            })

        df_new_users = pd.DataFrame(data)
        df_new_users = df_new_users.sort_values(by='WeightedReplyScore', ascending=False).reset_index(drop=True)

        # Save the results
        os.makedirs(self.output_dir, exist_ok=True)
        df_new_users.to_csv(self.output_file, index=False, encoding='utf-8-sig')
        logging.info(f"Successfully discovered {len(df_new_users)} new potential users.")
        logging.info(f"Sorted list saved to: {self.output_file}")

        # Print top 20 for quick review
        print("\n--- Top 20 潜在新用户（基于回复关系） ---")
        print(df_new_users.head(20).to_string())
        print("---------------------------------------------")


if __name__ == '__main__':
    discoverer = NewUserDiscoverer()
    discoverer.discover_and_rank()
