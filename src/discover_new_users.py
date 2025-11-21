
import pandas as pd
import os
import glob
from collections import defaultdict
import logging
from .utils import ensure_data_ready

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NewUserDiscoverer:
    """
    发现潜在的新用户，这些用户尚未被采集数据但在互动网络中表现活跃。

    核心逻辑：
    - 使用"上一行规则"识别真实的回复关系
    - 找到被核心用户回复但自身不是核心用户的外部用户
    - 使用回复者的PageRank作为权重进行排序
    """
    def __init__(self, replies_dir='X_replies', followers_dir='X_followers', output_dir='output'):
        # 确保数据就绪
        ensure_data_ready(followers_dir, replies_dir)

        self.replies_dir = replies_dir
        self.followers_dir = followers_dir
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, 'all_users_with_metrics.csv')
        self.output_file = os.path.join(output_dir, 'watchlist_potential_new_users.csv')

        self.core_users = set()  # 核心用户集合（已采集数据的用户）
        self.all_network_users = set()  # 整个网络中的所有用户
        self.user_pagerank = {}

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

        # 从文件名提取核心用户（已被采集数据的用户）
        follower_files = glob.glob(os.path.join(self.followers_dir, 'twitterExport_*_Following.csv'))
        reply_files = glob.glob(os.path.join(self.replies_dir, 'TwExport_*_Replies.csv'))

        for f in follower_files:
            core_user = os.path.basename(f).replace('twitterExport_', '').replace('_Following.csv', '')
            self.core_users.add(core_user.lower())

        for f in reply_files:
            core_user = os.path.basename(f).replace('TwExport_', '').replace('_Replies.csv', '')
            self.core_users.add(core_user.lower())

        logging.info(f"Identified {len(self.core_users)} core users (users with collected data).")
        return True

    def _process_replies(self):
        """
        使用"上一行规则"处理回复文件，发现被核心用户回复的外部用户

        核心逻辑：
        1. 只关注Type为Reply的行
        2. 回复的目标是上一行的作者
        3. 如果回复者是核心用户，而被回复者不是核心用户，则后者是潜在新用户
        4. 使用回复者的PageRank作为权重累加到被回复者的分数上
        """
        logging.info(f"Processing replies in directory: {self.replies_dir}...")
        reply_files = glob.glob(os.path.join(self.replies_dir, 'TwExport_*_Replies.csv'))

        if not reply_files:
            logging.warning(f"No reply files found in {self.replies_dir}. Cannot discover new users.")
            return None

        external_user_scores = defaultdict(float)
        external_user_reply_count = defaultdict(int)  # 记录被回复次数
        total_replies_checked = 0
        total_external_found = 0

        for i, file_path in enumerate(reply_files):
            try:
                df_replies = pd.read_csv(file_path, encoding='utf-8-sig')

                # 清理列名
                df_replies.columns = df_replies.columns.str.strip()

                logging.info(f"[{i+1}/{len(reply_files)}] Processing {os.path.basename(file_path)} ({len(df_replies)} rows)...")

                if len(df_replies) == 0:
                    continue

                # 使用shift(1)创建上一行的作者列，这是回复的目标用户
                df_replies['target_user'] = df_replies['Author Username'].shift(1)

                # 筛选出所有Type为Reply的行
                replies_only = df_replies[df_replies['Type'] == 'Reply'].copy()

                # 过滤掉target_user为空的情况
                replies_only = replies_only.dropna(subset=['target_user'])

                total_replies_checked += len(replies_only)

                # 遍历每条回复
                for _, row in replies_only.iterrows():
                    source_user = row['Author Username']  # 回复者
                    target_user = row['target_user']       # 被回复者

                    # 确保用户名有效
                    if (pd.notna(source_user) and pd.notna(target_user) and
                        str(source_user).strip() != '' and str(target_user).strip() != ''):

                        source_user_lower = source_user.lower()
                        target_user_lower = target_user.lower()

                        # 核心判断：回复者是核心用户，被回复者不是核心用户
                        # 这意味着被回复者是一个值得采集数据的潜在新用户
                        if (source_user_lower in self.core_users and
                            target_user_lower not in self.core_users):

                            # 获取回复者的PageRank作为权重
                            author_pagerank = self.user_pagerank.get(source_user, 0)

                            if author_pagerank > 0:
                                external_user_scores[target_user] += author_pagerank
                                external_user_reply_count[target_user] += 1
                                total_external_found += 1

            except Exception as e:
                logging.error(f"Failed to process file {file_path}: {e}")

        logging.info(f"检查了 {total_replies_checked} 条回复，发现 {total_external_found} 次对外部用户的回复")
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
