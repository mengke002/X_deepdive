# X社交网络深度洞察与增长策略分析计划

## 1. 项目概述

本项目旨在对一个特定的X（原Twitter）影响力用户群落进行深度数据挖掘与分析，最终将洞察转化为**可执行、个性化的社交账户增长策略**。项目采用“漏斗式”分析架构，结合**Python 广域挖掘**与**LLM 深度推理**，实现全流程闭环。

## 2. 核心架构与功能

### 2.1. 项目结构
```bash
X_deepdive/
├── src/                      # 核心源代码目录
│   ├── deep_mining.py        # [核心入口] 阶段一：深度挖掘、社群发现、行为/内容统计
│   ├── deep_analysis_llm.py  # [核心入口] 阶段二：LLM 定性分析
│   ├── macro_analysis.py     # [基础模块] 宏观网络构建与基础指标计算
│   ├── discover_new_users.py # [工具] 潜在新用户发现
│   ├── network_visualization.py # [工具] 交互式网络可视化
│   ├── user_info_dashboard.py   # [工具] 用户信息仪表盘
│   ├── llm_client.py         # LLM 客户端封装 (支持双轨模型)
│   ├── config.py             # 配置管理
│   ├── prompts.py            # LLM 提示词库
│   └── utils.py              # 通用工具 (如数据解压)
├── config.example.ini        # 配置文件模板
├── output/                   # 所有产出物目录
├── X_followers.zip           # (可选) 关注数据压缩包
├── X_replies.zip             # (可选) 回复数据压缩包
└── requirements.txt          # 依赖列表
```

### 2.2. 数据处理机制
项目支持两种数据输入方式，并在运行时自动处理：
1.  **文件夹模式**：根目录下存在 `X_followers/` 和 `X_replies/` 文件夹。
2.  **压缩包模式**：根目录下存在 `X_followers.zip` 和 `X_replies.zip`。
    *   *智能解压*：所有分析脚本启动时会自动检查文件夹是否存在，若不存在则尝试解压对应的Zip文件。这使得项目在GitHub Actions或未解压的本地环境中均可直接运行。

---

## 3. 快速开始

### 3.1. 环境准备
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 LLM (可选，仅阶段二需要)
# 方式 A: 环境变量 (推荐，适用于 GitHub Actions)
export OPENAI_API_KEY="sk-..."

# 方式 B: 配置文件
cp config.example.ini config.ini
# 编辑 config.ini 填入 API Key
```

### 3.2. 运行分析

**阶段一：广域统计与算法挖掘 (Deep Mining)**
此阶段无需 LLM Key，完全本地运行。
```bash
python -m src.deep_mining
```
*   **功能**: 全量加载数据 -> 构建社交网络 -> 计算 PageRank/Betweenness -> Louvain 社群发现 -> 行为指纹扫描 -> 生成所有 CSV 清单和统计报告。

**阶段二：LLM 驱动的定性增强 (LLM Analysis)**
此阶段需要配置 API Key。
```bash
python -m src.deep_analysis_llm
```
*   **功能**: 读取阶段一生成的 `output/list_*.csv` -> 并发调用 LLM (支持 Fast/Deep 模型自动切换) -> 生成深度 JSON 洞察报告。

**其他工具**:
```bash
# 生成交互式网络图 HTML
python -m src.network_visualization

# 发现潜在新用户 (未采集但高互动)
python -m src.discover_new_users

# 启动本地仪表盘 (Web UI)
python -m src.user_info_dashboard
```

---

## 4. 产出物清单 (Artifacts)

所有产出物均位于 `output/` 目录下：

### 4.1. 核心洞察 (LLM Generated)
位于 `output/llm_insights/`:
*   `Content_Blueprints.json`: **爆款拆解** (钩子、结构、情绪)。
*   `User_Strategy_Dossiers.json`: **策略画像** (人设、目标、打法)。
*   `Community_Insights.json`: **关系内涵** (互动意图推理)。
*   `Content_Idea_Bank.json`: **机会挖掘** (高价值选题建议)。

### 4.2. 决策清单 (CSV)
*   `watchlist_authorities.csv`: **权威枢纽** (PageRank Top)。
*   `watchlist_connectors.csv`: **破圈者** (Betweenness Top)。
*   `watchlist_rising_stars.csv`: **崛起新星** (高增长率)。
*   `list_users_key_players.csv`: **关键角色综合榜**。
*   `list_posts_outliers.csv`: **异常价值内容** (高干货/高流量)。
*   `list_content_opportunities.csv`: **内容机会** (未被满足的提问)。

### 4.3. 统计报告
*   `stats_activity_heatmap.csv`: 社区活跃时间分布。
*   `stats_content_efficiency.csv`: 不同媒体类型效能分析。
*   `stats_traffic_funnel.csv`: 浏览->互动的全局转化漏斗。

### 4.4. 可视化文件
*   `network_top200.html`: 交互式网络图网页。
*   `network_data.json`: 网络数据源文件。

---

## 5. 配置说明

系统支持 **Fast/Deep 双轨模型策略**，以平衡成本和效果。可在 `config.ini` 或环境变量中配置：

*   **Fast Models** (用于关系推理、信息提取): 默认 `gemini-flash-lite-latest`, `glm-4.5-flash`。
*   **Deep Models** (用于深度分析、策略画像): 默认 `gemini-flash-latest`, `glm-4.5`。
*   **自动托底**: 系统会按顺序尝试模型，如果首选模型失败（如 429 Rate Limit），会自动切换到备选模型。
