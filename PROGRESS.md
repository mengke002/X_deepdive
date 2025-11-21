# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个**X/Twitter社交网络分析框架**，旨在分析有影响力的Twitter用户群落，并生成可执行的增长策略。项目通过宏观层面的网络分析，识别社交网络中的关键影响者、破圈者和崛起新星。

**核心目标**：将Twitter网络的数据洞察转化为个性化、可执行的社交媒体增长策略，综合运用定量分析与LLM驱动的定性推理。

## 常用命令

### 运行分析

```bash
# 运行完整的宏观层面分析
python3.12 macro_analysis.py

# 生成交互式网络可视化
python3.12 network_visualization.py

# 发现新的潜在用户
python3.12 discover_new_users.py

# 启动交互式用户分析仪表盘
python3.12 user_info_dashboard.py
```

### 数据要求

- 将关注数据文件放在 `X_followers/` 目录，文件命名格式：`twitterExport_{用户名}_Following.csv`
- 将回复数据文件放在 `X_replies/` 目录，文件命名格式：`TwExport_{用户名}_Replies.csv`

## 核心架构

### 数据处理流程

```
原始CSV数据 → MacroNetworkAnalyzer → 网络图谱构建 → 指标计算 → 用户清单 + 可视化
```

### 核心组件

#### 1. `macro_analysis.py` - 宏观网络分析器

主分析引擎，执行以下功能：

- **加载数据**：读取 `X_followers/` 和 `X_replies/` 目录下的所有CSV文件
- **构建网络图谱**：
  - `G_static`：关注关系的有向图（静态网络）
  - `G_dynamic`：互动/回复关系的有向图（动态网络）
  - `G_combined`：静态和动态网络的加权组合
- **计算核心指标**：
  - **PageRank**：全局影响力，识别"权威枢纽"
  - **中介中心性 (Betweenness Centrality)**：识别连接不同社群的"信息桥梁"/"破圈者"
  - **入度 (In-degree)**：被关注数
- **生成用户清单**：输出三个按不同指标排序的完整CSV文件
- **导出网络数据**：导出GraphML和JSON格式，供可视化工具使用

**核心类**：`MacroNetworkAnalyzer`
- 构造函数参数：`followers_dir`（默认：'X_followers'），`replies_dir`（默认：'X_replies'）
- 主方法：`run_full_analysis()` - 执行完整分析流程

**重要属性**：
- `core_users`：核心用户集合（从文件名提取）
- `users_profile`：字典，存储所有用户的画像数据
- `G_static`, `G_dynamic`, `G_combined`：三个NetworkX有向图

#### 2. `network_visualization.py` - 交互式网络可视化

使用Pyvis生成交互式HTML网络图：

- 从 `output/network_data.json` 加载网络数据
- 创建三个视角：Top 50、Top 100、Top 200 最具影响力用户
- **视觉映射规则**：
  - 节点大小 = PageRank分数（影响力）
  - 节点颜色 = 社群ID（当前为单一颜色，预留社群检测功能）
  - 边的粗细 = 互动权重
  - 悬停信息 = 用户详细画像（姓名、简介、粉丝数、PageRank等）

**核心函数**：`create_interactive_network(network_data, output_path, top_n)`
- 筛选Top N用户（按PageRank）后再可视化，确保图表清晰可读
- 使用力导向布局（ForceAtlas2算法）

**关键设计**：为保证浏览器性能和视觉清晰度，网络图默认只展示影响力最高的一批用户（50/100/200），完整数据保存在CSV/JSON文件中。

#### 3. `discover_new_users.py` - 新用户发现器

根据回复关系和回复者的PageRank分数，发现并排序潜在的新用户。

- **加载数据**：读取 `output/all_users_with_metrics.csv` 获取网络用户的PageRank，从数据文件名提取核心用户集合。
- **处理回复**：遍历 `X_replies/` 目录下的所有CSV文件，使用"上一行规则"识别真实的回复关系。
- **过滤与评分**：
  - 筛选核心用户（已采集数据的用户）发出的回复。
  - 找出被回复者不在核心用户集合中的情况。
  - 对这些外部用户，根据回复者的PageRank分数进行加权评分。
- **生成清单**：输出一个按"影响力加权回复分数"排序的潜在新用户CSV文件，包含被回复次数和平均回复者PageRank。

**核心类**：`NewUserDiscoverer`
- 构造函数参数：`replies_dir`（默认：'X_replies'），`followers_dir`（默认：'X_followers'），`output_dir`（默认：'output'）
- 主方法：`discover_and_rank()` - 执行完整发现和排序流程。

#### 4. `ANALYSIS_FRAMEWORK.md` - 完整分析方法论

详细的中文分析框架文档，定义了三层分析方法：

- **宏观层面**：网络结构、关键角色、长期趋势
- **中观层面**：社群发现、热门内容、话题建模
- **微观层面**：个体用户策略画像

**包含内容**：
- 完整的分析流程和产出模块定义
- 附录A：LLM分析任务指令集，包含以下专业提示词：
  - 关系内涵推理 (Relationship Inference)
  - 用户策略画像 (User Strategy Profiling)
  - 爆款内容解构 (Viral Content Deconstruction)
  - 内容机会挖掘 (Content Opportunity Mining)

#### 5. `user_info_dashboard.py` - 交互式用户分析仪表盘

一个使用 Plotly Dash 构建的交互式Web应用，用于探索和分析网络中的核心用户。

- **数据加载**：从 `output/network_data.json` 加载数据，并为提高性能默认只显示PageRank最高的15%的用户。
- **核心功能**：
    - **主用户列表**：一个可搜索、可排序、可筛选的表格，展示了网络中的核心影响力用户。
    - **关注分析**：当在主列表选中一个用户时，下方会显示该用户所关注的人中，影响力最高的用户列表（按PageRank排序）。
    - **用户详情卡片**：在界面右侧展示选中用户（主列表或关注列表中的用户）的详细信息，包括姓名、简介、PageRank、粉丝数，并提供直接访问其X主页的链接。
- **技术栈**：`dash`, `dash-bootstrap-components`, `pandas`
- **运行方式**：
  ```bash
  # 启动仪表盘
  python3.12 user_info_dashboard.py
  ```

### 数据结构说明

#### 输入CSV格式

**关注数据** (`X_followers/twitterExport_*_Following.csv`)：
- 关键字段：`Username`（用户名）、`Name`（姓名）、`Bio`（个人简介）、`Followers Count`（粉丝数）、`Following Count`（关注数）、`Tweets Count`（推文数）、`Verified`（认证状态）、`Verified Type`（认证类型）、`Created At`（账号创建时间）、`Location`（位置）、`Website`（网站）、`Professional`（职业信息）

**回复数据** (`X_replies/TwExport_*_Replies.csv`)：
- 关键字段：`Type`（记录类型：Origin/Reply/Tweet）、`Author Username`（作者用户名）、`Text`（回复内容）、`View Count`（浏览量）、`Reply Count`（回复数）、`Retweet Count`（转发数）、`Favorite Count`（点赞数）、`Created At`（创建时间）、`Source`（来源）、`media_type`（媒体类型）
- **重要**：数据按对话流组织，当一行的Type为Reply时，其回复目标是上一行的作者

#### 输出文件

所有输出保存在 `output/` 目录：

**用户清单**（完整排序列表）：
- `watchlist_authorities.csv` - 按PageRank排序的"权威枢纽"清单（全局影响者）
- `watchlist_connectors.csv` - 按中介中心性排序的"破圈者"清单（跨社群桥梁）
- `watchlist_rising_stars.csv` - 按增长率排序的"崛起新星"清单（账号年龄<2年，按 PageRank/账号年龄 排序）
- `watchlist_potential_new_users.csv` - 按影响力加权回复分数排序的"潜在新用户"清单

**网络数据**：
- `all_users_with_metrics.csv` - 包含所有计算指标的完整用户画像
- `network_combined.graphml` - GraphML格式网络图（可用Gephi等工具打开）
- `network_data.json` - JSON格式，包含nodes和edges，供Web可视化使用

**可视化文件**：
- `network_top50.html` - Top 50用户的交互式网络图
- `network_top100.html` - Top 100用户的交互式网络图
- `network_top200.html` - Top 200用户的交互式网络图

### 技术栈

- **核心分析**：Python，使用pandas、NetworkX、numpy
- **可视化**：Pyvis（生成交互式HTML网络图）
- **计划中/未来**：
  - LLM集成用于定性分析（提示词已在ANALYSIS_FRAMEWORK.md中定义）
  - Seaborn/Pyecharts用于统计图表
  - 社群检测（Louvain算法）
  - 话题建模（LDA/BERTopic）

### 实施阶段

**Phase 1（当前已完成）**：基础网络构建、核心指标计算、用户清单生成、交互式可视化

**Phase 2（计划中）**：LLM驱动的关系推理、用户策略画像、社群文化分析、爆款内容解构

**Phase 3（计划中）**：策略生成、综合报告撰写、创作者增长手册

## 重要说明

### 数据加载模式

脚本会**遍历数据目录中的所有文件**来构建完整的网络视图。每个CSV文件代表从一个核心用户视角收集的数据，但分析会聚合所有文件来捕捉完整的社群关系。

### 回复数据的"上一行规则"

`X_replies` 的CSV文件是扁平化的对话流。当一行的 `Type` 为 `Reply` 时，它回复的目标就是它的上一行记录的作者。这是构建动态互动网络的核心规则，确保了回复关系的精确识别。

### 可视化性能考虑

网络可视化刻意限制为Top N用户（50/100/200），以保持浏览器性能和视觉清晰度。完整的网络数据保存在CSV/JSON输出文件中。

### 账号年龄计算

"崛起新星"清单筛选账号年龄小于730天（2年）的用户，并计算增长率 = PageRank / (账号年龄/365)。这个指标衡量单位时间内的影响力增长速度。

### 语言

项目所有沟通、注释、输出均使用中文。代码变量名和函数名使用英文。

## 文件结构

```
.
├── ANALYSIS_FRAMEWORK.md    # 完整分析方法论文档（中文）
├── macro_analysis.py         # 宏观网络分析主脚本
├── network_visualization.py  # 网络可视化脚本
├── discover_new_users.py     # 新用户发现脚本
├── user_info_dashboard.py    # 交互式用户分析仪表盘
├── X_followers/              # 关注数据目录（若干CSV文件）
├── X_replies/                # 回复数据目录（若干CSV文件）
├── X_posts/                  # 帖子数据目录（未使用）
├── output/                   # 输出目录
│   ├── watchlist_*.csv       # 用户清单
│   ├── all_users_with_metrics.csv
│   ├── network_combined.graphml
│   ├── network_data.json
│   └── network_top*.html     # 可视化文件
└── lib/                      # 第三方库（Pyvis相关）
```
