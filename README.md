# X社交网络深度洞察与增长策略分析计划

## 1. 项目概述

### 1.1. 核心目标
本项目旨在对一个特定的X（原Twitter）影响力用户群落进行深度数据挖掘与分析，最终将洞察转化为**可执行、个性化的社交账户增长策略**。项目采用“漏斗式”分析架构，结合**Python 广域挖掘**与**LLM 深度推理**，实现全流程闭环。

### 1.2. 核心分析架构
为了平衡分析的深度、效率与成本，我们采用两阶段架构：

1.  **阶段一：广域统计与算法挖掘 (Deep Mining)**
    *   **工具**: `deep_mining.py` (基于 Python, NetworkX, Louvain)。
    *   **目标**: 全量处理数据，构建网络，计算影响力，挖掘行为模式与内容基因。
    *   **产出**: 完整的用户清单、网络图谱、统计报告。

2.  **阶段二：LLM驱动的定性增强 (LLM Deep Analysis)**
    *   **工具**: `deep_analysis_llm.py` (基于 OpenAI Compatible API)。
    *   **目标**: 对阶段一筛选出的“Top N”候选清单进行深度的归因、推理和策略生成。
    *   **产出**: 爆款拆解、策略画像、关系内涵、内容机会库。

---

## 2. 项目结构与脚本说明

```bash
.
├── macro_analysis.py         # [基础] 宏观网络构建与基础指标计算
├── deep_mining.py            # [核心] 阶段一：深度挖掘、社群发现、行为/内容统计 (执行此脚本)
├── deep_analysis_llm.py      # [核心] 阶段二：LLM 定性分析 (依赖阶段一产出)
├── network_visualization.py  # [可视化] 生成交互式网络图
├── llm_client.py             # LLM 客户端封装
├── config.py                 # 配置管理
├── prompts.py                # LLM 提示词库
└── output/                   # 所有产出物目录
```

### 2.1. 运行指南

1.  **准备环境**:
    ```bash
    pip install -r requirements.txt

    # 复制配置模板
    cp config.example.ini config.ini
    # 编辑 config.ini 填入 API Key
    # 或者直接设置环境变量: export OPENAI_API_KEY="sk-..."
    ```

2.  **执行阶段一 (Deep Mining)**:
    ```bash
    python3 deep_mining.py
    ```
    *   *功能*: 解压数据 -> 构建网络 -> 计算PageRank/Betweenness -> 社群发现 -> 行为扫描 -> 生成CSV清单。

3.  **执行阶段二 (LLM Analysis)**:
    ```bash
    python3 deep_analysis_llm.py
    ```
    *   *功能*: 读取阶段一生成的 CSV -> 并发调用 LLM -> 生成 JSON 洞察报告。

---

## 3. 产出物清单 (Artifacts)

所有文件均位于 `output/` 目录下。

### 3.1. 用户清单 (Watchlists)
| 文件名 | 描述 | 筛选逻辑 |
|:---|:---|:---|
| `watchlist_authorities.csv` | **权威枢纽** | 按 PageRank 排序的全局影响者。 |
| `watchlist_connectors.csv` | **破圈者** | 按 Betweenness Centrality 排序的跨圈桥梁。 |
| `watchlist_rising_stars.csv` | **崛起新星** | 账号较新但影响力增长极快的用户。 |
| `list_users_key_players.csv` | **关键角色(综合)** | 包含上述维度及高专业度、高干货贡献者的精选名单 (Phase 2 输入)。 |
| `all_users_with_metrics.csv` | **全量用户画像** | 包含所有计算指标的完整用户表。 |

### 3.2. 内容与行为分析 (Content & Behavior)
| 文件名 | 描述 | 关键指标 |
|:---|:---|:---|
| `list_posts_outliers.csv` | **异常价值帖子** | 高干货指数(Bookmark/Like)、高流量或高争议的帖子。 |
| `list_content_opportunities.csv` | **内容机会** | 无人回答的高价值提问、热门辩论话题。 |
| `stats_activity_heatmap.csv` | **活跃节律报告** | 社区整体的 24小时活跃度分布。 |
| `stats_content_efficiency.csv` | **内容效能报告** | 不同媒体类型(Text/Image/Video)的互动表现对比。 |
| `stats_traffic_funnel.csv` | **流量漏斗报告** | View -> Like -> Reply 的全局转化率。 |

### 3.3. 关系与互动 (Interactions)
| 文件名 | 描述 |
|:---|:---|
| `list_interactions_strong_ties.csv` | **强互惠关系对** | 双向互动频繁的用户对及其互动样本。 |

### 3.4. LLM 深度洞察 (LLM Insights)
位于 `output/llm_insights/` 目录：
| 文件名 | 描述 |
|:---|:---|
| `Content_Blueprints.json` | **爆款拆解**：钩子类型、结构模式、情绪基调分析。 |
| `User_Strategy_Dossiers.json` | **策略画像**：用户人设、沟通风格、潜在目标推断。 |
| `Community_Insights.json` | **关系内涵**：互动背后的社交意图（请教、辩论、吹捧）。 |
| `Content_Idea_Bank.json` | **灵感题库**：基于未满足需求生成的具体选题建议。 |

### 3.5. 可视化
| 文件名 | 描述 |
|:---|:---|
| `network_combined.graphml` | 完整的网络图文件 (可用 Gephi 打开)。 |
| `network_data.json` | 适配 Web 展示的 JSON 数据。 |

---

## 4. 详细分析维度说明

### 4.1. 行为指纹 (Behavioral Fingerprinting)
- **活跃节律**: 分析 `Created At` 字段，生成 `stats_activity_heatmap.csv`，指导最佳发帖时间。
- **专业度指数**: 基于 `Source` (Typefully vs Web App) 识别专业创作者。
- **互动/广播比**: 计算 Reply / Origin 推文比例，区分"广播台"与"社群粘合剂"。
- **回复延迟**: 计算 Reply 与上一行 Target 的时间差，衡量用户活跃度。

### 4.2. 内容基因 (Content DNA)
- **干货指数 (Utility Score)**: `Bookmark Count / Favorite Count`。收藏代表实用价值，点赞代表情绪共鸣。
- **内容效能**: 统计不同 `media_type` 的转化率，生成 `stats_content_efficiency.csv`。
- **流量漏斗**: 全局计算 `View -> Like` 和 `View -> Reply` 转化率，生成 `stats_traffic_funnel.csv`。

### 4.3. LLM 定性分析
利用 GPT-4/Gemini 等模型，对筛选出的高价值数据进行文本层面的深度理解，提取结构化洞察（详见 `deep_analysis_llm.py` 和 `prompts.py`）。
