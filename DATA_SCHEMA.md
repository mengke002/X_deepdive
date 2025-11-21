# 数据格式说明 (DATA_SCHEMA.md)

本项目依赖于两类核心数据：**用户关注数据 (`X_followers`)** 和 **用户回复数据 (`X_replies`)**。这些数据是构建社交网络图谱、分析用户行为以及进行 LLM 深度洞察的基础。

## 1. 数据源概览

### 1.1. 文件组织结构
数据通常以文件夹或 Zip 压缩包的形式提供。项目根目录支持以下两种结构，系统会自动检测并处理：

*   **文件夹模式**:
    *   `X_followers/`: 包含所有关注数据的 CSV 文件。
    *   `X_replies/`: 包含所有回复数据的 CSV 文件。
*   **压缩包模式**:
    *   `X_followers.zip`: 解压后应包含 CSV 文件。
    *   `X_replies.zip`: 解压后应包含 CSV 文件。

### 1.2. 文件命名规则
每个 CSV 文件都以**核心用户**（即数据的采集中心）为视角。

*   **关注数据**: `twitterExport_{Core_User_ID}_Following.csv`
    *   例如: `twitterExport_manateelazycat_Following.csv`
    *   含义: 用户 `manateelazycat` 关注的用户列表。
*   **回复数据**: `TwExport_{Core_User_ID}_Replies.csv`
    *   例如: `TwExport_manateelazycat_Replies.csv`
    *   含义: 用户 `manateelazycat` 的回复记录（及其上下文）。

---

## 2. 数据字段详解

### 2.1. 用户关注数据 (`Following.csv`)
此文件记录了核心用户关注的所有账号的详细 Profile 信息。用于构建**静态关注网络**和**用户画像**。

| 字段名 | 数据类型 | 描述 |
|:---|:---|:---|
| `User ID` | String/Int | 用户的唯一数字标识符。 |
| `Name` | String | 用户的显示名称（昵称）。 |
| `Username` | String | 用户的唯一用户名（Handle，不含 @）。 |
| `Bio` | String | 用户个人简介。是 LLM 分析用户策略和画像的核心文本。 |
| `Tweets Count` | Int | 发推总数。 |
| `Followers Count` | Int | 粉丝数量。衡量影响力的基础指标。 |
| `Following Count` | Int | 关注数量。 |
| `Favourites Count` | Int | 点赞（喜欢）的总数。 |
| `Media Count` | Int | 包含媒体（图片/视频）的推文数量。 |
| `Professional` | String | 职业类别（如 "Category: Technology"）。 |
| `Location` | String | 用户填写的地理位置。 |
| `Website` | String | 用户个人主页链接。 |
| `Verified` | Boolean | 是否通过认证。 |
| `Is Blue Verified` | Boolean | 是否是 Twitter Blue 付费认证。 |
| `Verified Type` | String | 认证类型（如 `Business`, `Government`, `None`）。 |
| `Created At` | String | 账号创建时间（如 `2011-01-20 21:04:12`）。用于计算账号年龄。 |
| `Profile URL` | String | X 个人主页链接。 |
| `Avatar URL` | String | 头像图片链接。 |
| `Profile Banner URL` | String | 个人主页背景图链接。 |

### 2.2. 用户回复数据 (`Replies.csv`)
此文件记录了核心用户的回复行为、内容及其上下文。用于构建**动态互动网络**、计算**内容效能**和进行**LLM 内容分析**。

**重要逻辑：回复关系识别（上一行规则）**
数据是按“对话流”组织的。要确定谁回复了谁，必须遵循以下规则：
1.  定位到 `Type` 为 `Reply` 的行。
2.  该回复的**目标推文**是其在文件中的**紧邻上一行**记录。
3.  因此，**被回复者**（Target User）是**上一行记录**的 `Author Username`。

| 字段名 | 数据类型 | 描述 |
|:---|:---|:---|
| `ID` | String | 推文的唯一 ID。 |
| `Text` | String | 推文内容。LLM 分析的核心输入。 |
| `Language` | String | 语言代码（如 `zh`, `en`）。 |
| `Type` | String | 推文类型：<br>- `Origin`: 原创推文。<br>- `Reply`: 回复。<br>- `Retweet`: 转推。<br>- `Quote`: 引用推文。 |
| `Author Name` | String | 作者显示名称。 |
| `Author Username` | String | 作者用户名。 |
| `View Count` | Int | 浏览量。 |
| `Reply Count` | Int | 收到的回复数。 |
| `Retweet Count` | Int | 被转推数。 |
| `Quote Count` | Int | 被引用数。 |
| `Favorite Count` | Int | 点赞数。 |
| `Bookmark Count` | Int | 收藏数。计算“干货指数”的关键指标。 |
| `Created At` | String | 推文发布时间。用于分析活跃节律和回复延迟。 |
| `Tweet URL` | String | 推文链接。 |
| `Source` | String | 客户端来源（如 `Twitter for iPhone`, `Typefully`）。用于分析专业度。 |
| `hashtags` | String | 包含的话题标签。 |
| `urls` | String | 包含的外部链接。 |
| `media_type` | String | 媒体类型（`photo`, `video`, `gif`, 或空）。 |
| `media_urls` | String | 媒体文件链接。 |

---

## 3. `X_replies`数据说明

在分析`X_replies`数据时，一个常见的误区是试图通过正则表达式从回复的`Text`字段中提取`@username`来确定回复对象。**这是一个严重错误的方法**，因为用户可以编辑回复中的`@`提及，甚至完全删除它，导致关系识别不准确。

正确的、可靠的方法根植于采集数据的结构本身。

**核心原理：数据是按“对话流”组织的**

`X_replies`目录下的每个CSV文件都是从特定用户的“回复”页面按时间顺序从上到下采集的。这个页面不仅包含该用户的回复，还为了提供上下文，包含了被回复的原始推文。这使得整个CSV文件呈现为一种**扁平化的对话线程**。

**准确识别回复关系的基本规则：**

1.  在CSV文件中，当您定位到一行`Type`为`Reply`的记录时，它代表一个回复动作。
2.  **该回复所指向的目标推文，就是它在文件中的紧邻的上一行记录。**
3.  因此，被回复用户的唯一身份标识（`Username`），就是**上一行记录**的`Author Username`字段的值。

**实例解析：**

假设在`TwExport_Bitturing_Replies.csv`中我们有如下数据：

| Row | Type   | Author Username | Text                      |
|:----|:-------|:----------------|:--------------------------|
| 10  | Origin | `punk2898`      | RNM，Stable 这 SB 项目方... |
| 11  | **Reply**  | **`Bitturing`**     | @punk2898 多少？100wu？   |

- 第11行是`Bitturing`发起的一个回复。
- 根据规则，它回复的是第10行的推文。
- 第10行推文的作者是`punk2898`。
- **结论：** 这是一次`Bitturing -> punk2898`的互动。我们应使用`punk2898`作为关系图中的目标节点。

**挖掘动态过程：**

这个“上一行规则”是重建整个动态对话链的基础。通过从上到下处理文件，我们可以精确地构建出谁在何时回复了谁，形成一个有向的互动序列（例如：A -> B -> C -> B），从而进行更深度的时序和行为分析。

---

## 4. 数据使用注意事项

1.  **缺失值处理**:
    *   `Bio`, `Location`, `Website` 等字段可能为空。脚本处理时需做空值检查。
    *   `View Count` 等互动指标可能为 0 或空（旧推文），需填充为 0 处理。

2.  **字符编码**:
    *   所有 CSV 文件应使用 `utf-8-sig` 编码读取，以兼容包含中文和 Emoji 的内容。

3.  **隐私与脱敏**:
    *   虽然数据均为公开推文，但在进行 LLM 分析时，避免上传与分析任务无关的敏感个人信息。目前的 Prompt 设计主要关注公开的观点、策略和互动模式。