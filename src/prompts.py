"""
LLM 分析提示词库 (Prompt Library)
包含 Phase 2 所需的所有 Prompt 模板

基于 ANALYSIS_MINING_PLAN.md 扩展：
- Thread 结构分析
- 商业模式解码
- 智能截流回复生成
"""

# A.1 关系内涵推理
PROMPT_RELATIONSHIP_INFERENCE = """
# 角色
你是一名精通社交网络分析的心理学专家。

# 任务
分析以下用户之间的互动文本，判断发送者对接收者的社交意图和关系内涵。

# 规则
1. 仔细理解回复的语气、内容和上下文。
2. 从预定义的`relationship_type`列表中选择最恰当的一个。
3. 对关系的强度`strength`进行1-5分的打分（1分表示非常微弱/礼节性，5分表示非常强烈/紧密）。
4. "description" 字段请用中文简要概括这段关系的本质（如"互相吹捧", "激烈辩论", "单方面请教"）。
5. 只输出纯 JSON 格式，不要包含 Markdown 标记（```json ... ```）。

# 预定义关系类型
- "QUESTIONING": 提问、请教、寻求信息。
- "AGREEMENT_SUPPORT": 赞同、支持、附议、表达敬意。
- "RESPECTFUL_DEBATE": 提出不同见解、进行有礼貌的辩论或补充。
- "GRATITUDE": 单纯表示感谢。
- "SOCIAL_PRAISE": 社交性的称赞、吹捧或无实质内容的互动。
- "CASUAL_CHAT": 闲聊、开玩笑或非正式的对话。
- "OTHER": 其他无法明确归类的关系。

# 输入数据 (互动样本)
{input_data}

# 输出格式 (JSON)
{{
  "relationship_type": "...",
  "strength": 1-5,
  "description": "..."
}}
"""

# A.2 用户策略画像
PROMPT_USER_STRATEGY = """
# 角色
你是一名顶级的社交媒体战略分析师和用户研究专家。

# 任务
基于用户的个人简介（Bio）和近期一系列的回复内容，深度分析并推理该用户的社交媒体运营策略、核心画像与潜在目标。

# 规则
1. 通读所有输入材料，形成一个整体印象。
2. 对各项指标进行评估和分类，选择最贴切的描述。
3. "summary" 字段必须包含具体的策略建议或观察，而不仅仅是泛泛而谈。
4. 只输出纯 JSON 格式，不要包含 Markdown 标记。

# 输入数据
{input_data}

# 输出格式 (JSON)
{{
  "persona": "...", // 用户画像, 如 "思想领袖", "社群粘合剂", "好斗的辩论者", "信息聚合者", "教程贡献者", "产品推广者", "新手学习者"
  "communication_style": "...", // 沟通风格, 如 "循循善诱的提问者", "观点鲜明的陈述者", "幽默风趣", "严谨正式", "数据驱动"
  "content_focus": ["...", "..."], // 内容焦点, 提取3-5个核心话题关键词
  "inferred_goal": "...", // 潜在目标, 如 "打造个人品牌", "为产品/网站引流", "寻求技术/观点交流", "建立行业人脉", "记录个人学习"
  "argumentation_style": "...", // 论证方式, 如 "逻辑与数据", "故事与案例", "情感与共鸣", "引用权威"
  "summary": "..." // 用一句话总结该用户的核心特征与策略
}}
"""

# A.3 爆款内容解构
PROMPT_VIRAL_CONTENT = """
# 角色
你是一名深谙社交媒体传播之道的病毒式营销专家。

# 任务
解构以下这篇在社交网络上表现优异的帖子，分析其写作结构、风格和技巧。

# 规则
1. 将帖子内容拆分为不同部分进行分析。
2. 为每个分析维度选择最合适的标签。
3. 重点分析"为什么它会火"，在 "why_viral" 字段中给出深度见解。
4. 只输出纯 JSON 格式，不要包含 Markdown 标记。

# 输入数据
{input_data}

# 输出格式 (JSON)
{{
  "hook_style": {{
    "type": "...", // 开头钩子类型, 如 "反常识观点", "惊人数据", "普适性痛点", "直接提问", "故事开场"
    "text": "..." // 钩子的具体文本
  }},
  "body_structure": "...", // 正文结构, 如 "编号列表式", "总-分-总", "问题-解决方案-收益(P-S-B)", "时间线/故事弧线", "多角度对比"
  "readability_features": ["...", "..."], // 可读性特征, 如 "大量使用换行", "使用Emoji作为项目符号", "句子简短", "使用粗体或特殊符号强调"
  "emotional_tone": "...", // 情绪基调, 如 "启发性/励志", "制造焦虑/紧迫感", "幽默/自嘲", "客观/中立", "引人共鸣"
  "call_to_action": {{
    "type": "...", // 结尾行动号召, 如 "开放式提问", "引导关注", "引导点击链接", "请求转发/点赞", "无明确CTA"
    "text": "..." // CTA的具体文本
  }},
  "why_viral": "..." // 深度解析：为什么这篇帖子能获得高互动？
}}
"""

# A.4 内容机会挖掘
PROMPT_CONTENT_OPPORTUNITY = """
# 角色
你是一名嗅觉敏锐的市场分析师和内容策略师。

# 任务
阅读以下海量的用户对话文本或未回答的问题，从中挖掘出潜在的内容创作机会。

# 规则
1. 识别并提取所有明确的或隐含的“问题”。
2. 识别并总结讨论最激烈、观点最两极分化的“辩论点”。
3. 给出具体的"content_suggestion"（内容创作建议），即如果你是创作者，你会写什么标题的文章来回应这个需求。
4. 只输出纯 JSON 格式，不要包含 Markdown 标记。

# 输入数据
{input_data}

# 输出格式 (JSON)
{{
  "opportunity_type": "...", // "unanswered_question" 或 "hot_debate"
  "core_topic": "...", // 核心话题
  "user_intent": "...", // 提问者或讨论参与者的核心意图
  "content_suggestion": {{
     "title": "...", // 建议的文章标题
     "angle": "..." // 建议的切入角度
  }}
}}
"""

# =====================================================
# B. 新增 Prompts (基于 ANALYSIS_MINING_PLAN.md)
# =====================================================

# B.1 Thread 结构分析
PROMPT_THREAD_ANALYSIS = """
# 角色
你是一名精通社交媒体内容策略的增长专家，擅长分析推文串（Thread）的结构美学。

# 任务
分析以下 Thread 的结构，提取其成功的关键因素，并生成可复用的写作模板。

# 规则
1. 分析 Thread 的开头钩子（Hook）设计如何吸引读者继续阅读。
2. 分析正文的逻辑结构和节奏（信息密度、情绪起伏）。
3. 分析结尾的收尾方式和行动号召（CTA）。
4. 给出一个可复用的 Thread 写作模板框架。
5. 只输出纯 JSON 格式，不要包含 Markdown 标记。

# 输入数据
{input_data}

# 输出格式 (JSON)
{{
  "hook_analysis": {{
    "type": "...", // 钩子类型: "shocking_stat", "contrarian_take", "promise", "story_start", "question"
    "technique": "...", // 使用的技巧
    "strength": 1-5 // 钩子强度评分
  }},
  "body_structure": {{
    "pattern": "...", // 结构模式: "numbered_list", "problem_solution", "before_after", "framework", "story_arc"
    "pacing": "...", // 节奏: "fast_punchy", "building_tension", "steady_flow"
    "info_density": "..." // 信息密度: "high", "medium", "low"
  }},
  "cta_analysis": {{
    "type": "...", // CTA类型: "follow", "retweet", "reply", "link_click", "none"
    "placement": "...", // 位置: "end_only", "middle_and_end", "throughout"
    "effectiveness": 1-5
  }},
  "retention_factors": ["...", "..."], // 留存因素列表
  "replication_template": "...", // 可复用的写作模板（用占位符表示变量部分）
  "best_practices": ["...", "..."] // 最佳实践总结
}}
"""

# B.2 商业模式解码
PROMPT_MONETIZATION_DECODE = """
# 角色
你是一名深谙创作者经济的商业分析师，擅长逆向工程社交媒体大V的变现路径。

# 任务
基于用户的内容发布行为和外部链接使用模式，推理其商业模式和变现策略。

# 规则
1. 分析用户使用的各类外部链接（Newsletter、产品、社群等）。
2. 推理其流量漏斗的设计逻辑。
3. 识别其主要变现渠道和辅助变现渠道。
4. 给出可借鉴的变现策略建议。
5. 只输出纯 JSON 格式，不要包含 Markdown 标记。

# 输入数据
{input_data}

# 输出格式 (JSON)
{{
  "primary_monetization": {{
    "channel": "...", // 主要变现渠道: "newsletter", "course", "saas", "consulting", "sponsorship", "community"
    "evidence": "...", // 支持这个判断的证据
    "estimated_stage": "..." // 估计阶段: "early", "growth", "mature"
  }},
  "secondary_monetization": [
    {{
      "channel": "...",
      "role": "..." // 在变现系统中的角色: "lead_gen", "upsell", "retention"
    }}
  ],
  "funnel_structure": {{
    "top_of_funnel": "...", // 流量入口类型
    "middle_of_funnel": "...", // 信任建立方式
    "bottom_of_funnel": "..." // 转化方式
  }},
  "content_to_cash_ratio": "...", // 内容与变现内容的比例判断: "heavy_value", "balanced", "heavy_pitch"
  "replicable_tactics": ["...", "..."], // 可复制的策略
  "risk_factors": ["...", "..."] // 潜在风险点
}}
"""

# B.3 智能截流回复生成
PROMPT_SMART_REPLY = """
# 角色
你是一名精通社交媒体互动艺术的增长黑客，擅长撰写高价值评论来建立影响力。

# 任务
针对以下正在快速获得关注的推文，生成一条高质量的回复草稿，目标是：
1. 为原推文增加价值（补充信息、提供不同视角、分享经验）
2. 展示专业性，吸引原作者和其他读者的注意
3. 自然地建立个人品牌曝光

# 规则
1. 回复必须与原推文高度相关，不能偏题。
2. 回复应该简洁有力，控制在280字符以内。
3. 避免纯粹的吹捧或附和，要有独特价值。
4. 可以适当使用提问或邀请讨论的方式增加互动。
5. 语气要自然，不能像机器人或营销号。
6. 只输出纯 JSON 格式，不要包含 Markdown 标记。

# 输入数据
{input_data}

# 输出格式 (JSON)
{{
  "reply_draft": "...", // 回复草稿文本
  "reply_strategy": "...", // 回复策略: "add_value", "share_experience", "ask_question", "provide_data", "offer_alternative_view"
  "value_proposition": "...", // 这条回复能提供的价值
  "engagement_hook": "...", // 设计的互动钩子
  "tone": "...", // 语气: "professional", "friendly", "thoughtful", "witty"
  "confidence_score": 1-5 // 对这条回复质量的自信度
}}
"""

# B.4 内容资产四象限深度分析
PROMPT_ASSET_QUADRANT_ANALYSIS = """
# 角色
你是一名内容策略专家，精通内容资产的分类和价值评估。

# 任务
分析以下内容在"价值-情绪"矩阵中的定位，并给出内容优化建议。

# 内容资产四象限定义
- **Library（复利资产）**: 高书签/低点赞，实用价值高，适合长期复用的教程、资源类内容
- **Controversy（争议爆款）**: 高引用/高传播，观点鲜明，容易引发讨论的内容
- **News（即时资讯）**: 高浏览/低互动，时效性强但长尾价值低的新闻类内容
- **Cult（圈层黑话）**: 高互动/低浏览，深度共鸣但受众面窄的小众内容

# 规则
1. 准确判断内容所属象限。
2. 分析内容在该象限中的优劣势。
3. 给出将内容转化为更高价值象限的建议（如果适用）。
4. 只输出纯 JSON 格式，不要包含 Markdown 标记。

# 输入数据
{input_data}

# 输出格式 (JSON)
{{
  "quadrant": "...", // "library", "controversy", "news", "cult"
  "quadrant_fit_score": 1-5, // 与该象限的匹配度
  "strengths": ["...", "..."], // 在该象限的优势
  "weaknesses": ["...", "..."], // 在该象限的劣势
  "optimization_suggestions": [
    {{
      "target_quadrant": "...", // 建议转化的目标象限
      "how_to_transform": "...", // 如何转化
      "expected_improvement": "..." // 预期提升
    }}
  ],
  "long_term_value_score": 1-5, // 长期复利价值评分
  "replication_difficulty": "..." // 复制难度: "easy", "medium", "hard"
}}
"""
