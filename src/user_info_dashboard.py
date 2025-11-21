# dashboard.py
# 这是一个使用 Plotly Dash 构建的交互式社交网络分析仪表盘。
#
# --- 安装依赖 ---
# 在运行此脚本前，请确保已安装所需库：
# pip install pandas dash dash-bootstrap-components
#
import json
import os
import pandas as pd
import networkx as nx

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

# --- 数据加载与准备 ---
def load_data(network_data_path='output/network_data.json', top_n_percent=None):
    """
    加载并准备分析所需的数据。
    可选择只加载PageRank最高的N%的用户以提高性能。
    """
    if not os.path.exists(network_data_path):
        raise FileNotFoundError(f"错误: 未找到输入文件 '{network_data_path}'。请先运行 'macro_analysis.py'。")

    with open(network_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes_df = pd.DataFrame(data['nodes'])
    edges_df = pd.DataFrame(data['edges'])

    # 如果指定了 top_n_percent，则进行过滤
    if top_n_percent is not None and 0 < top_n_percent <= 1:
        print(f"原始用户数: {len(nodes_df)}")
        nodes_df = nodes_df.sort_values(by='pagerank', ascending=False)
        num_to_keep = int(len(nodes_df) * top_n_percent)
        top_nodes_df = nodes_df.head(num_to_keep)
        
        top_nodes_ids = set(top_nodes_df['id'])
        
        # 仅保留顶尖用户之间的边
        original_edge_count = len(edges_df)
        edges_df = edges_df[
            edges_df['source'].isin(top_nodes_ids) & edges_df['target'].isin(top_nodes_ids)
        ]
        
        nodes_df = top_nodes_df
        
        print(f"数据已过滤，仅显示PageRank最高的 {top_n_percent*100:.0f}% 的用户 ({len(nodes_df)} 个)。")
        print(f"边数量从 {original_edge_count} 减少到 {len(edges_df)}。")


    # 为表格创建可点击的Markdown链接
    nodes_df['id_md'] = nodes_df['id'].apply(lambda i: f'[@{i}](https://x.com/{i})')
    
    return nodes_df, edges_df

# --- APP 初始化与数据加载 ---
try:
    # 只加载 PageRank 最高的 15% 的用户，以提高性能和响应速度
    nodes_df, edges_df = load_data(top_n_percent=0.15)
except FileNotFoundError as e:
    print(e)
    exit()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "X/Twitter 网络分析仪表盘"


# --- 样式定义 ---

# 统一的表格样式，提升美感和可读性
TABLE_STYLE = {
    'style_header': {
        'backgroundColor': '#343a40',
        'fontWeight': 'bold',
        'color': 'white',
        'border': '1px solid #454d55'
    },
    'style_data': {
        'backgroundColor': '#23272b',
        'color': 'white',
        'border': '1px solid #454d55'
    },
    'style_cell': {
        'textAlign': 'left',
        'padding': '12px',
        'fontFamily': 'sans-serif',
    },
    'style_as_list_view': True, # 移除单元格竖线，更像列表
    'style_data_conditional': [
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': '#2c3034' # 实现条纹效果
        },
        {
            'if': {'state': 'selected'},
            'backgroundColor': 'rgba(0, 123, 255, 0.4)', # 选中行高亮
            'border': '1px solid #007bff'
        }
    ],
}

# --- 辅助函数 ---
def build_user_card_content(user_data):
    """构建用户详情卡片的内部组件列表，采用更美观的布局。"""
    if user_data is None:
        return html.P("无有效用户信息。", className="text-muted")

    return [
        html.H4(user_data['name'], className="text-center mb-1"),
        html.P(f"@{user_data['id']}", className="text-center text-muted small"),
        html.P(user_data['bio'] or "无个人简介", className="my-3 text-center small"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Strong("PageRank"),
                html.H5(f"{user_data['pagerank']:.2e}", className="text-info")
            ], width=6, className="text-center"),
            dbc.Col([
                html.Strong("粉丝数"),
                html.H5(f"{user_data['followers_count']:,}", className="text-light")
            ], width=6, className="text-center"),
        ], className="my-3"),
        html.Hr(),
        dbc.Row([
             dbc.Col(html.A("访问X主页", href=f"https://x.com/{user_data['id']}", target="_blank", className="btn btn-outline-primary w-100")),
        ]),
        html.P(f"账号创建于: {user_data['created_at'].split(' ')[0]}", className="small text-center text-muted mt-3")
    ]

# --- APP 布局定义 ---
header = html.Div([
    html.H2("X/Twitter 影响力网络分析", className="display-4"),
    html.P("一个用于探索和分析核心用户及其关注网络的交互式仪表盘", className="lead")
], className="bg-dark text-light p-5 rounded-3 mb-4")

app.layout = dbc.Container(fluid=True, className="p-4", children=[
    header,
    dbc.Row([
        # 2.1 左侧主面板 (用户列表 + 关注列表)
        dbc.Col(md=12, lg=8, children=[
            dbc.Card(body=True, className="mb-4", children=[
                html.H4("核心用户列表 (Top 15% PageRank)", className="card-title mb-3"),
                dash_table.DataTable(
                    id='user-table',
                    columns=[
                        {"name": "用户名", "id": "id_md", 'presentation': 'markdown'},
                        {"name": "姓名", "id": "name"},
                        {"name": "PageRank", "id": "pagerank", 'type': 'numeric', 'format': {'specifier': '.2e'}},
                        {"name": "粉丝数", "id": "followers_count", 'type': 'numeric', 'format': {'specifier': ','}},
                    ],
                    data=nodes_df.to_dict('records'),
                    sort_action="native",
                    filter_action="native",
                    row_selectable="single",
                    page_size=15,
                    markdown_options={"link_target": "_blank"},
                    **TABLE_STYLE
                )
            ]),
            dbc.Card(body=True, children=[
                html.H4("关注的最具影响力用户", id="followed-list-title", className="card-title mb-3"),
                html.P("在上方表格中选择一个用户，此处将显示其关注的最重要用户列表。", id="followed-list-placeholder", className="text-muted"),
                dash_table.DataTable(
                    id='followed-users-table',
                    columns=[
                        {"name": "用户名", "id": "id_md", 'presentation': 'markdown'},
                        {"name": "姓名", "id": "name"},
                        {"name": "PageRank", "id": "pagerank", 'type': 'numeric', 'format': {'specifier': '.2e'}},
                        {"name": "粉丝数", "id": "followers_count", 'type': 'numeric', 'format': {'specifier': ','}},
                    ],
                    data=[],
                    sort_action="native",
                    row_selectable="single",
                    page_size=10,
                    markdown_options={"link_target": "_blank"},
                    **TABLE_STYLE
                )
            ])
        ]),
        
        # 2.2 右侧详情面板 (拆分为两个)
        dbc.Col(md=12, lg=4, children=[
            dbc.Card(body=True, className="mb-4", children=[
                html.H4("主选用户详情", className="card-title mb-3"),
                html.Div(id='user-detail-panel-main', children=[
                    html.P("请从左侧核心列表中选择一个用户。", className="text-muted")
                ])
            ]),
            dbc.Card(body=True, children=[
                html.H4("关注者详情", className="card-title mb-3"),
                html.Div(id='user-detail-panel-followed', children=[
                    html.P("从“关注列表”中选择一位用户以查看。", className="text-muted")
                ])
            ])
        ])
    ])
])

# --- 回调函数定义 (APP的交互逻辑) ---

# 回调1: 更新下方“关注列表”
@app.callback(
    [Output('followed-users-table', 'data'),
     Output('followed-list-placeholder', 'style'),
     Output('followed-list-title', 'children')],
    [Input('user-table', 'selected_rows')],
    [State('user-table', 'data')]
)
def update_followed_list(selected_rows, table_data):
    if not selected_rows:
        return [], {'display': 'block'}, "关注的最具影响力用户"

    selected_user_id = table_data[selected_rows[0]]['id']
    outgoing_edges = edges_df[edges_df['source'] == selected_user_id]
    followed_ids = set(outgoing_edges['target'])

    if not followed_ids:
        title = f"@{selected_user_id} 未关注列表中的任何用户"
        return [], {'display': 'none'}, title

    followed_users_df = nodes_df[nodes_df['id'].isin(followed_ids)].copy()
    top_followed_df = followed_users_df.sort_values(by='pagerank', ascending=False).head(100)
    table_records = top_followed_df.to_dict('records')
    title = f"@{selected_user_id} 关注的 Top {len(top_followed_df)} 用户 (按PageRank排序)"

    return table_records, {'display': 'none'}, title

# 回调2: 统一更新右侧两个“详情”面板
@app.callback(
    [Output('user-detail-panel-main', 'children'),
     Output('user-detail-panel-followed', 'children')],
    [Input('user-table', 'selected_rows'),
     Input('followed-users-table', 'selected_rows')],
    [State('user-table', 'data'),
     State('followed-users-table', 'data'),
     State('user-detail-panel-main', 'children')]
)
def update_both_detail_panels(main_rows, followed_rows, main_data, followed_data, main_panel_content):
    
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'

    placeholder_main = html.P("请从左侧核心列表中选择一个用户。", className="text-muted")
    placeholder_followed = html.P("从“关注列表”中选择一位用户以查看。", className="text-muted")

    if trigger_id == 'user-table':
        if not main_rows:
            return placeholder_main, placeholder_followed
        
        user_id = main_data[main_rows[0]]['id']
        user_data = nodes_df[nodes_df['id'] == user_id].iloc[0]
        new_main_panel_content = build_user_card_content(user_data)
        
        return new_main_panel_content, placeholder_followed

    elif trigger_id == 'followed-users-table':
        if not followed_rows:
            return main_panel_content, placeholder_followed
        
        user_id = followed_data[followed_rows[0]]['id']
        user_data = nodes_df[nodes_df['id'] == user_id].iloc[0]
        new_followed_panel_content = build_user_card_content(user_data)

        return main_panel_content, new_followed_panel_content

    return placeholder_main, placeholder_followed


# --- 运行 APP ---
if __name__ == '__main__':
    print("仪表盘正在启动...")
    print("请在浏览器中打开 http://127.0.0.1:8050")
    app.run(debug=True)
