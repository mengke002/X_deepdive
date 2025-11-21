#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式网络可视化脚本
使用Pyvis生成可交互的HTML网络图
"""

import pandas as pd
import networkx as nx
import json
from pyvis.network import Network
import os


def load_network_data(json_path='output/network_data.json'):
    """加载网络数据"""
    print("加载网络数据...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  - 节点数: {len(data['nodes'])}")
    print(f"  - 边数: {len(data['edges'])}")
    return data


def create_interactive_network(network_data, output_path='output/network_interactive.html', top_n=200):
    """
    创建交互式网络可视化

    Args:
        network_data: 网络数据（包含nodes和edges）
        output_path: 输出HTML文件路径
        top_n: 只展示影响力最高的前N个节点
    """
    print(f"\n创建交互式网络可视化 (Top {top_n} 用户)...")

    # 按PageRank排序，选择Top N用户
    nodes_df = pd.DataFrame(network_data['nodes'])
    nodes_df['pagerank'] = nodes_df['pagerank'].fillna(0)
    top_nodes = nodes_df.nlargest(top_n, 'pagerank')
    top_usernames = set(top_nodes['username'].values)

    # 过滤边（只保留两端都在Top N中的边）
    edges_df = pd.DataFrame(network_data['edges'])
    filtered_edges = edges_df[
        (edges_df['source'].isin(top_usernames)) &
        (edges_df['target'].isin(top_usernames))
    ]

    print(f"  - 筛选后节点数: {len(top_nodes)}")
    print(f"  - 筛选后边数: {len(filtered_edges)}")

    # 创建Pyvis网络对象
    net = Network(
        height='1000px',
        width='100%',
        bgcolor='#222222',
        font_color='white',
        directed=True,
        notebook=False
    )

    # 添加节点
    print("  添加节点...")
    pagerank_max = top_nodes['pagerank'].max()
    pagerank_min = top_nodes['pagerank'].min()

    for _, node in top_nodes.iterrows():
        username = node['username']
        name = node.get('name', username) if pd.notna(node.get('name')) else username
        bio = str(node.get('bio', ''))[:200] if pd.notna(node.get('bio')) else ''
        followers = int(node.get('followers_count', 0)) if pd.notna(node.get('followers_count')) else 0
        pagerank = float(node.get('pagerank', 0)) if pd.notna(node.get('pagerank')) else 0
        betweenness = float(node.get('betweenness', 0)) if pd.notna(node.get('betweenness')) else 0

        # 节点大小映射PageRank（10-50像素）
        if pagerank_max > pagerank_min:
            size = 10 + 40 * (pagerank - pagerank_min) / (pagerank_max - pagerank_min)
        else:
            size = 25

        # 节点颜色（暂时使用单一颜色，后续可以根据社群ID着色）
        color = '#00BFFF'

        # 获取更多用户数据
        following = int(node.get('following_count', 0)) if pd.notna(node.get('following_count')) else 0
        tweets = int(node.get('tweets_count', 0)) if pd.notna(node.get('tweets_count')) else 0
        verified = node.get('verified', False)
        verified_type = node.get('verified_type', '')

        # 构建Twitter URL
        twitter_url = f"https://twitter.com/{username}"

        # 认证标记
        verified_badge = ''
        if verified and verified_type == 'blue':
            verified_badge = '✓'
        elif verified and verified_type == 'government':
            verified_badge = '⚪'
        elif verified and verified_type == 'business':
            verified_badge = '⭐'
        elif verified:
            verified_badge = '✓'

        # 格式化数字显示
        def format_number(num):
            if num >= 1_000_000:
                return f"{num/1_000_000:.1f}M"
            elif num >= 1_000:
                return f"{num/1_000:.1f}K"
            return str(num)

        # 构建精简的节点卡片（使用CSS类）
        title = f"""<div class="tc">
<div class="th"><b>{name}</b> {verified_badge}<br>@{username}</div>
<div class="tb">{bio[:100]+'...' if len(bio) > 100 else bio}</div>
<div class="ts">
<div class="si"><b>{format_number(followers)}</b><br>粉丝</div>
<div class="si"><b>{format_number(following)}</b><br>关注</div>
<div class="si"><b>{format_number(tweets)}</b><br>推文</div>
</div>
<div class="tm">PageRank: <b>{pagerank:.4f}</b><br>中介: <b>{betweenness:.4f}</b></div>
<a href="{twitter_url}" target="_blank" class="tl">🐦 查看主页</a>
</div>"""

        net.add_node(
            username,
            label=f"@{username}",
            title=title,
            size=size,
            color=color
        )

    # 添加边
    print("  添加边...")
    for _, edge in filtered_edges.iterrows():
        source = edge['source']
        target = edge['target']
        weight = edge.get('weight', 1)

        # 边的宽度映射权重
        width = min(1 + weight * 0.1, 5)

        net.add_edge(source, target, width=width, color='rgba(255,255,255,0.2)')

    # 设置交互选项（平衡性能和美观）
    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 12,
          "face": "Arial"
        },
        "scaling": {
          "min": 10,
          "max": 50
        }
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.5
          }
        },
        "smooth": {
          "type": "continuous",
          "roundness": 0.5
        }
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.1
        },
        "maxVelocity": 50,
        "minVelocity": 0.75,
        "solver": "barnesHut",
        "timestep": 0.5,
        "stabilization": {
          "enabled": true,
          "iterations": 500,
          "updateInterval": 50,
          "onlyDynamicEdges": false,
          "fit": true
        },
        "adaptiveTimestep": true
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 50,
        "hideEdgesOnDrag": true,
        "hideEdgesOnZoom": false,
        "navigationButtons": true,
        "keyboard": true,
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      }
    }
    """)

    # 保存HTML文件
    print(f"  保存可视化文件: {output_path}")
    net.save_graph(output_path)

    # 添加自定义CSS样式和标题
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # CSS样式（全局定义，减少重复）
    custom_css = """<style>
.tc{font-family:-apple-system,sans-serif;max-width:280px;background:linear-gradient(135deg,#1e3c72,#2a5298);border-radius:10px;box-shadow:0 4px 20px rgba(0,0,0,0.4);color:#fff}
.th{background:linear-gradient(90deg,#1DA1F2,#0d8bd9);padding:12px;border-radius:10px 10px 0 0;font-size:15px}
.tb{padding:10px 12px;font-size:12px;line-height:1.4;border-left:3px solid #1DA1F2;margin:8px 12px;color:rgba(255,255,255,0.9)}
.ts{display:flex;gap:8px;padding:8px 12px;justify-content:space-around}
.si{background:rgba(29,161,242,0.2);padding:8px;border-radius:6px;text-align:center;font-size:11px;flex:1}
.si b{display:block;font-size:16px;color:#1DA1F2;margin-bottom:2px}
.tm{background:rgba(0,0,0,0.3);padding:10px 12px;margin:8px 12px;border-radius:6px;font-size:11px}
.tm b{color:#FFD700}
.tl{display:block;background:linear-gradient(90deg,#1DA1F2,#0d8bd9);color:#fff;text-align:center;padding:10px;margin:8px 12px 12px;border-radius:6px;text-decoration:none;font-size:12px;font-weight:600}
.tl:hover{opacity:0.9}
/* 修复tooltip样式 */
.vis-tooltip{pointer-events:auto!important;z-index:9999!important}
</style>"""

    # JavaScript修复tooltip闪烁问题并停止物理引擎
    custom_js = """<script>
document.addEventListener('DOMContentLoaded', function() {
    // 延迟tooltip消失，避免闪烁
    var tooltipElement = null;
    var hideTimeout = null;

    document.addEventListener('mouseover', function(e) {
        if (e.target.closest('.vis-tooltip')) {
            clearTimeout(hideTimeout);
        }
    });

    // 监听vis.js tooltip
    setTimeout(function() {
        var observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                mutation.addedNodes.forEach(function(node) {
                    if (node.classList && node.classList.contains('vis-tooltip')) {
                        tooltipElement = node;
                        // 阻止tooltip立即消失
                        node.style.pointerEvents = 'auto';
                        clearTimeout(hideTimeout);
                    }
                });
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }, 1000);

    // 稳定后自动停止物理引擎
    if (typeof network !== 'undefined') {
        network.on('stabilizationIterationsDone', function() {
            network.setOptions({ physics: false });
            console.log('物理引擎已停止 - 网络已稳定');
        });

        // 备用方案：5秒后强制停止（防止永远稳定不了）
        setTimeout(function() {
            network.setOptions({ physics: false });
            console.log('物理引擎已强制停止');
        }, 5000);
    }
});
</script>"""

    # 在HTML中添加CSS、JS和标题
    header = f"""
    <div style="position: absolute; top: 10px; left: 10px; z-index: 1000; background: rgba(0,0,0,0.7); padding: 15px; border-radius: 5px; color: white;">
        <h2 style="margin: 0 0 10px 0;">X社交网络可视化分析</h2>
        <p style="margin: 5px 0; font-size: 14px;">节点大小 = 影响力 (PageRank)</p>
        <p style="margin: 5px 0; font-size: 14px;">边的粗细 = 互动频率</p>
        <p style="margin: 5px 0; font-size: 14px;">显示Top {top_n}最具影响力用户</p>
        <p style="margin: 5px 0; font-size: 12px; color: #aaa;">提示: 拖动节点、滚轮缩放、悬停查看详情</p>
    </div>
    """

    # 注入CSS到head，注入JS和header到body
    html_content = html_content.replace('</head>', custom_css + '</head>')
    html_content = html_content.replace('<body>', '<body>\n' + header)
    html_content = html_content.replace('</body>', custom_js + '</body>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  ✓ 可视化完成!")
    print(f"  在浏览器中打开: {output_path}")


def create_multiple_views(network_data, output_dir='output'):
    """创建多个视角的可视化"""
    print("\n" + "=" * 60)
    print("创建多视角网络可视化")
    print("=" * 60)

    # 1. 完整视图（Top 200）
    create_interactive_network(
        network_data,
        output_path=f'{output_dir}/network_top200.html',
        top_n=200
    )

    # 2. 精简视图（Top 100）
    create_interactive_network(
        network_data,
        output_path=f'{output_dir}/network_top100.html',
        top_n=100
    )

    # 3. 核心视图（Top 50）
    create_interactive_network(
        network_data,
        output_path=f'{output_dir}/network_top50.html',
        top_n=50
    )

    print("\n" + "=" * 60)
    print("可视化创建完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print(f"  - {output_dir}/network_top200.html (Top 200用户)")
    print(f"  - {output_dir}/network_top100.html (Top 100用户)")
    print(f"  - {output_dir}/network_top50.html (Top 50用户)")


def main():
    """主函数"""
    # 加载网络数据
    network_data = load_network_data()

    # 创建多视角可视化
    create_multiple_views(network_data)


if __name__ == '__main__':
    main()
