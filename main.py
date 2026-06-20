import requests
from bs4 import BeautifulSoup
import datetime
from openai import OpenAI
import os
import re
import json

# --- 1. 核心配置 ---
client_llm = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK")

repo_full_name = os.getenv('GITHUB_REPOSITORY', 'owner/repo')
repo_owner = os.getenv('GITHUB_REPOSITORY_OWNER', 'owner')
repo_name = repo_full_name.split('/')[-1]
# 这里的 URL 会根据你的新仓库名自动变化
GITHUB_PAGES_URL = f"https://{repo_owner}.github.io/{repo_name}/"

CATEGORIES = ['cs.RO']

def scrape_arxiv(category):
    """抓取 Arxiv 数据，并提取日期前缀和总论文数"""
    url = f"https://arxiv.org/list/{category}/recent?show=500"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')
        dls = soup.find_all('dl', id='articles')
        if not dls: return None, 0, []
        
        # 提取标题日期
        raw_date_str = soup.find_all('h3')[0].text.strip()
        # raw_date_str = soup.find_all('h3')[1].text.strip() # yesterday
        match = re.search(r'^(.*)\(showing \d+ of (\d+) entries', raw_date_str)
        if match:
            date_prefix = match.group(1).strip()
            total_entries = match.group(2)
        else:
            date_prefix = raw_date_str
            total_entries = "0"

        papers = []
        dt_tags = dls[0].find_all('dt')
        dd_tags = dls[0].find_all('dd')
        # dt_tags = dls[1].find_all('dt') # yesterday
        # dd_tags = dls[1].find_all('dd') # yesterday
        
        for dt, dd in zip(dt_tags, dd_tags):
            link_tag = dt.find('a', title='Abstract')
            if not link_tag: continue
            id_str = link_tag.text.replace('arXiv:', '').strip()
            title = dd.find('div', class_='list-title').text.replace('Title:', '').strip()
            abstract = dd.find('p', class_='mathjax').text.strip() if dd.find('p', class_='mathjax') else ""
            papers.append({"id": id_str, "title": title, "abstract": abstract[:1000]})
        
        return {"prefix": date_prefix, "total": total_entries}, len(papers), papers
    except Exception: return None, 0, []

def get_arxiv_full_text(paper_id):
    """利用 Arxiv HTML 渲染功能抓取正文，并剔除参考文献以节省 token"""
    url = f"https://arxiv.org/html/{paper_id}"
    try:
        res = requests.get(url, timeout=20)
        if res.status_code != 200: return None
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 1. 移除脚本、样式等无关标签
        for script in soup(["script", "style"]):
            script.decompose()

        # 2. 核心修改：移除参考文献部分
        # Arxiv HTML 常见的参考文献标识符包括类名 'ltx_bibliography' 或 ID 'bib'
        ref_tags = soup.find_all(['section', 'div'], class_=re.compile(r'bibliography|references', re.I))
        ref_tags += soup.find_all(['section', 'div'], id=re.compile(r'bib|references', re.I))
        
        for tag in ref_tags:
            tag.decompose()
            print(f"[{paper_id}] 已剔除参考文献部分")

        # 3. 返回正文（保留前 30000 字符）
        return soup.get_text()[:30000] 
    except Exception as e:
        print(f"抓取全文出错 {paper_id}: {e}")
        return None

def only_filter_and_report(papers):
    """仅执行初筛，返回高分 ID 列表"""
    if not papers: return "今日无新论文。"
    
    all_filtered_papers = []
    for i in range(0, len(papers), 40):
        chunk = papers[i:i+40]
        
        filter_prompt = f"""你是一个专注于【大模型具身智能】的顶级研究员。请从以下论文中筛选出符合要求的论文，并为它们打分（1-10分，10分为极度相关）。

        ✅ 必须保留（相关度 7-10 分）：
        1. VLA (Vision-Language-Action)、World Models (世界模型)、World Modeling、视频生成。
        2. World-Action Model、Video-Action Model、Diffusion Policy。
        3. 具身 Scaling Laws、跨具身数据集、多模态融合注意力。
        
        🛑 需要剔除（直接丢弃，不要输出）：
        1. 经典控制(PID, MPC)、硬件/软体/步态研究。
        2. 传统导航、路径规划(A*, RRT)、SLAM、传感器标定。
        3. 垂直场景：深海、巡检、无人机/车、攀爬、医疗/手术机器人。
        4. 经典视觉：单纯的人体姿态识别、纯 3D 重建(NeRF/GS)、单纯触觉。
        5. 多智能体协同/集群 (Swarm)、离散任务调度。

        ⚠️ 输出极其严格限制：仅输出 JSON 格式数组，必须包含以下字段：
        [
          {{
            "id": "论文ID",
            "title_en": "英文原题",
            "title_zh": "中文翻译标题",
            "score": 9
          }}
        ]
        
        待处理数据：
        {json.dumps(chunk)}
        """
        
        try:
            completion = client_llm.chat.completions.create(
                model="qwen3.6-flash",  # qwen-flash, qwen3.6-plus, qwen3-max, qwen3.5-flash
                messages=[{"role": "user", "content": filter_prompt}]
            )
            res = completion.choices[0].message.content
            match = re.search(r'\[.*\]', res, re.DOTALL)
            if match:
                all_filtered_papers.extend(json.loads(match.group(0)))
        except Exception: pass

    # 过滤出 8 分以上的作为建议
    recommendations = [p for p in all_filtered_papers if p.get('score', 0) >= 8]
    if not recommendations: return "今日无高分精选论文。"

    report = "📊 **今日具身智能论文初筛建议**\n"
    report += "请复制 ID 到 GitHub 手动触发解析：\n\n"
    actions_url = f"https://github.com/{repo_owner}/{repo_name}/actions"
    report += f"👉 [点击去手动触发解析]({actions_url})\n\n"
    for p in recommendations:
        report += f"- `ID: {p['id']}` | 分数: {p['score']} | {p.get('title_zh', '无标题')}\n"
    return report

def deep_dive_only(papers_to_process):
    """直接执行深度全文解析，跳过打分环节"""
    final_reports = []
    
    for idx, item in enumerate(papers_to_process, 1):
        paper_id = item['id']
        print(f"正在进行全文深度解析: {paper_id}...")
        full_text = get_arxiv_full_text(paper_id)

        if not full_text:
            print(f"HTML 全文暂未生成，尝试抓取摘要页...")
            res_abs = requests.get(f"https://arxiv.org/abs/{paper_id}", timeout=10)
            if res_abs.status_code == 200:
                abs_soup = BeautifulSoup(res_abs.text, 'html.parser')
                full_text = abs_soup.find('blockquote', class_='abstract').text
            else:
                full_text = None
        
        expert_prompt = f"""
        Role: 你是一位具身智能领域研究员。请用平实、地道的中文对论文进行高信息密度的总结。
        Task: 像在组会上给同事分享一样，直接讲清楚论文做了什么、改了哪里、效果如何。严禁过度修饰，严禁使用炫技式的词汇。

        请严格按以下结构输出（使用 Markdown）：

        **0. 论文标题**
        - **英文标题**: [在此填入论文原文标题]
        - **中文标题**: [在此填入精准的中文翻译]
        - **研究机构**: [在此填入作者所属的主要单位，如：DeepMind, Stanford University等]

        **1. 整体逻辑**
        - **研究任务**: [论文研究的任务是什么，如：根据文本生成图像]
        - **研究动机**: [例如发现了什么问题需要改进，比如VLA生成动作的速度太慢]
        - **本质改动**: [本质改动，如：用视频生成代替扩散策略做轨迹预测]
        - **技术溯源**: [基于 CLIP/OpenVLA/Llama3 等哪些开源基座？]

        **2. 技术拆解**
        - **重点改进**: [本质改动对应的模型或算法的改动]
        - **架构细节**: [输入输出、具体的模型结构、模型规模等]
        - **核心 Loss**: [主 Loss 构成，是否有辅助任务（如视频重建）？]

        **3. 实验结果**
        - **数据集和baseline**: [实验用的数据集和对比的方法]
        - **评价指标**: [实验用的评价指标，如何评价]
        - **实验结果**: [比baseline好多少]

        待处理全文内容：
        {full_text if full_text else "（全文抓取失败，请基于摘要分析核心逻辑）"}
        """
        
        try:
            # 深度解析建议用逻辑更强的模型（如 qwen-plus）
            completion = client_llm.chat.completions.create(
                model="qwen3.7-plus",  # qwen3.6-plus, qwen3.5-flash
                messages=[{"role": "user", "content": expert_prompt}]
            )
            report = completion.choices[0].message.content

            # --- 提取逻辑 ---
            import re
            title_en = re.search(r"英文标题\*\*: (.*)", report)
            title_zh = re.search(r"中文标题\*\*: (.*)", report)
            affiliation = re.search(r"研究机构\*\*: (.*)", report)
            t_en = title_en.group(1).strip() if title_en else f"Arxiv: {paper_id}"
            t_zh = title_zh.group(1).strip() if title_zh else ""
            aff = affiliation.group(1).strip() if affiliation else "未知机构"
            
            # --- 渲染 Markdown ---
            # 在标题下方增加一行显示机构信息
            md = f"### {idx}. 🔥 [{t_en}] ({t_zh})\n"
            md += f"- **研究机构**: `{aff}`\n" # 新增展示行
            md += f"- **Arxiv ID**: `{paper_id}` | [点击跳转](https://arxiv.org/abs/{paper_id})\n\n"
            md += f"{report}\n\n"
            md += "---\n"
            final_reports.append(md)
        except Exception as e:
            print(f"深度解析出错 {paper_id}: {e}")
            
    return "\n\n".join(final_reports)

def generate_archive_and_index(date_info, arxiv_content):
    """生成详情页并更新索引，仅统计 VLA 内容"""
    
    # 只统计 Arxiv/VLA 数量
    vla_count = (arxiv_content or "").count("###")
    
    # 构造详情页标题，移除了 HF 统计
    display_title = f"{date_info['prefix']} (VLA: {vla_count} of {date_info['total']} entries)"
    
    safe_date_filename = re.sub(r'[^\w\s-]', '', date_info['prefix']).replace(' ', '_')
    os.makedirs('archive', exist_ok=True)
    daily_file_path = f"archive/{safe_date_filename}.html"

    paper_ids = re.findall(r'abs/(\d+\.\d+)', arxiv_content)
    sources_text = "\n".join([f"https://arxiv.org/html/{pid}" for pid in paper_ids])

    def get_html_template(title, body_content, is_index_page=False, sources_block=""):
        back_link = "<a href='../index.html' style='margin-bottom:20px; display:block;'>← 返回主索引</a>" if not is_index_page else ""
        safe_body = body_content.replace('</script>', '<\\/script>')
        return f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
            <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
            <script>
                window.MathJax = {{
                    tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']] }},
                    options: {{ skipHtmlTags: ['script', 'style', 'textarea'] }}
                }};
            </script>
            <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
            <style>
                .markdown-body {{ box-sizing: border-box; min-width: 200px; max-width: 980px; margin: 0 auto; padding: 45px; }}
                @media (max-width: 767px) {{ .markdown-body {{ padding: 15px; }} }}
            </style>
        </head>
        <body class="markdown-body">
            {back_link}
            <h1>{title}</h1>
            <div id="content"></div>
            {sources_block}
            <script type="text/markdown" id="raw-markdown">{safe_body}</script>
            <script>
                const rawMdElement = document.getElementById('raw-markdown');
                if (rawMdElement) {{
                    document.getElementById('content').innerHTML = marked.parse(rawMdElement.textContent);
                    // 渲染后触发公式排版
                    if (window.MathJax && window.MathJax.typesetPromise) {{
                        window.MathJax.typesetPromise();
                    }}
                }}
                function copySources() {{ /* 原代码保持不变 */ }}
            </script>
        </body>
        </html>
        """

    # sources_html = ""
    # if sources_text:
    #     sources_html = f"""
    #     <div class="sources-box">
    #         <h3>🔗 NotebookLM Sources 集合区 (共 {len(paper_ids)} 篇)</h3>
    #         <textarea id="sources-text" readonly>{sources_text}</textarea>
    #         <button class="copy-btn" onclick="copySources()">📋 复制所有来源链接</button>
    #     </div>
    #     """

    # 仅保存 Arxiv 内容
    with open(daily_file_path, "w", encoding="utf-8") as f:
        f.write(get_html_template(f"🤖 具身大模型简报 - {display_title}", arxiv_content or "", False, ""))

    history_files = [f for f in os.listdir('archive') if f.endswith('.html')]
    indexed_history = []
    for f_name in history_files:
        try:
            with open(f"archive/{f_name}", "r", encoding="utf-8") as hf:
                file_soup = BeautifulSoup(hf.read(), 'html.parser')
                full_title = file_soup.title.string.replace("🤖 具身大模型简报 - ", "")
                
                # 1. 尝试匹配标准 Arxiv 日期
                date_match = re.search(r'([A-Za-z]{3}, \d{1,2} [A-Za-z]{3} \d{4})', full_title)
                
                if date_match:
                    date_obj = datetime.datetime.strptime(date_match.group(1), "%a, %d %b %Y")
                elif "Manual_Batch" in f_name:
                    # 2. 如果是手动模式，从文件名提取日期 (例如 Manual_Batch_0403 -> 04月03日)
                    date_str = f_name.replace("Manual_Batch_", "").replace(".html", "")
                    # 假设是当年，手动构造一个 datetime 对象用于排序
                    date_obj = datetime.datetime.strptime(date_str, "%m%d").replace(year=datetime.datetime.now().year)
                else:
                    # 3. 兜底方案：使用文件的最后修改时间
                    date_obj = datetime.datetime.fromtimestamp(os.path.getmtime(f"archive/{f_name}"))
                
                # 确保只要解析出 date_obj，就加入索引列表
                indexed_history.append((date_obj, full_title, f_name))
        except Exception as e:
            print(f"解析历史文件 {f_name} 出错: {e}")
            continue

    indexed_history.sort(key=lambda x: x[0], reverse=True)
    index_md = "### 📅 历史存档列表\n\n"
    for _, hist_title, f_name in indexed_history:
        index_md += f"- [{hist_title}](archive/{f_name})\n"
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(get_html_template("📚 具身大模型科研日报 - 历史索引", index_md, True))

    # 飞书推送卡片只显示 VLA 数量
    requests.post(FEISHU_WEBHOOK, json={
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": f"🌟 具身精选 | {display_title}"}, "template": "blue"},
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": f"今日共包含 **{vla_count}** 篇 VLA 筛选论文。"}},
                {"tag": "action", "actions": [{"tag": "button", "text": {"tag": "plain_text", "content": "🌐 查看网页 & 复制 Notebook 链接"}, "type": "primary", "url": GITHUB_PAGES_URL}]}
            ]
        }
    })

def send_feishu_notification(text):
    """发送纯文本或简单 Markdown 到飞书"""
    requests.post(FEISHU_WEBHOOK, json={
        "msg_type": "text",
        "content": {"text": text}
    })
    
if __name__ == "__main__":
    target_ids_str = os.getenv("TARGET_IDS", "")
    
    if target_ids_str:
        # --- 模式 A：执行手动解析模式 ---
        selected_ids = [i.strip() for i in target_ids_str.split(",") if i.strip()]
        papers_to_process = [{"id": pid} for pid in selected_ids]
        
        # 调用深度解析函数
        arxiv_content = deep_dive_only(papers_to_process)
        
        # 生成网页并推送（复用原函数）
        real_info, _, _ = scrape_arxiv(CATEGORIES[0]) 
        
        if real_info:
            date_info = real_info
        else:
            # 兜底：如果抓取失败，再使用当前的日期
            date_info = {
                "prefix": datetime.datetime.now().strftime('%a, %d %b %Y'), 
                "total": "0" 
            }
        # date_info = {
        #     "prefix": "Mon, 1 June 2026", 
        #     "total": "59" 
        # }
        generate_archive_and_index(date_info, arxiv_content)
    else:
        # --- 模式 B：定时任务执行初筛汇报 ---
        all_p = {}
        date_info = None
        for cat in CATEGORIES:
            info, _, ps = scrape_arxiv(cat)
            if info: date_info = info
            for p in ps: all_p[p['id']] = p
        
        # 仅初筛并汇报到飞书
        report_list = only_filter_and_report(list(all_p.values()))
        send_feishu_notification(report_list)
