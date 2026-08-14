# -*- coding: utf-8 -*-
"""
把课程的全部 Markdown 章节构建成单文件网页版手册。

用法(仓库根目录):  python3 docs/build_site.py
产物:
  docs/index.html   完整独立网页(可本地双击打开 / 用于 GitHub Pages)
依赖:  pip install markdown-it-py
"""
import html
import re
from datetime import date
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/zky001/hw910_llm_course"
BRANCH = "claude/huawei-910-training-guide-nefkef"
BLOB = f"{REPO_URL}/blob/{BRANCH}"

# (锚点id, 目录, 文件, 编号显示, 眉标, 侧栏标题)
PARTS = [
    ("开始", [
        ("ch00", "00-overview", "README.md", "00", "第 0 章", "硬件与生态"),
        ("ch01", "01-environment", "README.md", "01", "第 1 章", "环境搭建"),
    ]),
    ("跑通", [
        ("ch02", "02-single-card", "README.md", "02", "第 2 章", "单卡上手"),
        ("ch03", "03-multi-card-ddp", "README.md", "03", "第 3 章", "八卡 DDP"),
    ]),
    ("实战", [
        ("ch04", "04-lora-finetune", "README.md", "04", "第 4 章", "LoRA 微调"),
        ("ch05", "05-full-finetune-deepspeed", "README.md", "05", "第 5 章", "全参训练"),
        ("ch06", "06-pretrain-mindspeed", "README.md", "06", "第 6 章", "预训练"),
    ]),
    ("性能", [
        ("ch07", "07-profiling", "README.md", "07", "第 7 章", "性能剖析"),
        ("ch08", "08-optimization", "README.md", "08", "第 8 章", "性能优化"),
    ]),
    ("稳定与上线", [
        ("ch09", "09-troubleshooting", "README.md", "09", "第 9 章", "故障排查"),
        ("ch10", "10-inference", "README.md", "10", "第 10 章", "推理部署"),
    ]),
    ("附录", [
        ("appendix-a", "appendix", "version-matrix.md", "A", "附录 A", "版本配套"),
        ("appendix-b", "appendix", "cheatsheet.md", "B", "附录 B", "速查表"),
        ("appendix-c", "appendix", "cuda-to-npu.md", "C", "附录 C", "CUDA→NPU"),
    ]),
]

md = MarkdownIt("commonmark", {"html": False, "typographer": False}).enable("table")


# ---------------------------------------------------------------- 链接改写
def rewrite_links(text: str, chapter_dir: str) -> str:
    """把 markdown 里的仓库内相对链接改写为页面锚点 / GitHub blob 链接。"""
    # 章节 README -> 锚点
    text = re.sub(r"\((?:\.\./)?(\d{2})-[a-z0-9-]+/README\.md\)", r"(#ch\1)", text)
    # 附录 -> 锚点
    for name, anchor in [("version-matrix", "appendix-a"),
                         ("cheatsheet", "appendix-b"),
                         ("cuda-to-npu", "appendix-c")]:
        text = re.sub(rf"\((?:\.\./)?appendix/{name}\.md\)", f"(#{anchor})", text)
    # 根 README -> 总览
    text = text.replace("(../README.md)", "(#intro)")
    # 带目录前缀的脚本/配置 -> GitHub blob
    text = re.sub(r"\(((?:\d{2}-[a-z0-9-]+|appendix)/[A-Za-z0-9_./-]+\.(?:py|sh|json|yaml|md))\)",
                  rf"({BLOB}/\1)", text)
    # 同目录的脚本/配置 -> GitHub blob
    text = re.sub(r"\(([A-Za-z0-9_.-]+\.(?:py|sh|json|yaml))\)",
                  rf"({BLOB}/{chapter_dir}/\1)", text)
    return text


# ---------------------------------------------------------------- Markdown 转换
def convert(body_md: str, section_id: str):
    """markdown -> html；给 h2/h3 加锚点 id，抽取 h2 列表供侧栏使用。"""
    out = md.render(body_md)

    # mermaid 代码块 -> 原生 <pre class="mermaid">
    out = re.sub(r'<pre><code class="language-mermaid">(.*?)</code></pre>',
                 r'<pre class="mermaid">\1</pre>', out, flags=re.S)

    # 表格包滚动容器
    out = out.replace("<table>", '<div class="table-wrap"><table>')
    out = out.replace("</table>", "</table></div>")

    # 引用块 -> 提示样式
    def quote_cls(m):
        head = m.group(1)
        cls = "callout"
        if "✅" in head:
            cls += " q-ok"
        elif "⚠" in head:
            cls += " q-warn"
        elif "💡" in head:
            cls += " q-tip"
        return f'<blockquote class="{cls}"><p>{head}'
    out = re.sub(r"<blockquote>\s*<p>(.{0,80})", quote_cls, out, count=0, flags=re.S)

    # h2/h3 锚点
    subs, counter = [], {"n": 0}

    def add_id(m):
        tag, inner = m.group(1), m.group(2)
        counter["n"] += 1
        hid = f"{section_id}-s{counter['n']}"
        if tag == "h2":
            plain = re.sub(r"<[^>]+>", "", inner)
            subs.append((hid, plain))
        return f'<{tag} id="{hid}">{inner}<a class="hlink" href="#{hid}" aria-label="锚点">#</a></{tag}>'

    out = re.sub(r"<(h[23])>(.*?)</h[23]>", add_id, out, flags=re.S)
    return out, subs


def load_chapter(cid, cdir, fname):
    raw = (ROOT / cdir / fname).read_text(encoding="utf-8")
    raw = rewrite_links(raw, cdir)
    lines = raw.split("\n")
    h1 = lines[0].lstrip("# ").strip()
    title = h1.split("·", 1)[1].strip() if "·" in h1 else h1
    body = "\n".join(lines[1:]).strip()
    html_body, subs = convert(body, cid)
    return title, html_body, subs


# ---------------------------------------------------------------- 组装
def build():
    sections, nav_groups = [], []
    for part, chapters in PARTS:
        items = []
        for cid, cdir, fname, num, eyebrow, short in chapters:
            title, body, subs = load_chapter(cid, cdir, fname)
            sections.append(f"""
<section class="chapter" id="{cid}">
  <header class="ch-head">
    <div class="ch-num" aria-hidden="true">{num}</div>
    <div class="ch-title">
      <div class="eyebrow">{eyebrow}</div>
      <h1>{html.escape(title)}</h1>
    </div>
  </header>
{body}
</section>""")
            sub_html = "".join(
                f'<li><a href="#{hid}">{html.escape(txt)}</a></li>' for hid, txt in subs)
            items.append(
                f'<li class="nav-item" data-target="{cid}">'
                f'<a class="nav-link" href="#{cid}"><span class="nav-num">{num}</span>'
                f'<span>{html.escape(short)}</span></a>'
                f'<ul class="subs">{sub_html}</ul></li>')
        nav_groups.append(
            f'<div class="nav-group"><div class="nav-part">{part}</div>'
            f'<ul>{"".join(items)}</ul></div>')

    # 总览：根 README 去掉 h1 与开头的口号引用（口号进 hero）
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme = rewrite_links(readme, ".")
    readme_lines = readme.split("\n")[1:]
    while readme_lines and (not readme_lines[0].strip() or readme_lines[0].startswith(">")):
        readme_lines.pop(0)
    intro_html, _ = convert("\n".join(readme_lines).strip(), "intro")

    css = CSS
    js = JS
    today = date.today().isoformat()

    head_meta = ('<title>昇腾 910 训练手册</title>\n'
                 '<meta name="viewport" content="width=device-width, initial-scale=1">')
    page_body = f"""<style>{css}</style>
<button class="menu-btn" id="menuBtn" aria-label="打开目录">☰ 目录</button>
<div class="backdrop" id="backdrop"></div>
<div class="layout">
<aside class="sidebar" id="sidebar">
  <a class="brand" href="#intro">
    <span class="brand-mark">910</span>
    <span class="brand-name">昇腾 910 训练手册<small>8 卡大模型训练实战</small></span>
  </a>
  <nav class="nav">
    <div class="nav-group"><ul>
      <li class="nav-item" data-target="intro">
        <a class="nav-link" href="#intro"><span class="nav-num">◎</span><span>课程总览</span></a>
        <ul class="subs"></ul>
      </li>
    </ul></div>
    {"".join(nav_groups)}
  </nav>
  <div class="side-foot">
    <a href="{REPO_URL}" rel="noopener">GitHub 仓库 ↗</a>
    <button id="themeBtn" aria-label="切换深浅色">◐ 主题</button>
  </div>
</aside>
<main class="main">
  <section class="chapter" id="intro">
    <div class="hero">
      <div class="eyebrow">HW910 · LLM TRAINING COURSE</div>
      <h1 class="hero-title">昇腾 910 · 八卡大模型<br>训练实战手册</h1>
      <p class="hero-sub">用一台 8 卡昇腾 910 服务器，从零把大模型的微调、全参训练、预训练跑起来，并把性能调到位。</p>
      <div class="chips">
        <span class="chip">10 章 + 3 附录</span>
        <span class="chip">8 × Ascend 910</span>
        <span class="chip">PyTorch / torch_npu</span>
        <a class="chip chip-link" href="{REPO_URL}" rel="noopener">配套代码 ↗</a>
      </div>
    </div>
{intro_html}
  </section>
  {"".join(sections)}
  <footer class="foot">
    <p>本手册为社区教程，与华为官方无关；涉及商标归各自所有者。内容基线 2026-08，命令以官方文档为准。</p>
    <p><a href="{REPO_URL}" rel="noopener">GitHub: zky001/hw910_llm_course</a> · 构建于 {today}</p>
  </footer>
</main>
</div>
<script>{js}</script>
<script>
window.addEventListener('load', function () {{
  if (window.mermaid) {{
    var t = document.documentElement.getAttribute('data-theme');
    var dark = t === 'dark' || (!t && matchMedia('(prefers-color-scheme: dark)').matches);
    window.mermaid.initialize({{ startOnLoad: false, theme: dark ? 'dark' : 'neutral' }});
    window.mermaid.run();
  }}
}});
</script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
"""

    standalone = ("<!doctype html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n"
                  + head_meta + "\n"
                  "<link rel=\"icon\" href=\"data:image/svg+xml,"
                  "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E"
                  "%3Crect width='100' height='100' rx='18' fill='%23c7000b'/%3E"
                  "%3Ctext x='50' y='68' font-size='44' font-family='monospace' font-weight='bold' "
                  "fill='white' text-anchor='middle'%3E910%3C/text%3E%3C/svg%3E\">\n"
                  "</head>\n<body>\n" + page_body + "\n</body>\n</html>\n")
    (ROOT / "docs" / "index.html").write_text(standalone, encoding="utf-8")
    print(f"docs/index.html          {len(standalone)/1024:.0f} KB (独立网页)")
    return head_meta + "\n" + page_body


CSS = r"""
:root{
  --bg:#f7f6f5; --surface:#ffffff; --ink:#201d1c; --muted:#6b6461;
  --accent:#c7000b; --accent-ink:#ffffff; --accent-soft:rgba(199,0,11,.07);
  --line:#e6e2df; --chip:#efecea; --code-bg:#211e1d; --code-ink:#e8e2dc;
  --callout:#faf4f0; --ok:#1d7a4f; --warn:#a05a00; --tip:#215f9a;
  --shadow:0 1px 3px rgba(32,29,28,.06);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#161413; --surface:#1e1b1a; --ink:#e9e4df; --muted:#a39a94;
    --accent:#ff6257; --accent-ink:#1a0705; --accent-soft:rgba(255,98,87,.10);
    --line:#2e2a28; --chip:#2a2625; --code-bg:#121010; --code-ink:#ddd6cf;
    --callout:#232019; --ok:#5cc491; --warn:#e0a458; --tip:#7ab3e8;
    --shadow:0 1px 3px rgba(0,0,0,.4);
  }
}
:root[data-theme="dark"]{
  --bg:#161413; --surface:#1e1b1a; --ink:#e9e4df; --muted:#a39a94;
  --accent:#ff6257; --accent-ink:#1a0705; --accent-soft:rgba(255,98,87,.10);
  --line:#2e2a28; --chip:#2a2625; --code-bg:#121010; --code-ink:#ddd6cf;
  --callout:#232019; --ok:#5cc491; --warn:#e0a458; --tip:#7ab3e8;
  --shadow:0 1px 3px rgba(0,0,0,.4);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth; scroll-padding-top:24px}
@media (prefers-reduced-motion: reduce){ html{scroll-behavior:auto} *{transition:none!important} }
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:-apple-system,"PingFang SC","Hiragino Sans GB","Source Han Sans SC","Noto Sans CJK SC","Microsoft YaHei","Segoe UI",sans-serif;
  font-size:15.5px; line-height:1.85;
  -webkit-font-smoothing:antialiased;
}
a{color:var(--accent); text-decoration:none}
a:hover{text-decoration:underline}
a:focus-visible,button:focus-visible{outline:2px solid var(--accent); outline-offset:2px; border-radius:3px}
code,pre,.mono,.nav-num,.ch-num,.eyebrow{
  font-family:"SF Mono","JetBrains Mono","Cascadia Code",Menlo,Consolas,"Liberation Mono",monospace;
}

/* ---------- 布局 ---------- */
.layout{display:flex; min-height:100vh}
.sidebar{
  width:264px; flex:0 0 264px; background:var(--surface);
  border-right:1px solid var(--line);
  position:sticky; top:0; height:100vh; overflow-y:auto;
  display:flex; flex-direction:column; padding:20px 14px 14px;
}
.main{flex:1; min-width:0; display:flex; flex-direction:column; align-items:center; padding:0 28px}
.chapter{width:100%; max-width:800px; padding:44px 0 24px; border-bottom:1px solid var(--line)}
.chapter:last-of-type{border-bottom:none}
.foot{width:100%; max-width:800px; padding:28px 0 48px; color:var(--muted); font-size:.82rem; border-top:1px solid var(--line)}

/* ---------- 侧栏 ---------- */
.brand{display:flex; gap:10px; align-items:center; color:var(--ink); padding:2px 8px 16px}
.brand:hover{text-decoration:none}
.brand-mark{
  background:var(--accent); color:var(--accent-ink); font-weight:700;
  font-family:Menlo,Consolas,monospace; font-size:.8rem;
  border-radius:8px; padding:8px 7px; letter-spacing:.03em;
}
.brand-name{font-weight:700; font-size:.95rem; line-height:1.3}
.brand-name small{display:block; font-weight:400; color:var(--muted); font-size:.72rem; letter-spacing:.05em}
.nav{flex:1}
.nav ul{list-style:none; margin:0; padding:0}
.nav-part{
  font-size:.68rem; color:var(--muted); letter-spacing:.18em; margin:16px 8px 4px;
}
.nav-link{
  display:flex; gap:9px; align-items:baseline; color:var(--ink);
  padding:5px 8px; border-radius:6px; font-size:.88rem;
}
.nav-link:hover{background:var(--chip); text-decoration:none}
.nav-num{color:var(--muted); font-size:.72rem; min-width:1.6em}
.nav-item.active > .nav-link{background:var(--accent-soft); color:var(--accent); font-weight:600}
.nav-item.active > .nav-link .nav-num{color:var(--accent)}
.subs{display:none; margin:2px 0 6px!important; border-left:1px solid var(--line); margin-left:17px!important}
.nav-item.active .subs{display:block}
.subs a{display:block; color:var(--muted); font-size:.78rem; padding:2.5px 10px; line-height:1.45}
.subs a:hover{color:var(--accent); text-decoration:none}
.side-foot{
  border-top:1px solid var(--line); margin-top:12px; padding:12px 8px 0;
  display:flex; justify-content:space-between; align-items:center; font-size:.8rem;
}
.side-foot button{
  background:none; border:1px solid var(--line); color:var(--muted);
  border-radius:6px; padding:3px 9px; cursor:pointer; font-size:.78rem;
}
.side-foot button:hover{color:var(--ink); border-color:var(--muted)}

/* ---------- hero ---------- */
.hero{padding:18px 0 6px}
.eyebrow{
  color:var(--accent); font-size:.7rem; letter-spacing:.16em; font-weight:600;
}
.hero-title{
  font-size:2.15rem; line-height:1.32; margin:.5em 0 .4em; letter-spacing:.01em;
  text-wrap:balance;
}
.hero-sub{color:var(--muted); max-width:36em; margin:0 0 1.2em}
.chips{display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px}
.chip{
  background:var(--chip); border-radius:999px; padding:3px 13px;
  font-size:.78rem; color:var(--muted);
}
.chip-link{color:var(--accent); background:var(--accent-soft)}
.chip-link:hover{text-decoration:none; filter:brightness(.94)}

/* ---------- 章头与标题 ---------- */
.ch-head{display:flex; gap:18px; align-items:flex-end; margin-bottom:10px}
.ch-num{
  font-size:3rem; font-weight:700; color:var(--accent); opacity:.28;
  line-height:.9; letter-spacing:-.02em; user-select:none;
}
.ch-title h1{font-size:1.72rem; margin:.1em 0 0; line-height:1.35; text-wrap:balance}
h2{
  font-size:1.32rem; margin:2.2em 0 .7em; padding-bottom:.35em;
  border-bottom:1px solid var(--line); line-height:1.4;
}
h3{font-size:1.06rem; margin:1.8em 0 .5em}
.hlink{margin-left:.45em; font-size:.8em; opacity:0; color:var(--muted)}
h2:hover .hlink,h3:hover .hlink{opacity:.8}
p{margin:.75em 0}
strong{font-weight:650}
ul,ol{padding-left:1.6em; margin:.6em 0}
li{margin:.28em 0}
hr{border:none; border-top:1px solid var(--line); margin:2em 0}

/* ---------- 代码 ---------- */
code{
  background:var(--chip); border-radius:4px; padding:.12em .38em;
  font-size:.84em;
}
pre{
  background:var(--code-bg); color:var(--code-ink);
  border-radius:10px; padding:14px 16px; overflow-x:auto;
  line-height:1.62; font-size:.82rem; position:relative;
  box-shadow:var(--shadow); margin:.9em 0;
}
pre code{background:none; padding:0; font-size:1em; color:inherit}
.copy-btn{
  position:absolute; top:8px; right:8px; border:none; cursor:pointer;
  background:rgba(255,255,255,.09); color:#cfc8c1; border-radius:6px;
  font-size:.7rem; padding:3px 9px; opacity:0; transition:opacity .15s;
}
pre:hover .copy-btn{opacity:1}
.copy-btn:hover{background:rgba(255,255,255,.18)}
pre.mermaid{
  background:var(--surface); color:var(--ink); border:1px solid var(--line);
  display:flex; justify-content:center; box-shadow:none;
}

/* ---------- 表格 ---------- */
.table-wrap{overflow-x:auto; margin:.9em 0; border:1px solid var(--line); border-radius:10px}
table{border-collapse:collapse; width:100%; font-size:.86rem; line-height:1.6}
th,td{padding:8px 13px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top}
th{background:var(--chip); font-weight:650; white-space:nowrap}
tr:last-child td{border-bottom:none}
td code,th code{white-space:nowrap}
tbody tr:hover{background:var(--accent-soft)}

/* ---------- 引用/提示 ---------- */
blockquote{
  margin:1em 0; padding:10px 16px; background:var(--callout);
  border:1px solid var(--line); border-radius:10px; color:var(--ink);
}
blockquote p{margin:.4em 0}
blockquote.q-ok{border-color:color-mix(in srgb, var(--ok) 35%, var(--line))}
blockquote.q-warn{border-color:color-mix(in srgb, var(--warn) 40%, var(--line))}
blockquote.q-tip{border-color:color-mix(in srgb, var(--tip) 35%, var(--line))}

/* ---------- 移动端 ---------- */
.menu-btn{
  display:none; position:fixed; top:12px; left:12px; z-index:60;
  background:var(--surface); color:var(--ink); border:1px solid var(--line);
  border-radius:8px; padding:7px 13px; font-size:.85rem; cursor:pointer;
  box-shadow:var(--shadow);
}
.backdrop{display:none; position:fixed; inset:0; background:rgba(0,0,0,.4); z-index:49}
@media (max-width: 980px){
  .menu-btn{display:block}
  .sidebar{
    position:fixed; left:0; top:0; z-index:50; transform:translateX(-100%);
    transition:transform .2s ease; box-shadow:0 0 30px rgba(0,0,0,.25);
  }
  .sidebar.open{transform:none}
  .backdrop.show{display:block}
  .main{padding:0 18px}
  .chapter{padding-top:58px}
  .hero-title{font-size:1.7rem}
  .ch-num{font-size:2.2rem}
}
"""

JS = r"""
(function(){
  var doc = document;
  // ---- 主题切换 ----
  var themeBtn = doc.getElementById('themeBtn');
  try {
    var saved = localStorage.getItem('hw910-theme');
    if (saved) doc.documentElement.setAttribute('data-theme', saved);
  } catch(e){}
  if (themeBtn) themeBtn.addEventListener('click', function(){
    var cur = doc.documentElement.getAttribute('data-theme');
    var next = cur === 'dark' ? 'light' : (cur === 'light' ? 'dark'
      : (matchMedia('(prefers-color-scheme: dark)').matches ? 'light' : 'dark'));
    doc.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('hw910-theme', next); } catch(e){}
  });

  // ---- 移动端目录 ----
  var sidebar = doc.getElementById('sidebar');
  var btn = doc.getElementById('menuBtn');
  var backdrop = doc.getElementById('backdrop');
  function closeNav(){ sidebar.classList.remove('open'); backdrop.classList.remove('show'); }
  btn.addEventListener('click', function(){
    sidebar.classList.toggle('open'); backdrop.classList.toggle('show');
  });
  backdrop.addEventListener('click', closeNav);
  doc.addEventListener('keydown', function(e){ if (e.key === 'Escape') closeNav(); });
  sidebar.addEventListener('click', function(e){
    if (e.target.closest('a') && matchMedia('(max-width: 980px)').matches) closeNav();
  });

  // ---- 代码复制按钮 ----
  doc.querySelectorAll('pre').forEach(function(pre){
    if (pre.classList.contains('mermaid')) return;
    var b = doc.createElement('button');
    b.className = 'copy-btn'; b.type = 'button'; b.textContent = '复制';
    b.addEventListener('click', function(){
      var text = pre.querySelector('code') ? pre.querySelector('code').innerText : pre.innerText;
      function done(){ b.textContent = '已复制'; setTimeout(function(){ b.textContent = '复制'; }, 1600); }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function(){ fallback(); });
      } else { fallback(); }
      function fallback(){
        var ta = doc.createElement('textarea'); ta.value = text;
        doc.body.appendChild(ta); ta.select();
        try { doc.execCommand('copy'); done(); } catch(e){}
        doc.body.removeChild(ta);
      }
    });
    pre.appendChild(b);
  });

  // ---- 滚动高亮当前章节 ----
  var sections = Array.prototype.slice.call(doc.querySelectorAll('section.chapter'));
  var items = {};
  doc.querySelectorAll('.nav-item').forEach(function(it){ items[it.dataset.target] = it; });
  var ticking = false;
  function spy(){
    ticking = false;
    var y = window.scrollY + 140, current = sections[0];
    for (var i = 0; i < sections.length; i++) {
      if (sections[i].offsetTop <= y) current = sections[i]; else break;
    }
    doc.querySelectorAll('.nav-item.active').forEach(function(n){ n.classList.remove('active'); });
    var it = items[current.id];
    if (it) {
      it.classList.add('active');
      // 只滚动侧栏自身, 绝不触碰窗口滚动位置
      var top = it.offsetTop, bot = top + it.offsetHeight;
      if (top < sidebar.scrollTop + 70 || bot > sidebar.scrollTop + sidebar.clientHeight - 50) {
        sidebar.scrollTop = Math.max(0, top - sidebar.clientHeight * 0.35);
      }
    }
  }
  window.addEventListener('scroll', function(){
    if (!ticking) { ticking = true; requestAnimationFrame(spy); }
  }, { passive: true });
  spy();
})();
"""

if __name__ == "__main__":
    import sys
    content = build()
    # 可选: 传一个输出路径, 生成"无 html/head/body 包裹"的内容版(用于托管平台注入)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(content, encoding="utf-8")
        print(f"{sys.argv[1]}  ({len(content)/1024:.0f} KB, 内容版)")
