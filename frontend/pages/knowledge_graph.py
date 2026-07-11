import os
import sqlite3

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Knowledge Graph", layout="wide")
st.title("Семантический Граф Знаний R&D")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    search = st.text_input("Фильтр по материалу или процессу", placeholder="никель, шлак, флотация…")
with col2:
    node_limit = st.slider("Количество узлов", min_value=10, max_value=100, value=40, step=10)
with col3:
    show_docs = st.checkbox("Показывать документы", value=True)


@st.cache_data(ttl=60)
def load_facts(limit: int, search_term: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    db_path = os.path.join(project_root, "knowledge_base.db")

    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if search_term:
        term = f"%{search_term.lower()}%"
        cursor.execute(
            """
            SELECT material, process, source_file, outcome FROM facts
            WHERE LOWER(material) LIKE ? OR LOWER(process) LIKE ? OR LOWER(outcome) LIKE ?
            LIMIT ?
        """,
            (term, term, term, limit),
        )
    else:
        cursor.execute("SELECT material, process, source_file, outcome FROM facts LIMIT ?", (limit,))

    rows = cursor.fetchall()
    conn.close()
    return rows


rows = load_facts(node_limit, search)

if not rows:
    st.warning("Ничего не найдено. Попробуйте изменить фильтр или увеличить количество узлов.")
    st.stop()

nodes_data = {}
edges_data = []
node_id = 0


def get_or_create(label: str, group: str, title: str = "") -> int:
    global node_id
    key = (label, group)
    if key not in nodes_data:
        nodes_data[key] = {"id": node_id, "label": label, "group": group, "title": title}
        node_id += 1
    return nodes_data[key]["id"]


for material, process, source, outcome in rows:
    if not material or not process:
        continue

    mat_label = (material[:25] + "…") if len(material) > 25 else material
    proc_label = (process[:25] + "…") if len(process) > 25 else process
    src_label = (source[:20] + "…") if len(source) > 20 else source
    outcome_title = (outcome or "")[:200]

    mat_id = get_or_create(mat_label, "material", f"Материал: {material}")
    proc_id = get_or_create(proc_label, "process", f"Процесс: {process}\n\nРезультат: {outcome_title}")
    edges_data.append({"from": mat_id, "to": proc_id})

    if show_docs:
        src_id = get_or_create(src_label, "document", f"Источник: {source}")
        edges_data.append({"from": proc_id, "to": src_id})

# Сериализация в JS-массивы
nodes_js = []
for (label, group), info in nodes_data.items():
    color_map = {
        "material": {
            "background": "#1f77b4",
            "border": "#aec7e8",
            "highlight": {"background": "#6baed6", "border": "#ffffff"},
        },
        "process": {
            "background": "#e07b39",
            "border": "#fdae6b",
            "highlight": {"background": "#fd8d3c", "border": "#ffffff"},
        },
        "document": {
            "background": "#2ca02c",
            "border": "#98df8a",
            "highlight": {"background": "#41ab5d", "border": "#ffffff"},
        },
    }
    color = color_map.get(group, {"background": "#999", "border": "#ccc"})
    shape = {"material": "dot", "process": "diamond", "document": "square"}.get(group, "dot")
    size = {"material": 20, "process": 18, "document": 14}.get(group, 16)

    color_str = str(color).replace("'", '"')
    node_label = repr(info["label"])
    node_title = repr(info["title"])
    node_id_val = info["id"]
    nodes_js.append(
        f"{{id:{node_id_val},label:{node_label},title:{node_title}," f'color:{color_str},shape:"{shape}",size:{size}}}'
    )

edges_js = [f'{{from:{e["from"]},to:{e["to"]},color:{{color:"#555",highlight:"#aaa"}}}}' for e in edges_data]

html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet">
<style>
  body {{ margin:0; background:#0e1117; }}
  #graph {{ width:100%; height:600px; background:#0e1117; border:1px solid #2d3748; border-radius:8px; }}
  #info {{
    position:absolute; top:10px; right:10px;
    background:rgba(22,27,36,0.95); color:#e2e8f0;
    padding:12px 16px; border-radius:8px; border:1px solid #2d3748;
    font-family:sans-serif; font-size:13px; max-width:280px;
    display:none; line-height:1.6;
  }}
</style>
</head>
<body>
<div style="position:relative">
  <div id="graph"></div>
  <div id="info"></div>
</div>
<script>
var nodes = new vis.DataSet([{",".join(nodes_js)}]);
var edges = new vis.DataSet([{",".join(edges_js)}]);

var container = document.getElementById("graph");
var data = {{nodes: nodes, edges: edges}};
var options = {{
  nodes: {{
    font: {{color:"#e2e8f0", size:13, face:"Inter, sans-serif"}},
    borderWidth: 2,
    shadow: {{enabled:true, color:"rgba(0,0,0,0.5)", x:2, y:2, size:6}}
  }},
  edges: {{
    width: 1.5,
    smooth: {{type:"continuous"}},
    arrows: {{to: {{enabled:true, scaleFactor:0.5}}}}
  }},
  physics: {{
    enabled: true,
    forceAtlas2Based: {{
      gravitationalConstant: -60,
      centralGravity: 0.005,
      springLength: 120,
      springConstant: 0.08,
      damping: 0.4
    }},
    solver: "forceAtlas2Based",
    stabilization: {{iterations: 150}}
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 100,
    zoomView: true,
    dragView: true
  }},
  layout: {{improvedLayout: true}}
}};

var network = new vis.Network(container, data, options);

network.on("click", function(params) {{
  var info = document.getElementById("info");
  if (params.nodes.length > 0) {{
    var nodeId = params.nodes[0];
    var node = nodes.get(nodeId);
    info.style.display = "block";
    info.innerHTML = "<strong>" + node.label + "</strong><br><br>" + (node.title || "").replace(/\\n/g, "<br>");
  }} else {{
    info.style.display = "none";
  }}
}});
</script>
</body>
</html>
"""

components.html(html, height=620, scrolling=False)

st.markdown(
    """
**Легенда:**
 **Синие круги** — Материалы и вещества &nbsp;|&nbsp;
 **Оранжевые ромбы** — Технологические процессы &nbsp;|&nbsp;
 **Зелёные квадраты** — Документы-источники

*Кликните на узел чтобы увидеть подробности. Колесико мыши — зум.*
"""
)

materials = sum(1 for (_, g), _ in nodes_data.items() if g == "material")
processes = sum(1 for (_, g), _ in nodes_data.items() if g == "process")
docs = sum(1 for (_, g), _ in nodes_data.items() if g == "document")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Материалов", materials)
c2.metric("Процессов", processes)
c3.metric("Документов", docs)
c4.metric("Связей", len(edges_data))
