# 分析方法详解 🔬

本文档详细说明 Lenny's Podcast 知识图谱的技术实现方法。

## 目录

- [数据来源](#数据来源)
- [分析流程](#分析流程)
- [核心算法](#核心算法)
- [可视化实现](#可视化实现)
- [如何复现](#如何复现)
- [局限性](#局限性)

---

## 数据来源

### 原始数据

| 指标 | 数值 |
|------|------|
| 播客期数 | 297 期 |
| 总字数 | 2300万+ 字 |
| 嘉宾数量 | 297 位 |
| 平均每期 | ~77,000 字 |

### 数据格式

每期播客的完整文字稿，包含：
- 嘉宾发言
- 主持人提问
- 对话全文

---

## 分析流程

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      数据处理流水线                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ 文本清洗  │ -> │ 概念提取  │ -> │ 向量化   │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                       │                     │
│                                       ▼                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ 可视化   │ <- │ 网络构建  │ <- │ 聚类分析  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Step 1: 文本预处理

```python
# 清洗文本
def preprocess(text):
    text = text.lower()
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text
```

处理内容：
- 统一转小写
- 移除停用词（the, a, is 等）
- 词形还原（running → run）

### Step 2: 概念提取

定义 20 个核心概念维度：

```python
concepts = [
    'product', 'growth', 'leadership', 'strategy', 
    'customer', 'innovation', 'career', 'decision',
    'execution', 'founder', 'culture', 'team', 
    'data', 'design', 'marketing', 'engineering',
    'startup', 'scale', 'hiring', 'feedback'
]
```

对每位嘉宾计算概念频率：

```python
def extract_concepts(transcript, concepts):
    scores = {}
    for concept in concepts:
        count = transcript.count(concept)
        scores[concept] = min(count / 50, 15)  # 归一化
    return scores
```

### Step 3: 向量化

将每位嘉宾表示为 20 维概念向量：

```
Brian Chesky:  [12.3, 8.5, 15.0, 14.2, 9.8, ...]
Elena Verna:   [9.1, 15.0, 7.2, 10.5, 12.3, ...]
Marty Cagan:   [15.0, 6.3, 11.2, 9.8, 8.5, ...]
```

### Step 4: 相似度计算

使用余弦相似度衡量嘉宾间的关联：

```python
def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = sqrt(sum(a * a for a in v1))
    mag2 = sqrt(sum(b * b for b in v2))
    return dot / (mag1 * mag2)
```

**相似度阈值**: > 0.8 才建立连接

### Step 5: K-means 聚类

```python
from sklearn.cluster import KMeans

# 8 个聚类
kmeans = KMeans(n_clusters=8, random_state=42)
clusters = kmeans.fit_predict(concept_vectors)
```

**聚类结果**:

| 聚类 | 名称 | 核心概念 | 嘉宾数 |
|------|------|----------|--------|
| 0 | 领导力与愿景 | leadership, strategy | 7 |
| 1 | 增长与获客 | growth, marketing | 8 |
| 2 | 产品管理 | product, execution | 202 |
| 3 | 决策与心智 | decision, culture | 4 |
| 4 | 创新与技术 | innovation, engineering | 15 |
| 5 | 创业与融资 | founder, startup | 12 |
| 6 | 职业发展 | career, hiring | 38 |
| 7 | 用户研究 | customer, feedback | 11 |

### Step 6: 网络构建

```python
edges = []
for i, guest1 in enumerate(guests):
    for j, guest2 in enumerate(guests):
        if i >= j:
            continue
        sim = cosine_similarity(guest1.vector, guest2.vector)
        if sim > 0.8:
            edges.append({
                'source': guest1.name,
                'target': guest2.name,
                'weight': sim
            })

# 限制边数量，保留最强连接
edges = sorted(edges, key=lambda e: -e['weight'])[:2000]
```

**网络统计**:
- 节点数: 297
- 边数: 2000
- 平均度: 13.5
- 聚类系数: 0.72

---

## 核心算法

### 力导向布局

模拟物理系统，让图谱自动布局：

```javascript
function simulate() {
    nodes.forEach(n1 => {
        let fx = 0, fy = 0;
        
        // 1. 斥力：节点互相排斥
        nodes.forEach(n2 => {
            if (n1 === n2) return;
            const dist = distance(n1, n2);
            const force = REPULSION / (dist * dist);
            fx += (n1.x - n2.x) / dist * force;
            fy += (n1.y - n2.y) / dist * force;
        });
        
        // 2. 引力：连接的节点互相吸引
        edges.filter(e => e.source === n1.id).forEach(e => {
            const n2 = getNode(e.target);
            const dist = distance(n1, n2);
            const force = dist * ATTRACTION * e.weight;
            fx += (n2.x - n1.x) / dist * force;
            fy += (n2.y - n1.y) / dist * force;
        });
        
        // 3. 中心引力：防止飘散
        fx -= n1.x * CENTER_GRAVITY;
        fy -= n1.y * CENTER_GRAVITY;
        
        // 应用力
        n1.vx = (n1.vx + fx) * DAMPING;
        n1.vy = (n1.vy + fy) * DAMPING;
        n1.x += n1.vx;
        n1.y += n1.vy;
    });
}
```

**参数调优**:

| 参数 | 值 | 作用 |
|------|------|------|
| REPULSION | 800 | 节点间斥力强度 |
| ATTRACTION | 0.03 | 边的吸引力强度 |
| CENTER_GRAVITY | 0.01 | 中心引力强度 |
| DAMPING | 0.6 | 速度衰减系数 |

### 初始布局优化

为加速收敛，初始位置按聚类分布：

```javascript
nodes.forEach(node => {
    const angle = (node.cluster / 8) * 2 * Math.PI;
    const radius = 150 + Math.random() * 200;
    node.x = Math.cos(angle) * radius;
    node.y = Math.sin(angle) * radius;
});
```

---

## 可视化实现

### Canvas 渲染

使用 HTML5 Canvas 实现高性能渲染：

```javascript
function render() {
    ctx.clearRect(0, 0, width, height);
    
    // 绘制边
    edges.forEach(edge => {
        ctx.beginPath();
        ctx.moveTo(edge.source.x, edge.source.y);
        ctx.lineTo(edge.target.x, edge.target.y);
        ctx.strokeStyle = `rgba(124, 58, 237, ${edge.weight * 0.3})`;
        ctx.stroke();
    });
    
    // 绘制节点
    nodes.forEach(node => {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = CLUSTER_COLORS[node.cluster];
        ctx.fill();
    });
}
```

### 交互设计

| 交互 | 实现 |
|------|------|
| 拖拽节点 | mousedown + mousemove 更新节点位置 |
| 平移画布 | 拖拽空白区域更新 offset |
| 缩放 | wheel 事件更新 scale |
| 悬停提示 | mousemove 检测节点，显示 tooltip |
| 点击详情 | click 事件更新右侧面板 |

---

## 如何复现

### 环境准备

```bash
# Python 3.8+
pip install numpy scikit-learn
```

### 运行分析

```python
# analysis/analyze.py
import os
import json
from sklearn.cluster import KMeans
from collections import defaultdict

# 1. 读取文字稿
transcripts = {}
for fname in os.listdir('transcripts/'):
    if fname.endswith('.txt'):
        guest = fname[:-4]
        with open(f'transcripts/{fname}', 'r') as f:
            transcripts[guest] = f.read().lower()

# 2. 提取概念
concepts = ['product', 'growth', 'leadership', ...]
vectors = []
for guest, text in transcripts.items():
    vector = [text.count(c) / 50 for c in concepts]
    vectors.append(vector)

# 3. 聚类
kmeans = KMeans(n_clusters=8)
clusters = kmeans.fit_predict(vectors)

# 4. 构建边
edges = []
for i, (g1, v1) in enumerate(zip(transcripts.keys(), vectors)):
    for j, (g2, v2) in enumerate(zip(transcripts.keys(), vectors)):
        if i >= j:
            continue
        sim = cosine_sim(v1, v2)
        if sim > 0.8:
            edges.append({'source': g1, 'target': g2, 'weight': sim})

# 5. 输出
with open('graph_data.json', 'w') as f:
    json.dump({'nodes': nodes, 'edges': edges}, f)
```

### 自定义参数

```python
# 调整聚类数
N_CLUSTERS = 8  # 可改为 5-12

# 调整相似度阈值
SIMILARITY_THRESHOLD = 0.8  # 降低会增加边数

# 调整概念列表
CONCEPTS = ['product', 'growth', ...]  # 可自定义
```

---

## 局限性

### 数据层面
- 仅基于文字稿，未考虑语音语调
- 英文为主，概念提取可能有偏差
- 部分嘉宾只出现一次，数据量有限

### 方法层面
- 概念列表人工定义，可能遗漏重要维度
- K-means 对初始值敏感，结果可能有变化
- 相似度阈值人工设定，可能影响图谱结构

### 改进方向
- 使用 BERT 等预训练模型提取更丰富的语义
- 引入时间维度，分析话题演变
- 添加情感分析，识别嘉宾态度
- 构建知识图谱，提取实体关系

---

## 参考资料

- [Force-Directed Graph Drawing](https://en.wikipedia.org/wiki/Force-directed_graph_drawing)
- [K-means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

---

*如有问题，欢迎提交 Issue 讨论！*
