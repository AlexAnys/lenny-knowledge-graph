# Lenny's Podcast 知识宇宙 🌐

基于 297 期 Lenny's Podcast 完整文字稿的交互式知识图谱可视化与深度分析。

![Guests](https://img.shields.io/badge/Guests-297-purple)
![Connections](https://img.shields.io/badge/Connections-2000-blue)
![Clusters](https://img.shields.io/badge/Clusters-8-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🔗 在线演示

**[👉 点击访问知识图谱 → https://lenny-knowledge-graph.web.app/](https://lenny-knowledge-graph.web.app/)**



## 📖 项目背景

[Lenny Rachitsky](https://www.lennysnewsletter.com/) 是前 Airbnb 增长产品负责人，他的播客汇聚了科技行业最顶尖的思想者——从 Brian Chesky (Airbnb) 到 Tobi Lutke (Shopify)，从 Marc Benioff (Salesforce) 到 Dylan Field (Figma)。

本项目对 **297 期播客的完整文字稿（2300万+字）** 进行了系统性分析，提取跨越多位嘉宾、经过实战验证的核心模式和心智模型。

## ✨ 功能特点

### 📊 交互式知识图谱
- **力导向布局** - 类似 Obsidian 的动态图谱
- **拖拽交互** - 拖拽节点时触发物理模拟，松开后自动稳定
- **8 大主题聚类** - 基于 K-means 算法的自动分类
- **缩放与平移** - 鼠标滚轮缩放，拖拽画布平移
- **智能搜索** - 快速定位任意嘉宾
- **聚类过滤** - 按主题筛选显示

### 📄 综合报告
从海量文稿中提炼的 **10 大颠覆性洞见**，每条包含：
- 原文引用
- 深度解读
- 来源嘉宾
- 行动建议

### 💡 核心洞见
按 5 大主题分类的 25+ 条精选洞见：
- 👑 领导力与战略
- 📈 增长与获客
- 🎯 产品管理
- 🧠 决策与心智
- 👥 用户与职业

### 🔍 发现模式
- **聚类分析** - 8 个主题聚类详解
- **桥梁专家** - 跨领域连接的关键人物
- **网络中心性** - 最具影响力的节点分析
- **高频框架** - PMF, OKR, JTBD 等方法论统计
- **反常识洞见** - 挑战行业"常识"的观点汇总

## 🔬 分析方法

详细的技术方法请参阅 **[METHODOLOGY.md](METHODOLOGY.md)**

### 概述

```
原始数据 (297份文字稿, 23M+字)
    ↓
文本预处理 (清洗、分词)
    ↓
特征提取 (TF-IDF概念向量)
    ↓
相似度计算 (余弦相似度)
    ↓
聚类分析 (K-means, k=8)
    ↓
网络构建 (阈值过滤)
    ↓
布局算法 (力导向模拟)
    ↓
可视化呈现
```

### 核心技术

| 技术 | 用途 | 说明 |
|------|------|------|
| TF-IDF | 概念提取 | 识别每位嘉宾的核心话题 |
| 余弦相似度 | 关联计算 | 衡量嘉宾间的主题相似性 |
| K-means | 聚类分析 | 将嘉宾分为8个主题群组 |
| 力导向算法 | 图谱布局 | 生成美观的节点分布 |
| Canvas | 可视化 | 高性能图形渲染 |

## 🚀 快速开始

### 在线访问
直接访问：https://alexanys.github.io/lenny-knowledge-graph

### 本地运行

```bash
# 克隆仓库
git clone https://github.com/AlexAnys/lenny-knowledge-graph.git
cd lenny-knowledge-graph

# 启动本地服务器（任选一种）
python -m http.server 8000
# 或
npx serve .
# 或
php -S localhost:8000

# 访问
open http://localhost:8000
```

### 自定义分析

如果你想用自己的数据进行类似分析：

```bash
# 安装依赖
pip install numpy scikit-learn

# 运行分析脚本
python analysis/analyze.py --input your_transcripts/ --output graph_data.json
```

详见 [METHODOLOGY.md](METHODOLOGY.md) 中的完整指南。

## 📁 项目结构

```
lenny-knowledge-graph/
├── index.html              # 主应用（单文件，可独立运行）
├── README.md               # 项目说明
├── METHODOLOGY.md          # 详细分析方法
├── LICENSE                 # MIT 许可证
├── docs/
│   └── screenshot.png      # 截图
└── analysis/
    ├── analyze.py          # 数据分析脚本
    └── requirements.txt    # Python 依赖
```

## 🎯 十大核心洞见预览

1. **初学者心智比经验更重要** - Marc Benioff, Tobi Lutke
2. **创始人必须亲自做增长** - Elena Verna
3. **没有真正的"病毒式增长"** - Rahul Vohra
4. **清晰度比共识更重要** - Brian Chesky
5. **深入细节不是微管理** - Brian Chesky
6. **Pre-mortem 胜过 Post-mortem** - Shreyas Doshi
7. **90%的产品团队是功能工厂** - Marty Cagan
8. **忽略部分用户反馈是对的** - Rahul Vohra
9. **执行问题往往是战略问题** - Shreyas Doshi
10. **成功没有线性路径** - Marc Benioff

## 🛠 技术栈

- **前端**: 纯原生 HTML/CSS/JavaScript，无框架依赖
- **渲染**: Canvas 2D API
- **算法**: 自定义力导向物理引擎
- **分析**: Python + scikit-learn

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

- 发现问题？[提交 Issue](https://github.com/AlexAnys/lenny-knowledge-graph/issues)
- 想要改进？Fork 后提交 PR

## 📄 许可证

MIT License - 自由使用、修改和分发

## 🙏 致谢

- [Lenny Rachitsky](https://www.lennysnewsletter.com/) - 播客主持人
- 所有 297 位播客嘉宾的智慧贡献
- Claude AI - 分析与可视化实现

---

**如果这个项目对你有帮助，请给个 ⭐ Star！**
