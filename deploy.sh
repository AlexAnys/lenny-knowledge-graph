#!/bin/bash
# Firebase Hosting 快速部署脚本

# 1. 如果没有安装 firebase-tools，先安装
# npm install -g firebase-tools

# 2. 登录 Firebase
# firebase login

# 3. 创建项目（如果还没有）
# firebase projects:create lenny-knowledge-graph

# 4. 修改 .firebaserc 中的 YOUR_PROJECT_ID 为你的项目ID

# 5. 初始化并部署
# firebase init hosting  # 选择现有项目，public目录选 "."
# firebase deploy --only hosting

echo "部署步骤："
echo "1. npm install -g firebase-tools"
echo "2. firebase login"
echo "3. 编辑 .firebaserc，将 YOUR_PROJECT_ID 改为你的项目ID"
echo "4. firebase deploy --only hosting"
