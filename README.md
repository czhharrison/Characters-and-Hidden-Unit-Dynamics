# 基于 CNN 与 LSTM 的结构化字符识别与预测建模

本项目为 UNSW Neural Networks 课程 COMP9444 的课程作业，涵盖图像分类、前馈神经网络可视化、序列结构建模与 LSTM 嵌套语法学习。项目共分为三大部分，通过 PyTorch 实现多个网络结构，结合可视化分析理解模型的内部动态过程。

---

## 📁 项目结构

├── NetLin / NetFull / NetConv # 图像分类网络模型（KMNIST 数据集）
├── check.py / check_main.py # MLP 手动设权训练与激活图可视化
├── anbn.py # anbn 序列数据生成器
├── RNN + LSTM + 可视化输出 # 序列结构学习与可解释性分析
├── plot/ # 保存可视化结果（激活图、输出图、轨迹图等）
├── hw1.pdf # 最终报告及实验总结
---

## 🧠 项目功能与成果概览

### 1. 图像分类（KMNIST）

- 使用三种结构：线性模型（NetLin）、MLP（NetFull）、CNN（NetConv）
- 准确率分别为 **70% / 85% / 94%**
- 实现 PyTorch 自定义模型训练与测试框架

### 2. MLP 可视化与逻辑建模

- 构建具有手动权重设置功能的 MLP 二分类器
- 可视化每个神经元在输入空间中的响应区域（隐藏层激活图）
- 最终准确率达 **100%**

### 3. 序列建模与 LSTM 学习能力

- 使用自定义 anbn、anbncn 序列生成器学习字符计数机制
- 可视化 RNN 的隐藏状态变化与预测输出热力图
- 使用 LSTM 成功学习 Reber Grammar 的嵌套结构，表现优于普通 RNN

---

## 🧰 技术栈

- **框架工具**：PyTorch, NumPy, Matplotlib, argparse
- **模型结构**：SVM, MLP, CNN, RNN, LSTM
- **可视化方法**：隐藏状态轨迹图、激活热图、输出分布图等

---

## 📁 Project Structure

├── NetLin / NetFull / NetConv # CNN/MLP models for KMNIST classification
├── check.py / check_main.py # Manually weighted MLP + hidden unit visualization
├── anbn.py # Sequence generator for structured grammar
├── RNN + LSTM + visual outputs # Sequence modeling and interpretability tools
├── plot/ # Visual outputs (activation maps, trajectories, etc.)
├── hw1.pdf # Final report and analysis


---

## 🧠 Features and Results Overview

### 1. Image Classification (KMNIST)

- Three models: Linear (NetLin), MLP (NetFull), CNN (NetConv)
- Accuracy: **70% / 85% / 94%**
- Fully implemented with PyTorch using custom training loops

### 2. Visualizing MLP Decision Logic

- Designed a 2-layer MLP with manually configured weights
- Visualized activation regions of each hidden neuron in input space
- Achieved **100% binary classification accuracy**

### 3. Sequence Modeling and Grammar Learning

- Built sequence models to learn anbn and anbncn patterns with RNN
- Visualized hidden state dynamics and softmax prediction outputs
- Applied LSTM to learn Reber Grammar with accurate state transitions, outperforming plain RNNs

---

## 🧰 Tech Stack

- **Frameworks**: PyTorch, NumPy, Matplotlib, argparse
- **Models**: SVM, MLP, CNN, RNN, LSTM
- **Visualization**: Hidden state trajectories, activation heatmaps, output distributions

