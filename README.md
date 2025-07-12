# 基于 CNN 与 LSTM 的结构化字符识别与预测建模

## 📚 Overview | 项目概述

This project implements and analyzes multiple neural architectures including MLP, CNN, and RNN/LSTM to handle two types of tasks:  
(1) **Handwritten character image classification** using KMNIST dataset.  
(2) **Structured sequence modeling and prediction**, focusing on formal grammars such as `anbn`, `anbncn`, and Reber Grammar.  
The project explores how different architectures encode features, tracks hidden state transitions, and visualizes decision boundaries and predictions.  

本项目实现并分析了多种神经网络结构，包括 MLP、CNN 和 RNN/LSTM，分别用于两个任务：  
（1）使用 KMNIST 数据集进行**日文手写图像分类**；  
（2）模拟结构化语法（如 anbn、anbncn 和 Reber Grammar）进行**字符序列建模与预测**。  
项目重点探索不同模型对特征的编码方式，追踪隐藏状态变化，并通过可视化手段展示预测边界与动态。
---

## 📁 项目结构

```bash
├── NetLin / NetFull / NetConv # 图像分类网络模型（KMNIST 数据集）
├── check.py / check_main.py # MLP 手动设权训练与激活图可视化
├── anbn.py # anbn 序列数据生成器
├── RNN + LSTM + 可视化输出 # 序列结构学习与可解释性分析
├── plot/ # 保存可视化结果（激活图、输出图、轨迹图等）
├── hw1.pdf # 最终报告及实验总结
```
---
---

## 🔧 Components | 模块说明

### Part 1: KMNIST Image Classification | 图像分类
- Implemented three models: `NetLin` (linear), `NetFull` (1 hidden layer), and `NetConv` (CNN).
- Accuracy results:  
  - NetLin: 70%  
  - NetFull: 85%  
  - NetConv: **94%**
- Tools: `PyTorch`, `CrossEntropyLoss`, `ConfusionMatrix`, visualization of misclassified samples.

### Part 2: MLP Manual Design & Visualization | MLP 网络可视化与手动权重设计
- Built an MLP with manually defined weights (step/sigmoid activation).
- Visualized hidden unit activations (`hid_X_Y.jpg`) and output boundary (`out_X.jpg`).
- Achieved **100% accuracy** on 2D binary classification task.

### Part 3: RNN/LSTM for Grammar Learning | 递归网络语法建模
- Custom sequence generator for `anbn` and `anbncn` structures.
- Trained RNN/LSTM to track symbol order and predict next character.
- Hidden state trajectories visualized to show segmented counting and state reset mechanism.
- Applied LSTM on Reber Grammar to demonstrate **superior long-range dependency modeling**.

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

```bash
├── NetLin / NetFull / NetConv # CNN/MLP models for KMNIST classification
├── check.py / check_main.py # Manually weighted MLP + hidden unit visualization
├── anbn.py # Sequence generator for structured grammar
├── RNN + LSTM + visual outputs # Sequence modeling and interpretability tools
├── plot/ # Visual outputs (activation maps, trajectories, etc.)
├── hw1.pdf # Final report and analysis
```

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

