# FI预测CAD项目

## 项目概述
本项目旨在通过多种机器学习方法预测冠心病(CAD)风险，使用虚弱指数(FI)和其他健康指标作为特征。

## 数据说明
- 2011年协变量.csv: 包含人口统计学数据、吸烟饮酒习惯、血检等信息
- 2011年FI+CVD及变量.csv: 包含FI(虚弱指数)和CVD(心血管疾病)信息

## 项目结构
```
project/
│
├── configs/
│   ├── variables.yaml        # 变量筛选结果配置
│   ├── paths.yaml            # 数据路径、模型存储路径
│   ├── model_config.yaml     # 模型参数（XGBoost, RF, MLP等）
│   └── viz_config.yaml       # 可视化相关的设置
│
├── data/
│   ├── raw/                  # 原始数据
│   ├── cleaned/              # 清洗后数据
│   └── selected_features/    # 筛选变量后数据
│
├── src/
│   ├── utils/
│   │   └── config_loader.py  # 加载配置文件
│   ├── preprocessing.py      # 数据清洗与预处理
│   ├── feature_selection.py  # 特征选择逻辑
│   ├── train_model.py        # 模型训练代码
│   ├── evaluate_model.py     # 评估指标输出
│   └── visualize.py          # 统一格式化 SHAP / ROC / Bar图
│
├── notebooks/                # 可选：探索性分析笔记本
├── output/                   # 所有模型结果、图像等
└── main.py                   # 主调度文件
```

## 使用方法
1. 配置 `configs/paths.yaml` 中的数据路径
2. 运行 `main.py` 执行完整的数据处理和模型训练流程
3. 查看 `output/` 目录下的结果

## 模型说明
项目支持多种模型：
- Logistic回归
- XGBoost
- 随机森林
- 多层感知机(MLP)

每个模型可以通过 `configs/model_config.yaml` 进行参数配置。