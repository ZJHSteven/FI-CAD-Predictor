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
2. 使用 `setup_env.bat`（或手动执行 `uv sync`）准备环境
3. 运行 `run_project.bat`（等价于 `uv run python main.py`）执行完整流程
4. 查看 `output/` 目录下的结果

## API 服务（FastAPI）
本项目已提供一个最小可用的推理API，方便前端或其他系统调用。

### 启动方式
1. 先安装依赖（或双击 `setup_env.bat`）
2. 双击运行 `run_api.bat`（等价于 `uv run uvicorn ...`）
3. 默认地址：`http://127.0.0.1:8000`

### 常用接口
- `GET /health`：健康检查，返回模型数量与权重
- `GET /models`：返回当前参与集成的模型列表与权重
- `GET /features`：返回特征列表与默认值（便于前端生成表单）
- `POST /predict`：提交特征并返回预测结果

### 预测请求示例
```json
{
  "FI": 0.25,
  "Age": 65,
  "Gender": 1,
  "return_viz": true
}
```

### 预测返回示例
```json
{
  "label": 1,
  "probability": 0.78,
  "model_details": {
    "LogisticRegression": 0.76,
    "RidgeClassifier": 0.80
  },
  "used_models": {
    "LogisticRegression": 0.51,
    "RidgeClassifier": 0.49
  },
  "figures": [
    "/static/figures/correlation_heatmap.png",
    "/static/figures/pycaret/pycaret_logisticregression_roc.png"
  ]
}
```

### 静态图表访问
所有 `output/figures` 下的图表会挂载到 `/static`。  
例如：`output/figures/correlation_heatmap.png` → `/static/figures/correlation_heatmap.png`

## 前端最小测试页（React + TS）
项目内置了一个最小测试页面，便于人工验证API。

### 启动方式
1. 进入前端目录：`cd frontend`
2. 安装依赖：`npm install`
3. 启动开发服务器：`npm run dev`
4. 浏览器打开控制台给出的本地地址（通常是 `http://127.0.0.1:5173`）

### 使用说明
- 默认API地址为 `http://127.0.0.1:8000`
- 先启动后端API，再启动前端页面

## 模型说明
项目支持多种模型：
- Logistic回归
- XGBoost
- 随机森林
- 多层感知机(MLP)

每个模型可以通过 `configs/model_config.yaml` 进行参数配置。
