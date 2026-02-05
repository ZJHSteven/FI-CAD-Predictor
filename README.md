# FI/CAD 推理服务（部署版）

## 项目定位
本分支仅保留“模型推理服务”所需的最小文件与代码，用于直接上线后端 API。
训练、评估、可视化生成、数据处理等流程已移除。

## 目录结构（部署所需）
```
.
├── configs/
│   └── api_config.yaml        # 推理配置（特征、默认值、模型清单、权重）
├── output/
│   └── models/
│       └── pycaret/           # 已训练模型（.pkl）
├── src/
│   └── api/                   # FastAPI 推理服务
├── run_api.bat                # Windows 下启动 API
├── setup_env.bat              # Windows 下准备环境
├── pyproject.toml             # 依赖声明
└── uv.lock                    # uv 锁文件（如有更新请同步）
```

## 快速启动（Windows）
1. 双击 `setup_env.bat` 安装 Python 3.10 并同步依赖  
2. 双击 `run_api.bat` 启动服务  
3. 默认地址：`http://127.0.0.1:8000`

## 快速启动（Linux）
```bash
uv python install 3.10
uv sync
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

## 常用接口
- `GET /health`：健康检查，返回模型数量与权重  
- `GET /models`：返回当前参与集成的模型列表与权重  
- `GET /features`：返回特征列表与默认值（便于前端生成表单）  
- `POST /predict`：提交特征并返回预测结果  

## 预测请求示例
```json
{
  "FI": 0.25,
  "Age": 65,
  "Gender": 1,
  "return_viz": false
}
```

## 预测返回示例
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
  "figures": null
}
```

## 配置说明（重点）
`configs/api_config.yaml` 中包含三类核心信息：
1. `features.columns`：**模型特征列顺序**（必须与训练一致）  
2. `features.defaults`：**每个特征的默认值**  
3. `models`：**参与集成的模型清单**（name / path / weight）  

若更换模型或特征，请先更新上述配置，再启动服务。

## 可视化资源（可选）
如果你希望返回图表：
1. 在 `output/figures` 或 `output/figures/pycaret` 中放置 PNG 图片  
2. 调用 `POST /predict` 时传入 `"return_viz": true`  
3. 返回结果中的 `figures` 字段会给出可访问的静态路径（`/static/...`）

